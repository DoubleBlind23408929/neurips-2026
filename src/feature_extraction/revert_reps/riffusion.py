from __future__ import annotations

from typing import Any, Dict, Tuple

import h5py
import numpy as np
import torch
import torchaudio as ta
from diffusers import AutoencoderKL

from .registry import register_revert_handler


def _image_uint8_from_pixel_values(pixel_values: torch.Tensor) -> np.ndarray:
    pv = pixel_values.detach().cpu().float().clamp(-1.0, 1.0)
    img01 = (pv + 1.0) / 2.0
    img = (img01[0].permute(1, 2, 0).numpy() * 255.0).round()
    return np.clip(img, 0.0, 255.0).astype(np.uint8)


def _mel_amp_from_riffusion_image_uint8(
    rgb_hw3: np.ndarray,
    power_for_image: float,
    max_value: float,
) -> np.ndarray:
    if rgb_hw3.dtype != np.uint8 or rgb_hw3.ndim != 3 or rgb_hw3.shape[2] != 3:
        raise ValueError(f"Expected uint8 (H,W,3), got {rgb_hw3.dtype} {rgb_hw3.shape}")
    mono = rgb_hw3[:, :, 0]
    mono = np.flip(mono, axis=0).copy()
    data = 255.0 - mono.astype(np.float32)
    data = np.clip(data / 255.0, 0.0, 1.0)
    if power_for_image <= 0:
        raise ValueError("power_for_image must be > 0")
    spec_norm = np.power(data, 1.0 / power_for_image).astype(np.float32)
    return (spec_norm * float(max_value)).astype(np.float32)


def _build_inverse_modules(
    sr: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    f_min: float,
    f_max: float,
    griffin_iters: int,
    device: torch.device,
):
    inv_mel = ta.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        sample_rate=sr,
        f_min=f_min,
        f_max=f_max,
        norm=None,
        mel_scale="htk",
    ).to(device)
    griffin = ta.transforms.GriffinLim(
        n_fft=n_fft,
        n_iter=griffin_iters,
        win_length=win_length,
        hop_length=hop_length,
        window_fn=torch.hann_window,
        power=1.0,
        momentum=0.99,
        rand_init=True,
    ).to(device)
    return inv_mel, griffin


def _attrs_from_ds(ds: h5py.Dataset) -> Dict[str, Any]:
    sr = int(ds.attrs.get("riff_sr", 44100))
    step_ms = float(ds.attrs.get("step_size_ms", 10.0))
    win_ms = float(ds.attrs.get("window_duration_ms", 100.0))
    pad_ms = float(ds.attrs.get("padded_duration_ms", 400.0))
    n_fft = int(round(sr * pad_ms / 1000.0))
    hop_length = int(round(sr * step_ms / 1000.0))
    win_length = int(round(sr * win_ms / 1000.0))
    return {
        "sr": sr,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": win_length,
        "n_mels": int(ds.attrs.get("n_mels", 512)),
        "f_min": float(ds.attrs.get("f_min", 0.0)),
        "f_max": float(ds.attrs.get("f_max", 10000.0)),
        "power_for_image": float(ds.attrs.get("power_for_image", 0.25)),
        "vae_scaling_factor": float(ds.attrs.get("vae_scaling_factor", 0.18215)),
    }


class RiffusionLatentRevertHandler:
    name = "riffusion_latent"

    def can_handle(self, split_group: h5py.Group) -> bool:
        return "riffusion_latent" in split_group

    def num_items(self, split_group: h5py.Group) -> int:
        return int(split_group["riffusion_latent"].shape[0])

    def init_state(self, split_group: h5py.Group, device: torch.device, args: Any) -> Any:
        ds = split_group["riffusion_latent"]
        attrs = _attrs_from_ds(ds)
        inv_mel, griffin = _build_inverse_modules(
            sr=attrs["sr"],
            n_fft=attrs["n_fft"],
            hop_length=attrs["hop_length"],
            win_length=attrs["win_length"],
            n_mels=attrs["n_mels"],
            f_min=attrs["f_min"],
            f_max=attrs["f_max"],
            griffin_iters=int(getattr(args, "griffin_iters", 32)),
            device=device,
        )
        model_id = getattr(args, "riffusion_model_id", "riffusion/riffusion-model-v1")
        vae_dtype = torch.float16 if device.type == "cuda" else torch.float32
        vae = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=vae_dtype,
        ).to(device)
        vae.eval()
        return {
            "ds": ds,
            "attrs": attrs,
            "inv_mel": inv_mel,
            "griffin": griffin,
            "vae": vae,
            "vae_dtype": vae_dtype,
        }

    def decode_index(
        self,
        split_group: h5py.Group,
        index: int,
        state: Any,
        device: torch.device,
        args: Any,
    ) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        ds = state["ds"]
        attrs = state["attrs"]
        lat = torch.from_numpy(np.asarray(ds[index])).unsqueeze(0).to(device=device, dtype=state["vae_dtype"])
        lat_unscaled = lat / float(attrs["vae_scaling_factor"])
        with torch.no_grad():
            dec = state["vae"].decode(lat_unscaled).sample
        rgb = _image_uint8_from_pixel_values(dec)

        if "riffusion_max_value" in split_group and split_group["riffusion_max_value"].shape[0] > index:
            mv = float(split_group["riffusion_max_value"][index])
        else:
            mv = float(getattr(args, "fallback_max_value", 30000000.0))
        if not np.isfinite(mv) or mv <= 0.0:
            raise ValueError(f"Invalid max_value={mv}.")

        mel_amp = _mel_amp_from_riffusion_image_uint8(
            rgb,
            power_for_image=float(attrs["power_for_image"]),
            max_value=mv,
        )
        mel_t = torch.from_numpy(mel_amp).unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            lin_amp = state["inv_mel"](mel_t)
            wav = state["griffin"](lin_amp)
        wav = wav.detach().cpu().to(torch.float32)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        return wav, int(attrs["sr"]), {"max_value": mv, "latent_shape": tuple(lat.shape)}


register_revert_handler(RiffusionLatentRevertHandler())

