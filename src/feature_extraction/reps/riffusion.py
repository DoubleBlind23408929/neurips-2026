"""Riffusion reps: riffusion_img (uint8 image + max_value), riffusion_latent (VAE latents)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import torch
import torchaudio as ta
from diffusers import AutoencoderKL

from ..feature_utils import (
    riffusion_image_from_mel_amplitudes,
    riffusion_preprocess_image_like_official,
)
from ..registry import RepHandler, register_rep_handler

# Riffusion-hobby defaults
RIFF_SR = 44100
RIFF_STEP_MS = 10.0
RIFF_WIN_MS = 100.0
RIFF_PAD_MS = 400.0
RIFF_N_MELS = 512
RIFF_FMIN = 0.0
RIFF_FMAX = 10000.0
RIFF_POWER = 0.25


def _make_spec_mel(device: torch.device):
    hop = int(round(RIFF_SR * RIFF_STEP_MS / 1000.0))
    win_len = int(round(RIFF_SR * RIFF_WIN_MS / 1000.0))
    n_fft = int(round(RIFF_SR * RIFF_PAD_MS / 1000.0))
    spec = ta.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop,
        win_length=win_len,
        pad=0,
        window_fn=torch.hann_window,
        power=None,
        normalized=False,
        wkwargs=None,
        center=True,
        pad_mode="reflect",
        onesided=True,
    ).to(device)
    mel = ta.transforms.MelScale(
        n_mels=RIFF_N_MELS,
        sample_rate=RIFF_SR,
        f_min=RIFF_FMIN,
        f_max=RIFF_FMAX,
        n_stft=n_fft // 2 + 1,
        norm=None,
        mel_scale="htk",
    ).to(device)
    return spec, mel


class RiffusionImgRepHandler:
    name = "riffusion_img"

    def init_models(self, device: torch.device, config: Optional[Dict[str, Any]] = None) -> Any:
        return _make_spec_mel(device)

    def probe_shape(
        self,
        win_samples: int,
        device: torch.device,
        state: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tuple[int, ...], Dict[str, Any]]:
        cfg = config or {}
        target_sr = int(cfg.get("target_sr", 16000))
        duration_sec = win_samples / target_sr
        spec, mel = state
        win_riff = int(round(duration_sec * RIFF_SR))
        probe = (1e-4) * torch.randn(win_riff, device=device)
        xb = probe.unsqueeze(0)
        spec_c = spec(xb)
        amp = spec_c.abs()
        mel_amp = mel(amp)
        rgb, _ = riffusion_image_from_mel_amplitudes(mel_amp, power_for_image=RIFF_POWER)
        if rgb is None:
            raise RuntimeError("Probe produced invalid max_value for riffusion_img.")
        x_img = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        meta = {
            "riff_sr": RIFF_SR,
            "step_size_ms": RIFF_STEP_MS,
            "window_duration_ms": RIFF_WIN_MS,
            "padded_duration_ms": RIFF_PAD_MS,
            "n_fft": int(round(RIFF_SR * RIFF_PAD_MS / 1000.0)),
            "hop_length": int(round(RIFF_SR * RIFF_STEP_MS / 1000.0)),
            "win_length": int(round(RIFF_SR * RIFF_WIN_MS / 1000.0)),
            "n_mels": RIFF_N_MELS,
            "f_min": RIFF_FMIN,
            "f_max": RIFF_FMAX,
            "power_for_image": RIFF_POWER,
        }
        return tuple(x_img.shape), meta

    def create_datasets(
        self,
        grp: h5py.Group,
        shape: Tuple[int, ...],
        meta: Dict[str, Any],
        *,
        np_dtype,
        compression: Optional[str],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, h5py.Dataset]:
        comp = None if (compression in (None, "none")) else compression
        # shape (3, 512, W)
        main = grp.create_dataset(
            "riffusion_img",
            shape=(0, shape[0], shape[1], shape[2]),
            maxshape=(None, shape[0], shape[1], shape[2]),
            dtype=np.uint8,
            chunks=True,
            compression=comp,
        )
        for k, v in meta.items():
            main.attrs[k] = v
        maxv = grp.create_dataset(
            "riffusion_max_value",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float32,
            chunks=True,
        )
        return {"main": main, "max_value": maxv}

    def extract(
        self,
        seg: torch.Tensor,
        *,
        device: torch.device,
        state: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        cfg = config or {}
        target_sr = int(cfg.get("target_sr", 16000))
        spec, mel = state
        if target_sr != RIFF_SR:
            seg_rs = ta.functional.resample(seg.unsqueeze(0), target_sr, RIFF_SR)
        else:
            seg_rs = seg.unsqueeze(0)
        seg_rs = seg_rs.to(device)
        spec_c = spec(seg_rs)
        amp = spec_c.abs()
        mel_amp = mel(amp)
        rgb, mv = riffusion_image_from_mel_amplitudes(mel_amp, power_for_image=RIFF_POWER)
        if rgb is None:
            # Skip this sample (caller will check and continue)
            return None, {}
        feat = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        return feat, {"max_value": mv}


class RiffusionLatentRepHandler:
    name = "riffusion_latent"

    def init_models(self, device: torch.device, config: Optional[Dict[str, Any]] = None) -> Any:
        cfg = config or {}
        vae_dtype_str = cfg.get("riffusion_vae_dtype", "float16")
        vae_dtype = torch.float16 if vae_dtype_str == "float16" else torch.float32
        model_id = cfg.get("riffusion_model_id", "riffusion/riffusion-model-v1")
        spec, mel = _make_spec_mel(device)
        vae = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=vae_dtype,
        ).to(device)
        vae.eval()
        vae_scale = float(getattr(vae.config, "scaling_factor", 0.18215))
        return (spec, mel, vae, vae_scale, vae_dtype)

    def probe_shape(
        self,
        win_samples: int,
        device: torch.device,
        state: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tuple[int, ...], Dict[str, Any]]:
        cfg = config or {}
        target_sr = int(cfg.get("target_sr", 16000))
        duration_sec = win_samples / target_sr
        spec, mel, vae, vae_scale, vae_dtype = state
        win_riff = int(round(duration_sec * RIFF_SR))
        probe = (1e-4) * torch.randn(win_riff, device=device)
        xb = probe.unsqueeze(0)
        spec_c = spec(xb)
        amp = spec_c.abs()
        mel_amp = mel(amp)
        rgb, _ = riffusion_image_from_mel_amplitudes(mel_amp, power_for_image=RIFF_POWER)
        if rgb is None:
            raise RuntimeError("Probe produced invalid max_value for riffusion_latent.")
        ten01 = riffusion_preprocess_image_like_official(rgb)
        pix = (ten01 * 2.0 - 1.0).unsqueeze(0).to(device=device, dtype=vae_dtype)
        lat = vae.encode(pix).latent_dist.mode() * vae_scale
        shape = tuple(lat[0].detach().cpu().shape)
        meta = {
            "riffusion_model_id": cfg.get("riffusion_model_id", "riffusion/riffusion-model-v1"),
            "vae_scaling_factor": vae_scale,
            "preprocess_truncate_to_multiple_of_32": 1,
            "preprocess_resize": "LANCZOS",
            "riff_sr": RIFF_SR,
            "step_size_ms": RIFF_STEP_MS,
            "window_duration_ms": RIFF_WIN_MS,
            "padded_duration_ms": RIFF_PAD_MS,
            "n_mels": RIFF_N_MELS,
            "f_min": RIFF_FMIN,
            "f_max": RIFF_FMAX,
            "power_for_image": RIFF_POWER,
        }
        return shape, meta

    def create_datasets(
        self,
        grp: h5py.Group,
        shape: Tuple[int, ...],
        meta: Dict[str, Any],
        *,
        np_dtype,
        compression: Optional[str],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, h5py.Dataset]:
        comp = None if (compression in (None, "none")) else compression
        main = grp.create_dataset(
            "riffusion_latent",
            shape=(0, shape[0], shape[1], shape[2]),
            maxshape=(None, shape[0], shape[1], shape[2]),
            dtype=np_dtype,
            chunks=True,
            compression=comp,
        )
        for k, v in meta.items():
            main.attrs[k] = v
        return {"main": main}

    def extract(
        self,
        seg: torch.Tensor,
        *,
        device: torch.device,
        state: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        cfg = config or {}
        target_sr = int(cfg.get("target_sr", 16000))
        spec, mel, vae, vae_scale, vae_dtype = state
        if target_sr != RIFF_SR:
            seg_rs = ta.functional.resample(seg.unsqueeze(0), target_sr, RIFF_SR)
        else:
            seg_rs = seg.unsqueeze(0)
        seg_rs = seg_rs.to(device)
        spec_c = spec(seg_rs)
        amp = spec_c.abs()
        mel_amp = mel(amp)
        rgb, _ = riffusion_image_from_mel_amplitudes(mel_amp, power_for_image=RIFF_POWER)
        if rgb is None:
            return None, {}
        ten01 = riffusion_preprocess_image_like_official(rgb)
        pix = (ten01 * 2.0 - 1.0).unsqueeze(0).to(device=device, dtype=vae_dtype)
        lat = vae.encode(pix).latent_dist.mode() * vae_scale
        feat = lat[0].detach().cpu().to(torch.float32)
        return feat, {}


register_rep_handler(RiffusionImgRepHandler())
register_rep_handler(RiffusionLatentRepHandler())
