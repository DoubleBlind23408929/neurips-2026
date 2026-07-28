#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Decode riffusion_latent (VAE latents) stored in HDF5 back to audio.

Expected datasets under {split}/:
- riffusion_latent: float16/float32, shape (N, 4, h, w)    (already scaled by vae_scaling_factor)
Optional (if available):
- riffusion_max_value: float32, shape (N,)

Process:
1) latent -> divide by scaling_factor -> VAE.decode -> pixel_values in [-1, 1]
2) pixel_values -> uint8 RGB image
3) image -> mel amplitudes (inverse of riffusion image_from_spectrogram for mono)
4) InverseMelScale -> linear amplitudes
5) GriffinLim -> waveform
6) save wav (sr from attrs, default 44100)

Note: No phase information; GriffinLim reconstruction is approximate (same as riffusion).
"""

import argparse
import os
import h5py
import numpy as np
import torch
import torchaudio as ta
from diffusers import AutoencoderKL


def image_uint8_from_pixel_values(pixel_values: torch.Tensor) -> np.ndarray:
    """
    pixel_values: (1,3,H,W) in [-1,1]
    returns uint8 RGB image: (H,W,3)
    """
    pv = pixel_values.detach().cpu().float().clamp(-1.0, 1.0)
    img01 = (pv + 1.0) / 2.0  # [0,1]
    img = (img01[0].permute(1, 2, 0).numpy() * 255.0).round()
    img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    return img


def mel_amp_from_riffusion_image_uint8(
    rgb_hw3: np.ndarray,
    power_for_image: float,
    max_value: float,
) -> np.ndarray:
    """
    Inverse of riffusion-hobby image_from_spectrogram() for mono case.

    Forward (mono) was:
      mono = uint8(255 - ( (S/max)^power * 255 ))
      rgb = stack(mono,mono,mono)
      rgb = flipud(rgb)

    Here we undo:
      rgb -> mono -> flipud back -> data = (255 - mono)/255 -> S = (data^(1/power)) * max_value
    """
    if rgb_hw3.dtype != np.uint8 or rgb_hw3.ndim != 3 or rgb_hw3.shape[2] != 3:
        raise ValueError(f"Expected uint8 (H,W,3), got {rgb_hw3.dtype} {rgb_hw3.shape}")

    mono = rgb_hw3[:, :, 0]  # (H,W)
    mono = np.flip(mono, axis=0).copy()  # undo FLIP_TOP_BOTTOM

    data = 255.0 - mono.astype(np.float32)
    data = data / 255.0
    data = np.clip(data, 0.0, 1.0)

    if power_for_image <= 0:
        raise ValueError("power_for_image must be > 0")

    spec_norm = np.power(data, 1.0 / power_for_image).astype(np.float32)
    mel_amp = spec_norm * float(max_value)
    return mel_amp.astype(np.float32)  # (H,W)


def build_inverse_audio_modules(
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


def main():
    ap = argparse.ArgumentParser("Decode riffusion_latent from HDF5 back to audio wav")
    ap.add_argument("--h5", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--model-id", default="riffusion/riffusion-model-v1",
                    help="Diffusers model id (used to load VAE subfolder)")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--out-wav", required=True)

    ap.add_argument("--griffin-iters", type=int, default=32)
    ap.add_argument("--max-value", type=float, default=30000000.0,
                    help="Fallback max_value if riffusion_max_value is not present")
    ap.add_argument("--normalize", action="store_true",
                    help="Peak-normalize audio to [-0.99,0.99]")

    args = ap.parse_args()
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    # ---- load latent + attrs ----
    with h5py.File(args.h5, "r") as f:
        if args.split not in f:
            raise KeyError(f"Split '{args.split}' not found. Available: {list(f.keys())}")
        g = f[args.split]

        if "riffusion_latent" not in g:
            raise KeyError(f"Missing {args.split}/riffusion_latent in H5")

        lat_ds = g["riffusion_latent"]
        n = lat_ds.shape[0]
        if not (0 <= args.index < n):
            raise IndexError(f"--index {args.index} out of range [0, {n-1}]")

        lat = lat_ds[args.index]  # (4,h,w)
        lat = torch.from_numpy(np.array(lat)).unsqueeze(0)  # (1,4,h,w)

        # attrs
        vae_scale = float(lat_ds.attrs.get("vae_scaling_factor", 0.18215))
        sr = int(lat_ds.attrs.get("riff_sr", 44100))
        step_ms = float(lat_ds.attrs.get("step_size_ms", 10.0))
        win_ms = float(lat_ds.attrs.get("window_duration_ms", 100.0))
        pad_ms = float(lat_ds.attrs.get("padded_duration_ms", 400.0))
        n_mels = int(lat_ds.attrs.get("n_mels", 512))
        f_min = float(lat_ds.attrs.get("f_min", 0.0))
        f_max = float(lat_ds.attrs.get("f_max", 10000.0))
        power = float(lat_ds.attrs.get("power_for_image", 0.25))

        # if max_value exists, prefer it
        if "riffusion_max_value" in g and g["riffusion_max_value"].shape[0] > args.index:
            mv = float(g["riffusion_max_value"][args.index])
        else:
            mv = float(args.max_value)

    hop_length = int(round(sr * step_ms / 1000.0))
    win_length = int(round(sr * win_ms / 1000.0))
    n_fft = int(round(sr * pad_ms / 1000.0))

    if not np.isfinite(mv) or mv <= 0.0:
        raise ValueError(f"Invalid max_value={mv}. Provide --max-value or store riffusion_max_value.")

    # ---- load VAE ----
    # Use float16 on CUDA typically, float32 on CPU safer
    vae_dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae = AutoencoderKL.from_pretrained(
        args.model_id,
        subfolder="vae",
        torch_dtype=vae_dtype,
    ).to(device)
    vae.eval()

    # ---- decode latent -> image ----
    lat = lat.to(device=device, dtype=vae_dtype)
    # stored latents are scaled; undo scaling before decode
    lat_unscaled = lat / vae_scale

    with torch.no_grad():
        dec = vae.decode(lat_unscaled).sample  # (1,3,H,W) in [-1,1]

    rgb = image_uint8_from_pixel_values(dec)  # (H,W,3)

    # ---- image -> mel amp ----
    mel_amp = mel_amp_from_riffusion_image_uint8(rgb, power_for_image=power, max_value=mv)  # (H,W)

    # ---- mel amp -> waveform ----
    inv_mel, griffin = build_inverse_audio_modules(
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels if mel_amp.shape[0] == n_mels else mel_amp.shape[0],
        f_min=f_min,
        f_max=f_max,
        griffin_iters=args.griffin_iters,
        device=device,
    )

    mel_t = torch.from_numpy(mel_amp).unsqueeze(0).to(device=device, dtype=torch.float32)  # (1,n_mels,T)

    with torch.no_grad():
        lin_amp = inv_mel(mel_t)      # (1, n_fft//2+1, T)
        wav = griffin(lin_amp)        # (1, N)

    wav = wav.detach().cpu().float()

    if args.normalize:
        peak = wav.abs().max().item()
        if peak > 1e-12:
            wav = wav * (0.99 / peak)

    out_dir = os.path.dirname(args.out_wav)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ta.save(args.out_wav, wav, sample_rate=sr)

    print(f"Saved: {args.out_wav}")
    print(f"index={args.index} sr={sr} n_fft={n_fft} hop={hop_length} win={win_length} "
          f"n_mels={mel_amp.shape[0]} power={power} max_value={mv} vae_scale={vae_scale} "
          f"latent_shape={tuple(lat.shape)} img_shape={rgb.shape}")


if __name__ == "__main__":
    main()