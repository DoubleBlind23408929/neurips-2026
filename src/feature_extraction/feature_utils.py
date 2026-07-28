from __future__ import annotations

"""
Feature-space utilities shared across extractors:
- canonicalize mel / generic (C,T) features
- riffusion spectrogram-to-image helpers
"""

from typing import Tuple

import numpy as np
import torch
from PIL import Image


def canonicalize_mel_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize pipeline output to (n_mels, T) from shapes like:
      (n_mels, T) | (1, n_mels, T) | (1, 1, n_mels, T) | ...
    """
    if x.dim() < 2:
        raise RuntimeError(f"Unexpected mel dim={x.dim()}, need at least 2.")
    n_mels, T = x.shape[-2], x.shape[-1]
    return x.reshape(-1, n_mels, T)[0]


def canonicalize_feat_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize to (C, T).
    Accepts:
      (C,T) | (1,C,T) | (B,C,T) -> pick first
      (T,C) | (1,T,C) | (B,T,C) -> transpose to (C,T)
    """
    if x.dim() < 2:
        raise RuntimeError(f"Unexpected feat dim={x.dim()}, need at least 2.")

    if x.dim() == 3:
        x = x[0]

    a, b = x.shape
    if b in (64, 128, 256, 512, 768, 1024, 1280) and a not in (64, 128, 256, 512, 768, 1024, 1280):
        x = x.transpose(0, 1)
    return x.contiguous()


def riffusion_image_from_mel_amplitudes(
    mel_amp: torch.Tensor,  # (1, 512, T) on device
    power_for_image: float = 0.25,
) -> tuple[np.ndarray, float]:
    """
    Exact mono encoding like riffusion-hobby image_from_spectrogram().
    Returns:
      rgb: uint8 (H=512, W=T, 3) AFTER FLIP_TOP_BOTTOM
      max_value: float
    """
    mel_np = mel_amp.detach().cpu().numpy().astype(np.float32)  # (1,512,T)
    mv = float(np.max(mel_np))
    if (not np.isfinite(mv)) or (mv <= 0.0):
        return None, mv  # caller decides to skip
    data = mel_np / mv
    data = np.power(data, power_for_image)
    data = data * 255.0
    data = 255.0 - data
    data = data.astype(np.uint8)  # (1,512,T)
    mono = data[0]  # (512,T)
    rgb = np.stack([mono, mono, mono], axis=2)  # (512,T,3)
    rgb = np.flip(rgb, axis=0).copy()  # FLIP_TOP_BOTTOM; copy fixes negative strides
    return rgb, mv


def riffusion_preprocess_image_like_official(rgb_hw3: np.ndarray) -> torch.Tensor:
    """
    Mimic riffusion-hobby preprocess_image():
      - truncate (w,h) to multiples of 32 via (x - x % 32)
      - resize (LANCZOS) to (w,h)
      - return tensor in [0,1] as (3,h,w)
    """
    if rgb_hw3.dtype != np.uint8 or rgb_hw3.ndim != 3 or rgb_hw3.shape[2] != 3:
        raise ValueError(f"Expected uint8 (H,W,3), got {rgb_hw3.dtype} {rgb_hw3.shape}")

    im = Image.fromarray(rgb_hw3, mode="RGB")
    w, h = im.size
    w2 = w - (w % 32)
    h2 = h - (h % 32)
    # official code does resize to these dims
    if (w2, h2) != (w, h):
        im = im.resize((w2, h2), resample=Image.LANCZOS)

    arr = np.array(im, dtype=np.float32) / 255.0  # (h,w,3)
    ten = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,h,w)
    return ten

