"""CQT rep — hyperparameters from musicxlab ISMIR2019 large-vocabulary chord recognition.

Internal sr is aligned with MuQ (24000 Hz); hop_length=960 gives 25 fps.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import torch
import torchaudio as ta

from ..registry import register_rep_handler

_CQT_SR = 24000


def _resample(seg: torch.Tensor, target_sr: int, cqt_sr: int) -> np.ndarray:
    if target_sr != cqt_sr:
        seg = ta.functional.resample(seg.unsqueeze(0), target_sr, cqt_sr).squeeze(0)
    return seg.squeeze().cpu().numpy()


def _compute_cqt(audio: np.ndarray, cfg: dict) -> np.ndarray:
    import librosa

    n_bins = int(cfg.get("cqt_n_bins", 288))
    bins_per_octave = int(cfg.get("cqt_bins_per_octave", 36))
    hop_length = int(cfg.get("cqt_hop_length", 960))
    fmin = cfg.get("cqt_fmin") or librosa.note_to_hz("F#0")

    cqt = librosa.hybrid_cqt(
        audio,
        sr=_CQT_SR,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin=fmin,
        hop_length=hop_length,
    )
    # center=True (librosa default) adds padding → T = 1 + floor(N/hop); truncate to floor(N/hop)
    t = len(audio) // hop_length
    return np.abs(cqt[:, :t]).astype(np.float32)  # (n_bins, T)


class CQTRepHandler:
    name = "cqt"

    def init_models(self, device: torch.device, config: Optional[Dict[str, Any]] = None) -> Any:
        return None

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
        dummy = np.zeros(int(round(duration_sec * _CQT_SR)), dtype=np.float32)
        feat = _compute_cqt(dummy, cfg)
        return tuple(feat.shape), {}

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
        c, t = shape[0], shape[1]
        main = grp.create_dataset(
            "cqt",
            shape=(0, c, t),
            maxshape=(None, c, t),
            dtype=np_dtype,
            chunks=True,
            compression=comp,
        )
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
        audio = _resample(seg, target_sr, _CQT_SR)
        feat = _compute_cqt(audio, cfg)
        return torch.from_numpy(feat), {}


register_rep_handler(CQTRepHandler())
