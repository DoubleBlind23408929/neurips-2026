from __future__ import annotations

import os
from glob import glob
from typing import Iterable, List, Tuple

import torch

from ..audio_utils import load_mono, resample_to
from ..registry import DatasetHandler, register_dataset_handler


def _find_musicnet_wavs(root: str, split: str) -> List[str]:
    """
    MusicNet layout:
      root/train_data/*.wav   (split="train")
      root/test_data/*.wav    (split="test")
    """
    split_dir = os.path.join(root, f"{split}_data")
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"MusicNet split dir not found: {split_dir}")
    files = sorted(glob(os.path.join(split_dir, "*.wav")))
    return files


class MusicNetDatasetHandler(DatasetHandler):
    """
    MusicNet dataset handler.
    track_id is the WAV filename stem (e.g. "1727").
    """

    name = "musicnet"

    def find_files(self, root: str, split: str, *, mode: str = "mix") -> Iterable[str]:
        del mode
        return _find_musicnet_wavs(root, split)

    def load_track(
        self,
        path: str,
        target_sr: int,
        *,
        mode: str = "mix",
        musdb_target: str = "mixture",
    ) -> Tuple[torch.Tensor, str, str]:
        del mode, musdb_target
        wav, sr0 = load_mono(path)
        wav = resample_to(wav, sr0, target_sr).to(torch.float32)
        track_id = os.path.splitext(os.path.basename(path))[0]
        return wav, track_id, "mix"


register_dataset_handler(MusicNetDatasetHandler())
