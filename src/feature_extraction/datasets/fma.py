from __future__ import annotations

import os
from glob import glob
from typing import Iterable, List, Tuple

import torch

from ..audio_utils import load_mono, resample_to
from ..registry import DatasetHandler, register_dataset_handler


def _find_fma_files(root: str, split: str) -> List[str]:
    """
    Expected layout:
      root/{train,validation,test}/XXX/XXXXXX.mp3
    where XXX is 3-digit bucket folder and XXXXXX is track id.
    """
    files = glob(os.path.join(root, split, "**", "*.mp3"), recursive=True)
    files.sort()
    return files


class FmaDatasetHandler(DatasetHandler):
    """
    FMA full tracks (mp3 files in bucket subfolders).
    """

    name = "fma"

    def find_files(self, root: str, split: str, *, mode: str = "stem") -> Iterable[str]:
        # mode/musdb_target not used for FMA; it's always single-track audio.
        return _find_fma_files(root, split)

    def load_track(
        self,
        path: str,
        target_sr: int,
        *,
        mode: str = "stem",
        musdb_target: str = "mixture",
    ) -> Tuple[torch.Tensor, str, str]:
        # Use filename (without extension) as stable track id, e.g. 014000.
        track_id = os.path.splitext(os.path.basename(path))[0]
        stem_id = "mix"
        wav, sr0 = load_mono(path)
        wav = resample_to(wav, sr0, target_sr).to(torch.float32)
        return wav, track_id, stem_id


register_dataset_handler(FmaDatasetHandler())

