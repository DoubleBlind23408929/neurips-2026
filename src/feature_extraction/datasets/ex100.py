from __future__ import annotations

import os
from glob import glob
from typing import Iterable, List, Tuple

import torch

from ..audio_utils import load_mono, resample_to
from ..registry import DatasetHandler, register_dataset_handler


def _find_ex100_files(root: str, split: str) -> List[str]:
    """
    Layout: flat directory of .flac files (optionally inside a split subfolder).

    Tries root/split/ first; if that doesn't exist falls back to root/ directly.
    Supported extensions: flac, wav, mp3, ogg, m4a.
    """
    split_dir = os.path.join(root, split)
    search_root = split_dir if os.path.isdir(split_dir) else root

    exts = ("flac", "wav", "mp3", "ogg", "m4a")
    files: List[str] = []
    for ext in exts:
        # non-recursive: files sit directly under search_root
        files.extend(glob(os.path.join(search_root, f"*.{ext}")))
    files.sort()
    return files


class Ex100DatasetHandler(DatasetHandler):
    """
    ex100 dataset: a flat directory of audio files (one file = one track).

    Layout:
        <root>/
            001.flac
            002.flac
            ...

    Or with a split subfolder:
        <root>/train/001.flac  ...
        <root>/test/001.flac   ...
    """

    name = "ex100"

    def find_files(self, root: str, split: str, *, mode: str = "stem") -> Iterable[str]:
        return _find_ex100_files(root, split)

    def load_track(
        self,
        path: str,
        target_sr: int,
        *,
        mode: str = "stem",
        musdb_target: str = "mixture",
    ) -> Tuple[torch.Tensor, str, str]:
        track_id = os.path.splitext(os.path.basename(path))[0]
        stem_id  = "audio"
        wav, sr0 = load_mono(path)
        wav = resample_to(wav, sr0, target_sr).to(torch.float32)
        return wav, track_id, stem_id


register_dataset_handler(Ex100DatasetHandler())
