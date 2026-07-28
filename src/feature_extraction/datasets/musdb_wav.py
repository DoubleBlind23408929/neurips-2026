from __future__ import annotations

import os
from glob import glob
from typing import Iterable, List, Tuple

import torch
import torchaudio as ta

from ..audio_utils import _musdb_select_stem_to_mono, resample_to
from ..registry import DatasetHandler, register_dataset_handler


def _find_musdb_stemwav_files(root: str, split: str) -> List[str]:
    # layout: root/{train,test}/*.stem.wav
    files = glob(os.path.join(root, split, "*.stem.wav"))
    files.sort()
    return files


class MusdbWavDatasetHandler(DatasetHandler):
    """
    MUSDB multi‑channel stem wav exported as .stem.wav (mixture+stems).
    """

    name = "musdb_wav"

    def item_unit(self, mode: str) -> str:
        return "stem" if mode == "stem" else "file"

    def find_files(self, root: str, split: str, *, mode: str = "stem") -> Iterable[str]:
        # mode is ignored; MUSDB uses a single .stem.wav layout.
        return _find_musdb_stemwav_files(root, split)

    def load_track(
        self,
        path: str,
        target_sr: int,
        *,
        mode: str = "stem",
        musdb_target: str = "mixture",
    ) -> Tuple[torch.Tensor, str, str]:
        track_id = os.path.basename(path).replace(".stem.wav", "")
        stem_id = musdb_target

        # For MUSDB we currently rely on torchaudio for multi-channel decode.
        wav_full, sr0 = ta.load(path)
        wav_mono = _musdb_select_stem_to_mono(wav_full, musdb_target)
        wav_mono = resample_to(wav_mono, sr0, target_sr).to(torch.float32)
        return wav_mono, track_id, stem_id


register_dataset_handler(MusdbWavDatasetHandler())

