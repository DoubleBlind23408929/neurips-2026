from __future__ import annotations

import os
import re
from glob import glob
from typing import Iterable, List, Tuple

import torch

from ..audio_utils import load_mono, resample_to
from ..registry import DatasetHandler, register_dataset_handler

_TRACK_ID_RE = re.compile(r"^(\d+)")


def _find_jamendo_files(root: str, split: str) -> List[str]:
    """
    Expected layout:
      root/<split>/<track_id> - <artist> - <title>/no_vocals.wav

    e.g.  root/mdx_q/752621 - LARTSOUND - DIGITAL/no_vocals.wav
    """
    split_dir = os.path.join(root, split)
    search_root = split_dir if os.path.isdir(split_dir) else root
    files = glob(os.path.join(search_root, "*", "no_vocals.wav"))
    files.sort()
    return files


def _track_id_from_path(path: str) -> str:
    """Extract numeric Jamendo track id from the parent directory name.
    e.g. '752621' from '.../752621 - LARTSOUND - DIGITAL/no_vocals.wav'.
    Falls back to the full directory name if no leading digits found."""
    dir_name = os.path.basename(os.path.dirname(path))
    m = _TRACK_ID_RE.match(dir_name)
    return m.group(1) if m else dir_name


class JamendoMaxCapsDatasetHandler(DatasetHandler):
    """
    Jamendo-MaxCaps dataset handler (separated vocals layout).

    Layout::
      root/<split>/<track_id> - <artist> - <title>/no_vocals.wav

    Use --split to select the separator variant (e.g. ``mdx_q``).
    """

    name = "jamendomaxcaps"

    def find_files(self, root: str, split: str, *, mode: str = "stem") -> Iterable[str]:
        del mode
        files = _find_jamendo_files(root, split)
        if not files:
            raise FileNotFoundError(
                f"No no_vocals.wav files found for jamendomaxcaps under "
                f"root='{root}' (split='{split}')"
            )
        return files

    def load_track(
        self,
        path: str,
        target_sr: int,
        *,
        mode: str = "stem",
        musdb_target: str = "mixture",
    ) -> Tuple[torch.Tensor, str, str]:
        del mode, musdb_target
        track_id = _track_id_from_path(path)
        stem_id = "no_vocals"
        wav, sr0 = load_mono(path)
        wav = resample_to(wav, sr0, target_sr).to(torch.float32)
        return wav, track_id, stem_id


register_dataset_handler(JamendoMaxCapsDatasetHandler())
