from __future__ import annotations

import hashlib
import os
from glob import glob
from pathlib import Path
from typing import Iterable, List, Tuple

import torch

from ..audio_utils import load_mono, resample_to
from ..registry import DatasetHandler, register_dataset_handler


def _read_path_list(list_path: Path) -> List[str]:
    base_dir = list_path.parent
    paths: List[str] = []
    with list_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            raw = line.strip()
            if (not raw) or raw.startswith("#"):
                continue
            p = Path(raw)
            if not p.is_absolute():
                p = (base_dir / p).resolve()
            paths.append(str(p))
    return paths


def _resolve_list_file(root: str, split: str) -> Path:
    root_path = Path(root).resolve()
    if root_path.is_file():
        return root_path

    # Directory mode fallback:
    # try common split list naming conventions.
    candidates = [
        root_path / f"{split}.txt",
        root_path / f"{split}.lst",
        root_path / f"{split}.list",
        root_path / f"{split}.tsv",
        root_path / split / "paths.txt",
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()

    # Final fallback: first txt/lst/list file under root.
    loose: List[str] = []
    loose.extend(glob(str(root_path / "*.txt")))
    loose.extend(glob(str(root_path / "*.lst")))
    loose.extend(glob(str(root_path / "*.list")))
    if loose:
        return Path(sorted(loose)[0]).resolve()

    raise FileNotFoundError(
        f"MTG path list file not found from root='{root}', split='{split}'. "
        "Pass --root as a list file path, or place split list (e.g., train.txt) under root."
    )


def _stable_track_id_from_path(path: str) -> str:
    # Keep readable stem while avoiding collisions across same basename.
    stem = os.path.splitext(os.path.basename(path))[0]
    key = os.path.normpath(os.path.abspath(path)).encode("utf-8", errors="ignore")
    short = hashlib.md5(key).hexdigest()[:10]
    return f"{stem}_{short}"


class MtgPathListDatasetHandler(DatasetHandler):
    """
    MTG dataset loader from a path-list file.
    Each non-empty, non-comment line is treated as one audio file path.
    """

    name = "mtg"

    def find_files(self, root: str, split: str, *, mode: str = "stem") -> Iterable[str]:
        del mode
        list_file = _resolve_list_file(root, split)
        files = _read_path_list(list_file)
        if not files:
            raise FileNotFoundError(f"MTG list is empty: {list_file}")
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
        wav, sr0 = load_mono(path)
        wav = resample_to(wav, sr0, target_sr).to(torch.float32)
        track_id = _stable_track_id_from_path(path)
        stem_id = "mix"
        return wav, track_id, stem_id


register_dataset_handler(MtgPathListDatasetHandler())

