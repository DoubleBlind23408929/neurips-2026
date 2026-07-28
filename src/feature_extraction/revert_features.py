#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Structured feature-to-audio revert tool.

Supported reversible reps are registered in `src.model_tools.revert_reps`.
Current built-ins:
- mel
- riffusion_latent
"""

from __future__ import annotations

import argparse
import os
from typing import List

import h5py
import torch
import torchaudio as ta

from .revert_reps.registry import get_revert_handler, list_revert_handlers
from . import revert_reps  # noqa: F401  # ensure built-ins are registered


def _decode_uid(uid_obj) -> str:
    if isinstance(uid_obj, bytes):
        return uid_obj.decode("utf-8", errors="ignore")
    return str(uid_obj)


def _resolve_indices(total: int, index: int, start: int, count: int) -> List[int]:
    if index >= 0:
        if not (0 <= index < total):
            raise IndexError(f"--index {index} out of range [0, {total - 1}]")
        return [index]
    if start < 0:
        raise ValueError("--start must be >= 0")
    if start >= total:
        return []
    end = total if count < 0 else min(total, start + count)
    return list(range(start, end))


def main() -> None:
    ap = argparse.ArgumentParser("Revert reversible features from HDF5 to audio wav")
    ap.add_argument("--h5", required=True, help="Input feature H5")
    ap.add_argument("--split", default="train", help="Split/group in H5")
    ap.add_argument("--rep", default="auto", help="Rep to revert (auto|mel|riffusion_latent)")
    ap.add_argument("--index", type=int, default=-1, help="Single index to decode; -1 means range")
    ap.add_argument("--start", type=int, default=0, help="Range start when --index=-1")
    ap.add_argument("--count", type=int, default=1, help="Range length; -1 means all")
    ap.add_argument("--out-dir", required=True, help="Directory to write wav files")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--normalize", action="store_true", help="Peak normalize each wav to 0.99")

    # mel-specific
    ap.add_argument("--audioldm2-dir", default=None, help="Local AudioLDM2 path for mel revert")
    ap.add_argument("--default-sr", type=int, default=16000, help="Fallback SR for mel")

    # riffusion-specific
    ap.add_argument("--riffusion-model-id", default="riffusion/riffusion-model-v1")
    ap.add_argument("--griffin-iters", type=int, default=32)
    ap.add_argument("--fallback-max-value", type=float, default=30000000.0)

    ap.add_argument("--list-reps", action="store_true", help="List available reversible reps and exit")
    args = ap.parse_args()

    if args.list_reps:
        print("Available reversible reps:", sorted(list_revert_handlers().keys()))
        return

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    with h5py.File(args.h5, "r") as f:
        if args.split not in f:
            raise KeyError(f"Split '{args.split}' not found. Available: {list(f.keys())}")
        g = f[args.split]

        if args.rep == "auto":
            selected = None
            for name, handler in list_revert_handlers().items():
                if handler.can_handle(g):
                    selected = name
                    break
            if selected is None:
                raise RuntimeError(
                    f"No reversible rep found under split '{args.split}'. "
                    f"Available handlers: {sorted(list_revert_handlers().keys())}"
                )
            rep_name = selected
        else:
            rep_name = args.rep

        handler = get_revert_handler(rep_name)
        if not handler.can_handle(g):
            raise RuntimeError(f"Rep '{rep_name}' is not present in split '{args.split}'.")

        total = handler.num_items(g)
        indices = _resolve_indices(total, args.index, args.start, args.count)
        if not indices:
            print("No indices to process.")
            return

        os.makedirs(args.out_dir, exist_ok=True)
        state = handler.init_state(g, device, args)

        uid_ds = g["uid"] if "uid" in g else None
        for idx in indices:
            wav, sr, meta = handler.decode_index(g, idx, state, device, args)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            wav = wav.detach().cpu().to(torch.float32)
            if args.normalize:
                peak = float(wav.abs().max().item())
                if peak > 1e-12:
                    wav = wav * (0.99 / peak)

            uid = _decode_uid(uid_ds[idx]) if uid_ds is not None else f"{idx:06d}"
            out_wav = os.path.join(args.out_dir, f"{idx:06d}_{uid}_{rep_name}.wav")
            ta.save(out_wav, wav, sample_rate=int(sr))
            print(f"[OK] idx={idx} sr={sr} -> {out_wav} meta={meta}")


if __name__ == "__main__":
    main()

