#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug tool: re-extract the first N segments and compare against a reference H5.

For each of the first --n-segments orig entries in the H5:
  1. Locate the original audio file via the dataset handler
  2. Re-extract features with the same rep handler
  3. Compare against what is stored in the H5
  4. Report max-abs-error, mean-abs-error, cosine similarity, and a PASS/FAIL

Usage
-----
python -m src.feature_extraction.verify_features \
    --h5      data/slakh2100/muq_train.h5 \
    --split   train \
    --root    dataset/slakh2100_flac_redux \
    --dataset slakh \
    --rep     muq_layer \
    --mode    mix \
    --n-segments 10
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch

from .audio_utils import load_mono, resample_to
from .registry import get_dataset_handler, get_rep_handler

# auto-register all built-in handlers
from . import datasets  # noqa: F401
from . import reps      # noqa: F401


# ─────────────────────────────────────────────────────────────────
# H5 reading helpers
# ─────────────────────────────────────────────────────────────────

def _decode(arr: np.ndarray) -> np.ndarray:
    if len(arr) > 0 and isinstance(arr.flat[0], (bytes, np.bytes_)):
        return np.array([x.decode("utf-8") for x in arr], dtype=object)
    return arr


def _read_aug_tags(grp: h5py.Group) -> np.ndarray:
    """Per-row aug tags. Prefer numeric `semitone` (current layout); fall back to
    a stored `aug` string column (legacy layout)."""
    if "semitone" in grp:
        return np.array(
            ["orig" if int(s) == 0 else f"ps{int(s):+d}" for s in grp["semitone"][:]],
            dtype=object,
        )
    return _decode(grp["aug"][:])


def _read_meta(grp: h5py.Group, n: int) -> List[Dict[str, Any]]:
    """Read the first n orig entries from an H5 split group."""
    augs      = _read_aug_tags(grp)
    starts    = grp["start"][:]
    ends      = grp["end"][:]
    track_ids = _decode(grp["trackId"][:])
    stem_ids  = _decode(grp["stemId"][:])
    uids      = _decode(grp["uid"][:])

    entries = []
    for i, aug in enumerate(augs):
        if aug != "orig":
            continue
        entries.append({
            "h5_idx":   i,
            "uid":      str(uids[i]),
            "track_id": str(track_ids[i]),
            "stem_id":  str(stem_ids[i]),
            "start":    float(starts[i]),
            "end":      float(ends[i]),
        })
        if len(entries) >= n:
            break
    return entries


# ─────────────────────────────────────────────────────────────────
# Path lookup
# ─────────────────────────────────────────────────────────────────

def _build_file_index_fast(
    root: str,
    split: str,
    dataset: str,
    mode: str,
) -> Dict[tuple, str]:
    """
    Build (track_id, stem_id) → path without loading audio.
    Derives ids from path the same way each dataset handler does.
    """
    ds_handler = get_dataset_handler(dataset)
    files = list(ds_handler.find_files(root, split, mode=mode))

    if dataset == "slakh":
        index: Dict[tuple, str] = {}
        for p in files:
            if mode == "stem":
                track_id = os.path.basename(os.path.dirname(os.path.dirname(p)))
                stem_id  = os.path.splitext(os.path.basename(p))[0]
            else:
                track_id = os.path.basename(os.path.dirname(p))
                stem_id  = "mix"
            index[(track_id, stem_id)] = p
        return index

    # generic fallback: stem_id = "mix" or "audio", track_id = filename stem
    index = {}
    for p in files:
        track_id = os.path.splitext(os.path.basename(p))[0]
        stem_id  = "mix"
        index[(track_id, stem_id)] = p
    return index


# ─────────────────────────────────────────────────────────────────
# Comparison
# ─────────────────────────────────────────────────────────────────

def _compare(ref: np.ndarray, got: np.ndarray) -> Dict[str, float]:
    ref = ref.astype(np.float32)
    got = got.astype(np.float32)
    diff = np.abs(ref - got)
    max_abs   = float(diff.max())
    mean_abs  = float(diff.mean())
    rel_err   = float(diff.mean() / (np.abs(ref).mean() + 1e-8))

    ref_flat = ref.flatten()
    got_flat = got.flatten()
    cosine = float(
        np.dot(ref_flat, got_flat)
        / (np.linalg.norm(ref_flat) * np.linalg.norm(got_flat) + 1e-12)
    )
    return {
        "max_abs_err":  max_abs,
        "mean_abs_err": mean_abs,
        "rel_err":      rel_err,
        "cosine_sim":   cosine,
    }


def _verdict(stats: Dict[str, float], atol: float) -> str:
    if stats["max_abs_err"] <= atol:
        return "PASS ✓"
    return f"FAIL ✗  (max_abs={stats['max_abs_err']:.4e} > atol={atol:.1e})"


# ─────────────────────────────────────────────────────────────────
# Main verification logic
# ─────────────────────────────────────────────────────────────────

def verify(
    h5_path: str,
    split: str,
    root: str,
    dataset: str = "slakh",
    rep: str = "muq_layer",
    mode: str = "mix",
    n_segments: int = 10,
    target_sr: int = 16000,
    atol: float = 1e-3,
    feature_tag: Optional[str] = None,
    # rep model kwargs
    mert_model_id: str = "m-a-p/MERT-v1-330M",
    mert_layer: int = -1,
    muq_model_id: str = "OpenMuQ/MuQ-large-msd-iter",
    muq_layer: int = -1,
    musicfm_model_id: str = "tky823/MusicFM",
    musicfm_layer: int = 7,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[verify] device={device}  h5={h5_path}  split={split}")
    print(f"         dataset={dataset}  rep={rep}  mode={mode}  n={n_segments}")

    # ── 1. Load rep handler ────────────────────────────────────────────────
    rep_handler = get_rep_handler(rep)
    rep_config: Dict[str, Any] = {
        "target_sr":        target_sr,
        "mert_model_id":    mert_model_id,
        "mert_layer":       mert_layer,
        "muq_model_id":     muq_model_id,
        "muq_layer":        muq_layer,
        "musicfm_model_id": musicfm_model_id,
        "musicfm_layer":    musicfm_layer,
        "audioldm2_dir":    os.environ.get("AUDIO_LDM2_DIR"),
        "riffusion_model_id": "riffusion/riffusion-model-v1",
        "riffusion_vae_dtype": "float16",
    }
    print("[verify] Initialising rep model …")
    rep_state = rep_handler.init_models(device, config=rep_config)

    # the feature dataset name inside the H5 group
    ftag = feature_tag or rep

    # ── 2. Read reference metadata ─────────────────────────────────────────
    print("[verify] Reading H5 metadata …")
    with h5py.File(h5_path, "r") as f:
        if split not in f:
            raise KeyError(f"Split '{split}' not in '{h5_path}'. Available: {list(f.keys())}")
        grp = f[split]
        if ftag not in grp:
            raise KeyError(
                f"Feature '{ftag}' not in '{h5_path}/{split}'. Available: {list(grp.keys())}"
            )
        entries = _read_meta(grp, n_segments)
        ref_feats = [grp[ftag][e["h5_idx"]] for e in entries]

    if not entries:
        print("[verify] No orig entries found in H5.")
        return

    print(f"[verify] Found {len(entries)} orig entries to verify.\n")

    # ── 3. Build audio file index ──────────────────────────────────────────
    print("[verify] Building audio file index …")
    file_index = _build_file_index_fast(root, split, dataset, mode)
    print(f"[verify] Indexed {len(file_index)} audio files.\n")

    # ── 4. Re-extract and compare ──────────────────────────────────────────
    pass_count = 0
    fail_count = 0

    with torch.no_grad():
        for i, (entry, ref_np) in enumerate(zip(entries, ref_feats)):
            track_id = entry["track_id"]
            stem_id  = entry["stem_id"]
            start    = entry["start"]
            end      = entry["end"]
            uid      = entry["uid"]

            key = (track_id, stem_id)
            if key not in file_index:
                print(f"  [{i:02d}] {uid}  SKIP — audio file not found for {key}")
                fail_count += 1
                continue

            audio_path = file_index[key]
            wav, sr0 = load_mono(audio_path)
            wav = resample_to(wav, sr0, target_sr).to(torch.float32)

            # extract the exact segment
            s_sample = int(round(start * target_sr))
            e_sample = int(round(end   * target_sr))
            seg = wav[..., s_sample:e_sample].squeeze(0)

            win = e_sample - s_sample
            if seg.shape[-1] < win:
                print(f"  [{i:02d}] {uid}  SKIP — audio too short "
                      f"(got {seg.shape[-1]}, need {win})")
                fail_count += 1
                continue

            feat, _ = rep_handler.extract(seg, device=device, state=rep_state, config=rep_config)
            if feat is None:
                print(f"  [{i:02d}] {uid}  SKIP — rep handler returned None")
                fail_count += 1
                continue

            got_np = feat.detach().cpu().numpy()
            ref_np_f = np.array(ref_np, dtype=np.float32)
            stats = _compare(ref_np_f, got_np)
            verdict = _verdict(stats, atol)

            if "PASS" in verdict:
                pass_count += 1
            else:
                fail_count += 1

            print(
                f"  [{i:02d}] {uid}\n"
                f"        {verdict}\n"
                f"        max_abs={stats['max_abs_err']:.4e}  "
                f"mean_abs={stats['mean_abs_err']:.4e}  "
                f"rel_err={stats['rel_err']:.4e}  "
                f"cosine={stats['cosine_sim']:.6f}\n"
                f"        ref_shape={ref_np_f.shape}  "
                f"got_shape={got_np.shape}  "
                f"ref_range=[{ref_np_f.min():.4f}, {ref_np_f.max():.4f}]"
            )

    # ── 5. Summary ─────────────────────────────────────────────────────────
    total = pass_count + fail_count
    print(f"\n{'='*55}")
    print(f"  Result:  {pass_count}/{total} PASS   {fail_count}/{total} FAIL   atol={atol:.1e}")
    print(f"{'='*55}")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Re-extract features and compare against a reference H5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--h5",      required=True, help="Reference H5 file")
    ap.add_argument("--split",   required=True, help="Split group name in the H5")
    ap.add_argument("--root",    required=True, help="Audio dataset root directory")
    ap.add_argument("--dataset", default="slakh",
                    help="Dataset name (default: slakh)")
    ap.add_argument("--rep",     default="muq_layer",
                    help="Rep handler name (default: muq_layer)")
    ap.add_argument("--mode",    default="mix", choices=["stem", "mix"],
                    help="Slakh mode (default: mix)")
    ap.add_argument("--n-segments", type=int, default=10,
                    help="Number of orig segments to verify (default: 10)")
    ap.add_argument("--target-sr",  type=int, default=16000)
    ap.add_argument("--atol", type=float, default=1e-3,
                    help="Absolute tolerance for PASS/FAIL (default: 1e-3)")
    ap.add_argument("--feature-tag", default=None,
                    help="H5 dataset key for features (default: same as --rep)")
    # model ids
    ap.add_argument("--mert-model-id",    default="m-a-p/MERT-v1-330M")
    ap.add_argument("--mert-layer",       type=int, default=-1)
    ap.add_argument("--muq-model-id",     default="OpenMuQ/MuQ-large-msd-iter")
    ap.add_argument("--muq-layer",        type=int, default=-1)
    ap.add_argument("--musicfm-model-id", default="tky823/MusicFM")
    ap.add_argument("--musicfm-layer",    type=int, default=7)

    args = ap.parse_args()
    verify(
        h5_path=args.h5,
        split=args.split,
        root=args.root,
        dataset=args.dataset,
        rep=args.rep,
        mode=args.mode,
        n_segments=args.n_segments,
        target_sr=args.target_sr,
        atol=args.atol,
        feature_tag=args.feature_tag,
        mert_model_id=args.mert_model_id,
        mert_layer=args.mert_layer,
        muq_model_id=args.muq_model_id,
        muq_layer=args.muq_layer,
        musicfm_model_id=args.musicfm_model_id,
        musicfm_layer=args.musicfm_layer,
    )


if __name__ == "__main__":
    main()
