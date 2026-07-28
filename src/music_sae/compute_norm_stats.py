#!/usr/bin/env python3
"""Compute per-channel mean and std from an H5 file and save as .npz.

Features are stored as (C, T) per sample.  We compute stats over the C
dimension (collapsing N and T), producing mean/std of shape (C,).

Usage:
    python -m music_sae.compute_norm_stats \
        --h5 /path/to/features.h5 \
        --split train \
        --featureTag feat \
        --out /path/to/mert_stats.npz
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True, nargs="+")
    p.add_argument("--split", default="train")
    p.add_argument("--featureTag", default="feat")
    p.add_argument("--out", required=True, help="Output .npz path")
    args = p.parse_args()

    all_sums: np.ndarray | None = None
    all_sq: np.ndarray | None = None
    total_n = 0  # total number of (C,) vectors accumulated = N * T

    for hp in args.h5:
        with h5py.File(hp, "r") as f:
            ds = f[args.split][args.featureTag]
            # ds shape: (N, C, T)
            N, C, T = ds.shape[0], ds.shape[-2], ds.shape[-1]
            s = np.zeros(C, dtype=np.float64)
            sq = np.zeros(C, dtype=np.float64)
            chunk = 128
            for start in range(0, N, chunk):
                end = min(start + chunk, N)
                x = ds[start:end].astype(np.float64)   # (B, C, T)
                # collapse (B, T) → accumulate per channel C
                flat = x.transpose(0, 2, 1).reshape(-1, C)  # (B*T, C)
                s += flat.sum(axis=0)
                sq += (flat * flat).sum(axis=0)
            total_n += N * T

        if all_sums is None:
            all_sums, all_sq = s, sq
        else:
            all_sums += s
            all_sq += sq
        print(f"[compute_norm_stats] processed {hp}  N={N}  C={C}  T={T}")

    mean = all_sums / total_n
    std = np.sqrt(np.maximum((all_sq / total_n) - mean ** 2, 1e-8))
    np.savez(args.out, mean=mean.astype(np.float32), std=std.astype(np.float32))
    print(f"[compute_norm_stats] saved → {args.out}  shape=({len(mean)},)  mean[:4]={mean[:4]}")


if __name__ == "__main__":
    main()
