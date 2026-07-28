#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified metrics computation for key and chord inference results.
Prints a single JSON line so shell scripts can parse without awk.

Usage:
    # Key accuracy from infer_key.py / probe_key.py output:
    python -m src.analysis.eval_metrics key <result_file>

    # Chord WCSR from infer_key.py chord output:
    python -m src.analysis.eval_metrics chord <result_file> [--fps 10]

Key output JSON keys:
    seg_total, seg_acc, song_total, song_acc

Chord output JSON keys:
    root, majmin, mirex,
    song_root, song_majmin, song_mirex,
    raw_root, raw_majmin, raw_mirex,          (only if --raw-file given)
    raw_song_root, raw_song_majmin, raw_song_mirex
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import re

from ..evaluation.eval_key_from_tsv import (
    NOTE2PC,
    NOTE_NAMES_12,
    load_rows,
    parse_key_signature_major_pc,
)

# Sentinel value written to the ref_key column for a within-segment key change.
# Kept in sync with src.analysis.infer_key.KEY_CHANGE_TAG (duplicated here so
# this metrics module stays free of infer_key's heavy torch/h5py imports).
KEY_CHANGE_TAG = "CHANGE"


def _parse_key_root_mode(label: str) -> Optional[Tuple[int, int]]:
    """Parse a key label to (root_pc, is_minor) WITHOUT merging relative keys.

    Enharmonics are still merged (C# == Db).  Returns None for N / N/A / junk.
    "C:maj" → (0, 0),  "A:min" → (9, 1).
    """
    s = str(label).strip()
    if not s or s.upper() in {"N", "N/A"}:
        return None
    parts = s.split(":", 1)
    root = parts[0].strip()
    mode = parts[1].strip().lower() if len(parts) > 1 else "maj"
    m = re.match(r"^([A-Ga-g])([#b]?)$", root)
    if not m:
        return None
    letter = m.group(1).upper()
    acc = m.group(2)
    if acc == "b":
        acc = "B"
    root_pc = NOTE2PC.get((letter + acc).upper(), None)
    if root_pc is None:
        return None
    return (int(root_pc) % 12, 1 if mode.startswith("min") else 0)


def _mirex_weighted_score(
    ref: Optional[Tuple[int, int]],
    est: Optional[Tuple[int, int]],
) -> float:
    """MIREX audio key-detection weighted score (mir_eval.key.weighted_score).

    ref / est are (tonic_pc, is_minor).  Returns:
        1.0  same key (tonic + mode)
        0.5  perfect fifth (same mode, est tonic = ref tonic +/- 7)
        0.3  relative major/minor (maj→min at +9, min→maj at +3)
        0.2  parallel major/minor (same tonic, opposite mode)
        0.0  otherwise (or missing estimate)
    """
    if ref is None or est is None:
        return 0.0
    r_tonic, r_min = ref
    e_tonic, e_min = est
    if e_tonic == r_tonic and e_min == r_min:
        return 1.0
    if e_min == r_min and ((e_tonic - r_tonic) % 12 in (7, 5)):
        return 0.5
    if r_min == 0 and e_min == 1 and e_tonic == (r_tonic + 9) % 12:
        return 0.3
    if r_min == 1 and e_min == 0 and e_tonic == (r_tonic + 3) % 12:
        return 0.3
    if e_min != r_min and e_tonic == r_tonic:
        return 0.2
    return 0.0


def _decide_majmin_by_degrees(
    maj_pool: List[float],
    min_pool: List[float],
    maj_degree: int,
) -> Tuple[int, int]:
    """Major-vs-minor by I+IV+V activation sums (see infer_key.decide_maj_min_by_degrees).

    maj_pool / min_pool are raw ring activations in chord-root space; maj_degree
    is the maj-side ring degree (0=major chord, 5=forth).  Returns (root_pc,
    is_minor), major wins ties.
    """
    q = max(range(12), key=lambda i: maj_pool[i])
    x = (q - maj_degree + 12) % 12
    p = max(range(12), key=lambda i: min_pool[i])
    xv = (maj_pool[x]
          + max(maj_pool[(x + 5) % 12], min_pool[(x + 5) % 12])
          + max(maj_pool[(x + 7) % 12], min_pool[(x + 7) % 12]))
    pv = (min_pool[p]
          + max(min_pool[(p + 5) % 12], maj_pool[(p + 5) % 12])
          + max(min_pool[(p + 7) % 12], maj_pool[(p + 7) % 12]))
    return (x, 0) if xv >= pv else (p, 1)


def _decide_majmin_3ring(
    maj_pool: List[float],
    min_pool: List[float],
    forth_pool: List[float],
) -> Tuple[int, int]:
    """3-ring major-vs-minor (see infer_key.decide_maj_min_3ring).

    Major hypothesis takes its IV from the forth ring, minor hypothesis uses its
    own ring throughout (all in chord-root space):
      xv = maj[x] + forth[x+5] + maj[x+7]
      pv = min[p] + min[p+5]   + min[p+7]
    Returns (root_pc, is_minor), major wins ties.
    """
    x = max(range(12), key=lambda i: maj_pool[i])
    p = max(range(12), key=lambda i: min_pool[i])
    xv = maj_pool[x] + forth_pool[(x + 5) % 12] + maj_pool[(x + 7) % 12]
    pv = min_pool[p] + min_pool[(p + 5) % 12] + min_pool[(p + 7) % 12]
    return (x, 0) if xv >= pv else (p, 1)


# ── 3-SAE tonic voting (see infer_key.key_from_vote2 / key_from_vote3) ─────────
# means_per_sae is a list of 3 pooled ring vectors (sub-SAE 0/1/2); the candidate
# tonic from sub-SAE i is corrected by +i semitones.

def _ring_vote_cands(means_per_sae, degree, is_minor, offsets=(0, 1, 2)):
    out = []
    for off, m in zip(offsets, means_per_sae):
        a = max(range(12), key=lambda i: m[i])
        root = (a - degree + off + 12) % 12
        out.append((root, 1 if is_minor else 0, m[a]))
    return out


def _tally_vote(cands) -> Tuple[int, int]:
    cnt: Dict[Tuple[int, int], int] = {}
    act: Dict[Tuple[int, int], float] = {}
    for r, md, a in cands:
        k = (r, md)
        cnt[k] = cnt.get(k, 0) + 1
        act[k] = act.get(k, 0.0) + a
    best = max(cnt.values())
    tied = [k for k in cnt if cnt[k] == best]
    win = tied[0] if len(tied) == 1 else max(tied, key=lambda k: act[k])
    return win[0], win[1]


def _key_from_vote2(majside_means, min_means, maj_degree) -> Tuple[int, int]:
    """Returns (tonic_pc, is_minor); major-side + minor ring decide both."""
    return _tally_vote(_ring_vote_cands(majside_means, maj_degree, False)
                       + _ring_vote_cands(min_means, 0, True))


def _sig_vote_cands(means_per_sae, sig_shift, offsets=(0, 1, 2)):
    """Key-signature-space candidates (major sig_shift=0, minor sig_shift=3)."""
    out = []
    for off, m in zip(offsets, means_per_sae):
        a = max(range(12), key=lambda i: m[i])
        out.append(((a + off + sig_shift) % 12, 0, m[a]))
    return out


def _key_from_vote3(maj_means, min_means, forth_means) -> Tuple[int, int]:
    """Returns (tonic_pc, is_minor); forth+min decide mode, major+min signature.

    The signature vote is in key-signature space so major/minor candidates that
    imply the same signature combine (major shifts +0, minor +3).
    """
    _mr, mode_is_min = _tally_vote(_ring_vote_cands(forth_means, 5, False)
                                   + _ring_vote_cands(min_means, 0, True))
    sig_pc, _sd = _tally_vote(_sig_vote_cands(maj_means, 0)
                              + _sig_vote_cands(min_means, 3))
    tonic = sig_pc if mode_is_min == 0 else (sig_pc + 9) % 12
    return tonic, mode_is_min


def _key_from_vote3_avg(maj_means, min_means, forth_means, offsets=(0, 1, 2)) -> Tuple[int, int]:
    """Avg version (see infer_key.key_from_vote3_avg): forth-vs-minor averaged peaks
    decide MODE (major ties); the major-chord ring's averaged argmax is the key
    signature; tonic = signature (maj) or signature+9 (min)."""
    def _avg(vecs):
        n = len(offsets)
        acc = [0.0] * 12
        for k in range(n):
            v = vecs[k]
            s = float(sum(v))
            if s > 0:
                for i in range(12):
                    acc[(i + offsets[k]) % 12] += v[i] / s
        return [a / n for a in acc]

    forth_avg = _avg(forth_means)
    min_avg = _avg(min_means)
    maj_avg = _avg(maj_means)
    is_min = 0 if max(forth_avg) >= max(min_avg) else 1
    sig = max(range(12), key=lambda i: maj_avg[i])
    return (sig if is_min == 0 else (sig + 9) % 12), is_min


def _decide_majmin_multisae(maj_vecs, min_vecs, offsets=(0, 1, 2),
                            maj_degree=0, min_degree=0) -> Tuple[int, int]:
    """Align-average across sub-SAEs, global argmax (see infer_key.decide_maj_min_multisae).

    maj_vecs / min_vecs: 3 raw per-sub-SAE 12-vectors each.  Each is L1-normalized
    and rolled by +offset_k, then averaged into one vector per ring; the ring with
    the larger averaged global argmax wins (major ties).  The peak position maps to
    a tonic via the ring degree: root = (argmax - degree).  Returns (tonic_pc, is_minor).
    """
    def _avg(vecs):
        n = len(offsets)
        acc = [0.0] * 12
        for k in range(n):
            v = vecs[k]
            s = float(sum(v))
            if s > 0:
                for i in range(12):
                    acc[(i + offsets[k]) % 12] += v[i] / s
        return [a / n for a in acc]

    maj_avg = _avg(maj_vecs)
    min_avg = _avg(min_vecs)
    x = max(range(12), key=lambda i: maj_avg[i])
    p = max(range(12), key=lambda i: min_avg[i])
    if maj_avg[x] >= min_avg[p]:
        return (x - maj_degree) % 12, 0
    return (p - min_degree) % 12, 1


# ---------------------------------------------------------------------------
# Joint scoring (major_minor_joint): refined combined-minor template
# ---------------------------------------------------------------------------
# Current default model:
#   1) roll s1/s2 by +1/+2 and raw-average the three sub-SAEs,
#   2) power-compress with power=0.50,
#   3) L1-normalize major and minor rings separately,
#   4) use compact multi-template scoring:
#        Smaj = max(I + W45*IV + W45*V + W26*vi,
#                   I + W45*IV + W45*V + W26*ii)
#        Smin = max(i + W45*iv + W45*v + W26*bVI,
#                   i + W45*iv + W45*V + W26*bVI,
#                   i + W26*ii + W45*IV + W45*V)
#      with W45=0.4, W26=0.1,
#   5) direct 24-class decision.
# ---------------------------------------------------------------------------

def _align_avg_raw(sae_blocks, offsets=(0, 1, 2)) -> List[float]:
    """Roll each RAW sub-SAE block by +offset_k (cross-SAE pitch correction) and
    average — no per-sub-SAE normalization, so ring magnitude is preserved."""
    n = len(offsets)
    acc = [0.0] * 12
    for k in range(n):
        v = sae_blocks[k]
        for i in range(12):
            acc[(i + offsets[k]) % 12] += float(v[i])
    return [a / n for a in acc]


def _joint_norms(maj_blocks, min_blocks, offsets=(0, 1, 2),
                 power=0.50) -> Tuple[List[float], List[float]]:
    """maj/min ring vectors for the refined joint scoring.

    Merge sub-SAEs (shift-average raw), power-compress (x**0.50 by default),
    then L1-normalize each ring SEPARATELY.
    """
    maj = _align_avg_raw(maj_blocks, offsets)
    mn = _align_avg_raw(min_blocks, offsets)
    maj = [max(x, 0.0) ** power for x in maj]
    mn = [max(x, 0.0) ** power for x in mn]
    sm = float(sum(maj))
    sp = float(sum(mn))
    maj = [x / sm for x in maj] if sm > 0 else [0.0] * 12
    mn = [x / sp for x in mn] if sp > 0 else [0.0] * 12
    return maj, mn


def _joint_scores(maj, mn) -> Tuple[List[float], List[float]]:
    """Per-root major/minor scores for the compact multi-template joint model.

    Major: max(I+IV+V+vi, I+IV+V+ii).
    Minor: max(natural, harmonic, melodic) using only major/minor rings.
    """
    W45 = 0.4     # 4/5 support; same weight everywhere
    W26 = 0.1     # 2/6 support; same weight everywhere

    Smaj = [0.0] * 12
    Smin = [0.0] * 12
    for X in range(12):
        maj_base = (
            1.0 * maj[X]
            + W45 * maj[(X + 5) % 12]   # IV
            + W45 * maj[(X + 7) % 12]   # V
        )
        Smaj[X] = max(
            maj_base + W26 * mn[(X + 9) % 12],  # vi
            maj_base + W26 * mn[(X + 2) % 12],  # ii
        )

        smin_nat = (
            1.0 * mn[X]
            + W45 * mn[(X + 5) % 12]    # iv
            + W45 * mn[(X + 7) % 12]    # v
            + W26 * maj[(X + 8) % 12]   # bVI
        )
        smin_harm = (
            1.0 * mn[X]
            + W45 * mn[(X + 5) % 12]    # iv
            + W45 * maj[(X + 7) % 12]   # V
            + W26 * maj[(X + 8) % 12]   # bVI
        )
        smin_melodic = (
            1.0 * mn[X]
            + W26 * mn[(X + 2) % 12]    # ii
            + W45 * maj[(X + 5) % 12]   # IV
            + W45 * maj[(X + 7) % 12]   # V
        )
        Smin[X] = max(smin_nat, smin_harm, smin_melodic)
    return Smaj, Smin


def _joint_decide(Smaj, Smin) -> Tuple[int, int]:
    """Direct 24-class key decision -> (root_pc, is_minor)."""
    scores24 = list(Smaj) + list(Smin)
    idx = max(range(24), key=lambda i: scores24[i])
    if idx < 12:
        return idx, 0
    return idx - 12, 1


def _joint_decide_gt(Smaj, Smin, gt_is_min: int) -> Tuple[int, int]:
    """gt oracle: the root is the argmax score WITHIN the known gt mode.

    gt major → (argmax(Smaj), maj); gt minor → (argmax(Smin), min). No signature-
    first step — the search is confined to the gt mode's own scores.
    """
    if gt_is_min:
        return max(range(12), key=lambda x: Smin[x]), 1
    return max(range(12), key=lambda x: Smaj[x]), 0

# ---------------------------------------------------------------------------
# Key metrics (computed directly from the TSV)
# ---------------------------------------------------------------------------

def compute_key_metrics(result_file: Path, maj_degree: int = 0,
                        vote2_decision: str = "tally") -> Dict:
    rows = load_rows(result_file)

    # Group valid-reference rows by track so we can drop key-changing songs.
    # A CHANGE-tagged ref_key marks a segment whose frames span >1 distinct key
    # (a within-segment modulation, written by ref_key_label_for_segment).  The
    # per-segment majority labels below would hide such a change, so any track
    # with even one CHANGE segment is recorded as key-changing and dropped.
    track_items: Dict[str, List[Tuple[object, int]]] = defaultdict(list)
    changed_tracks: set = set()
    for r in rows:
        if str(r.ref_label).strip().upper() == KEY_CHANGE_TAG:
            changed_tracks.add(r.track_id)
            continue
        ref_sig = parse_key_signature_major_pc(r.ref_label)
        if ref_sig is None:
            continue
        track_items[r.track_id].append((r, int(ref_sig)))

    # Joint files carry a second (minor-side) activation block; when present the
    # song-level prediction pools maj and min separately and compares the peaks.
    # The 3-ring joint file adds a third (forth-ring) block.
    has_min = any(getattr(r, "min_act_12", None) is not None for r in rows)
    has_forth = any(getattr(r, "forth_act_12", None) is not None for r in rows)
    # 3-SAE vote files store per-sub-SAE ring means (maj_sae always; forth_sae for
    # vote3).  The song level pools each sub-SAE's ring mean across segments, then
    # re-votes.
    has_vote2 = any(getattr(r, "maj_sae", None) is not None for r in rows)
    has_vote3 = any(getattr(r, "forth_sae", None) is not None for r in rows)
    # Template scoring (major + minor ring norms → 48 key/mode candidates) drives
    # both segment and song prediction for the joints major_minor_tonic
    # (vote2 layout + --vote2-decision template) and major_forth_minor_tonic
    # (vote3 layout; forth block ignored).  gt files (vote2 layout + 'avg') keep
    # their own decision.  Segment predictions are recomputed here from the cached
    # ring blocks, so the cached est_key column (old logic) is not used.
    # major_minor_joint: power=0.40 combined-minor joint templates,
    # signature-first decision; song level averages the 24 segment scores. joint_gt
    # is the gt oracle: the root is the argmax within the known gt mode.
    use_joint = has_vote2 and vote2_decision in ("joint", "joint_gt")

    # Segment-level accumulators (stable-key songs only).
    seg_total = seg_hit = seg_hit_strict = 0
    seg_maj = seg_min = 0
    seg_hit_maj = seg_hit_min = 0            # merged-correct, split by gt mode
    seg_predmaj_in_maj = seg_predmin_in_min = 0  # predicted-mode matches gt mode
    seg_mirex_sum = 0.0                      # MIREX weighted score, summed
    # Song-level accumulators.
    song_total = song_hit = song_hit_strict = 0
    song_maj = song_min = 0
    song_hit_maj = song_hit_min = 0
    song_predmaj_in_maj = song_predmin_in_min = 0
    song_mirex_sum = 0.0

    for track, items in track_items.items():
        # Within-segment modulation flagged upstream → key-changing, drop song.
        if track in changed_tracks:
            continue
        # Stable-key songs only: a single *full* reference key (root + mode)
        # across segments.  Do not only check key signature: relative-major/minor
        # changes (e.g. C:maj <-> A:min) must also be treated as key-changing.
        ref_rm_unique = {rm for (r, _sig) in items
                         for rm in [_parse_key_root_mode(r.ref_label)]
                         if rm is not None}
        if len(ref_rm_unique) != 1:
            continue
        if len({sig for (_r, sig) in items}) != 1:
            continue
        ref_sig = items[0][1]

        act_sum = [0.0] * 12      # major-side activation pool (or merged, if no min block)
        min_sum = [0.0] * 12      # minor-side activation pool (joint files only)
        forth_sum = [0.0] * 12    # forth-ring activation pool (3-ring joint only)
        # Per-sub-SAE ring pools (3-SAE vote files), 3 blocks of 12 each.
        maj_sae_sum   = [[0.0] * 12 for _ in range(3)]
        min_sae_sum   = [[0.0] * 12 for _ in range(3)]
        forth_sae_sum = [[0.0] * 12 for _ in range(3)]
        est_mode_votes = [0, 0]   # predicted modes [maj, min]
        ref_mode_counts = [0, 0]  # reference modes  [maj, min]
        ref_rm_set = set()
        smaj_sum = [0.0] * 12                                  # joint: summed Smaj over segs
        smin_sum = [0.0] * 12                                  # joint: summed Smin over segs
        joint_nseg = 0

        for (r, _sig) in items:
            # The joint variant recomputes the segment label from the cached ring
            # blocks (the stored est_key column is from the old logic, unused here).
            if use_joint and getattr(r, "maj_sae", None) is not None \
                    and getattr(r, "min_sae", None) is not None:
                maj_n, min_n = _joint_norms(r.maj_sae, r.min_sae)
                Smaj, Smin = _joint_scores(maj_n, min_n)
                if vote2_decision == "joint_gt":
                    gt_rm = _parse_key_root_mode(r.ref_label)
                    j_root, j_min = _joint_decide_gt(
                        Smaj, Smin, gt_rm[1] if gt_rm is not None else 0)
                else:
                    j_root, j_min = _joint_decide(Smaj, Smin)
                est_label = f"{NOTE_NAMES_12[j_root]}:{'min' if j_min else 'maj'}"
                for X in range(12):
                    smaj_sum[X] += Smaj[X]
                    smin_sum[X] += Smin[X]
                joint_nseg += 1
            else:
                est_label = r.est_label
            est_sig = parse_key_signature_major_pc(est_label)
            merged_ok = est_sig is not None and est_sig == ref_sig
            seg_total += 1
            if merged_ok:
                seg_hit += 1

            # ── Strict: predicted (root, mode) must equal reference (root, mode) ──
            ref_rm = _parse_key_root_mode(r.ref_label)
            est_rm = _parse_key_root_mode(est_label)
            if est_rm is not None and ref_rm is not None and est_rm == ref_rm:
                seg_hit_strict += 1
            if ref_rm is not None:
                ref_mode_counts[ref_rm[1]] += 1
                ref_rm_set.add(ref_rm)
                est_is_min = est_rm[1] if est_rm is not None else None
                if ref_rm[1]:                      # gt-minor segment
                    seg_min += 1
                    if merged_ok:
                        seg_hit_min += 1
                    if est_is_min == 1:
                        seg_predmin_in_min += 1
                else:                              # gt-major segment
                    seg_maj += 1
                    if merged_ok:
                        seg_hit_maj += 1
                    if est_is_min == 0:
                        seg_predmaj_in_maj += 1
            if est_rm is not None:
                est_mode_votes[est_rm[1]] += 1
            seg_mirex_sum += _mirex_weighted_score(ref_rm, est_rm)
            for i, v in enumerate(r.key_act_12):
                act_sum[i] += float(v)
            r_min = getattr(r, "min_act_12", None)
            if r_min is not None:
                for i, v in enumerate(r_min):
                    min_sum[i] += float(v)
            r_forth = getattr(r, "forth_act_12", None)
            if r_forth is not None:
                for i, v in enumerate(r_forth):
                    forth_sum[i] += float(v)
            # Per-sub-SAE ring blocks (3-SAE vote files).
            r_maj_sae = getattr(r, "maj_sae", None)
            if r_maj_sae is not None:
                for s in range(3):
                    for i, v in enumerate(r_maj_sae[s]):
                        maj_sae_sum[s][i] += float(v)
            r_min_sae = getattr(r, "min_sae", None)
            if r_min_sae is not None:
                for s in range(3):
                    for i, v in enumerate(r_min_sae[s]):
                        min_sae_sum[s][i] += float(v)
            r_forth_sae = getattr(r, "forth_sae", None)
            if r_forth_sae is not None:
                for s in range(3):
                    for i, v in enumerate(r_forth_sae[s]):
                        forth_sae_sum[s][i] += float(v)

        # ── Song-level prediction (key signature + mode) ──
        song_total += 1
        if use_joint:
            # Average the per-segment 24 scores, then the same signature-first decision
            # (joint_gt forces the mode to the song's gt mode).
            if joint_nseg > 0:
                Smaj_avg = [s / joint_nseg for s in smaj_sum]
                Smin_avg = [s / joint_nseg for s in smin_sum]
                if vote2_decision == "joint_gt":
                    gt_is_min = 1 if ref_mode_counts[1] > ref_mode_counts[0] else 0
                    root_t, pred_is_min = _joint_decide_gt(Smaj_avg, Smin_avg, gt_is_min)
                else:
                    root_t, pred_is_min = _joint_decide(Smaj_avg, Smin_avg)
            else:
                root_t, pred_is_min = 0, 0
            pred = (root_t + 3) % 12 if pred_is_min else root_t
        elif has_vote3:
            # 3-SAE avg, three rings: pool each sub-SAE's ring block across the song's
            # segments, then forth-vs-minor averaged peaks decide mode, major ring
            # averaged argmax the signature (major_forth_minor_tonic).
            tonic_t, pred_is_min = _key_from_vote3_avg(maj_sae_sum, min_sae_sum, forth_sae_sum)
            pred = (tonic_t + 3) % 12 if pred_is_min else tonic_t
        elif has_vote2:
            # 3-SAE, two rings: pool per-sub-SAE blocks across the song, then decide.
            # 'avg' (major_minor_tonic) = L1-normalize, align-average across sub-SAEs
            # over frame-sum blocks, global argmax; 'tally' = majority re-vote.
            if vote2_decision == "avg":
                # maj_degree handles a forth ring stored in the maj block (single
                # 'forth' variant); major_minor_tonic keeps maj_degree=0.
                tonic_t, pred_is_min = _decide_majmin_multisae(
                    maj_sae_sum, min_sae_sum, maj_degree=maj_degree)
            else:
                tonic_t, pred_is_min = _key_from_vote2(maj_sae_sum, min_sae_sum, maj_degree)
            pred = (tonic_t + 3) % 12 if pred_is_min else tonic_t
        elif has_forth:
            # 3-ring joint: major hypothesis takes its IV from the forth ring,
            # minor hypothesis uses its own ring throughout.  pred is the signature.
            root_t, pred_is_min = _decide_majmin_3ring(act_sum, min_sum, forth_sum)
            pred = (root_t + 3) % 12 if pred_is_min else root_t
        elif has_min:
            # Joint: pool the raw maj/min ring activations, then decide by the
            # I+IV+V (tonic/subdominant/dominant) sums.  pred is the key signature.
            root_t, pred_is_min = _decide_majmin_by_degrees(act_sum, min_sum, maj_degree)
            pred = (root_t + 3) % 12 if pred_is_min else root_t
        else:
            # Merged-activation argmax for the signature, mode by segment vote.
            pred = int(max(range(12), key=lambda i: act_sum[i]))
            pred_is_min = 1 if est_mode_votes[1] > est_mode_votes[0] else 0
        merged_song_ok = pred == ref_sig
        if merged_song_ok:
            song_hit += 1

        song_is_min = 1 if ref_mode_counts[1] > ref_mode_counts[0] else 0
        if song_is_min:
            song_min += 1
            if merged_song_ok:
                song_hit_min += 1
            if pred_is_min == 1:
                song_predmin_in_min += 1
        else:
            song_maj += 1
            if merged_song_ok:
                song_hit_maj += 1
            if pred_is_min == 0:
                song_predmaj_in_maj += 1

        # ── Song-level prediction: root from signature argmax, mode from est vote ──
        root_pred = (pred - 3) % 12 if pred_is_min else pred
        pred_rm_song = (root_pred, pred_is_min)
        if len(ref_rm_set) == 1:
            if pred_rm_song == next(iter(ref_rm_set)):
                song_hit_strict += 1

        # ── Song-level MIREX: reference tonic from signature + majority mode ──
        ref_tonic_song = (ref_sig - 3) % 12 if song_is_min else ref_sig
        song_mirex_sum += _mirex_weighted_score((ref_tonic_song, song_is_min), pred_rm_song)

    def _acc(hit: int, tot: int) -> float:
        return round(100.0 * hit / tot, 4) if tot > 0 else 0.0

    return {
        # ── segment-level ──
        "seg_total":          seg_total,
        "seg_acc":            _acc(seg_hit, seg_total),
        "seg_acc_strict":     _acc(seg_hit_strict, seg_total),
        "seg_maj":            seg_maj,
        "seg_min":            seg_min,
        "seg_hit_maj":        seg_hit_maj,
        "seg_acc_maj":        _acc(seg_hit_maj, seg_maj),
        "seg_hit_min":        seg_hit_min,
        "seg_acc_min":        _acc(seg_hit_min, seg_min),
        "seg_predmaj_in_maj": seg_predmaj_in_maj,
        "seg_predmaj_ratio":  _acc(seg_predmaj_in_maj, seg_maj),
        "seg_predmin_in_min": seg_predmin_in_min,
        "seg_predmin_ratio":  _acc(seg_predmin_in_min, seg_min),
        "seg_mirex":          _acc(seg_mirex_sum, seg_total),
        # fraction of segments whose predicted mode matches the gt mode
        "seg_majmin_acc":     _acc(seg_predmaj_in_maj + seg_predmin_in_min, seg_total),
        # ── song-level ──
        "song_total":          song_total,
        "song_acc":            _acc(song_hit, song_total),
        "song_acc_strict":     _acc(song_hit_strict, song_total),
        "song_maj":            song_maj,
        "song_min":            song_min,
        "song_hit_maj":        song_hit_maj,
        "song_acc_maj":        _acc(song_hit_maj, song_maj),
        "song_hit_min":        song_hit_min,
        "song_acc_min":        _acc(song_hit_min, song_min),
        "song_predmaj_in_maj": song_predmaj_in_maj,
        "song_predmaj_ratio":  _acc(song_predmaj_in_maj, song_maj),
        "song_predmin_in_min": song_predmin_in_min,
        "song_predmin_ratio":  _acc(song_predmin_in_min, song_min),
        "song_mirex":          _acc(song_mirex_sum, song_total),
        "song_majmin_acc":     _acc(song_predmaj_in_maj + song_predmin_in_min, song_total),
    }


# ---------------------------------------------------------------------------
# Chord metrics (delegate to evaluate_chord, parse its stdout)
# ---------------------------------------------------------------------------

def _parse_chord_stdout(text: str) -> Tuple[
    Optional[float], Optional[float], Optional[float],
    Optional[float], Optional[float], Optional[float],
    Optional[int],
]:
    """Return (root, majmin, mirex, song_root, song_majmin, song_mirex, n_songs)."""
    def _find(section: str, metric: str) -> Optional[float]:
        in_section = False
        for line in text.splitlines():
            if section in line:
                in_section = True
            if in_section and metric in line:
                # last token, strip %
                val = line.split()[-1].rstrip("%")
                try:
                    return round(float(val), 4)
                except ValueError:
                    return None
        return None

    root       = _find("WCSR",        "root")
    majmin     = _find("WCSR",        "majmin")
    mirex      = _find("WCSR",        "mirex")
    song_root  = _find("SONG-LEVEL",  "root")
    song_majmin= _find("SONG-LEVEL",  "majmin")
    song_mirex = _find("SONG-LEVEL",  "mirex")
    # Total distinct songs (tracks) evaluated, from the "Tracks: N" INFO line.
    m_tracks = re.search(r"Tracks:\s*(\d+)", text)
    n_songs = int(m_tracks.group(1)) if m_tracks else None
    return root, majmin, mirex, song_root, song_majmin, song_mirex, n_songs


def compute_chord_metrics(result_file: Path, fps: float = 10.0) -> Dict:
    cmd = [
        sys.executable, "-m", "src.evaluation.evaluate_chord",
        "--result-file", str(result_file),
        "--fps", str(fps),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"evaluate_chord failed:\n{proc.stderr}"
        )
    root, majmin, mirex, sr, sm, smx, n_songs = _parse_chord_stdout(proc.stdout)
    return {
        "root":       root,
        "majmin":     majmin,
        "mirex":      mirex,
        "song_root":  sr,
        "song_majmin":sm,
        "song_mirex": smx,
        "n_songs":    n_songs,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Unified key/chord metrics → JSON line"
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    # ── key ──────────────────────────────────────────────────────────────────
    kp = sub.add_parser("key", help="Compute key accuracy metrics")
    kp.add_argument("result_file", help="Key result TSV from infer_key.py / probe_key.py")
    kp.add_argument("--maj-degree", type=int, default=0,
                    help="Maj-side ring degree for joint files (0=major chord, 5=forth). "
                         "Only used when the file has both maj/min ring blocks.")
    kp.add_argument("--vote2-decision",
                    choices=["tally", "avg", "joint", "joint_gt"], default="tally",
                    help="Decision for 2-ring 3-SAE files: 'tally' (majority re-vote), "
                         "'avg' (align-average + global argmax, for gt / single-ring), "
                         "or 'joint' (sqrt-compress + signature-first decision, for "
                         "major_minor_joint; joint_gt argmaxes within the gt mode).")

    # ── chord ─────────────────────────────────────────────────────────────────
    cp = sub.add_parser("chord", help="Compute chord WCSR metrics")
    cp.add_argument("result_file", help="Chord result file from infer_key.py")
    cp.add_argument("--fps",      type=float, default=10.0)
    cp.add_argument("--raw-file", default=None,
                    help="Optional raw chord result file; adds raw_* keys to output")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    result_file = Path(args.result_file)
    if not result_file.exists():
        print(json.dumps({"error": f"file not found: {result_file}"}))
        sys.exit(1)

    if args.mode == "key":
        metrics = compute_key_metrics(result_file, maj_degree=args.maj_degree,
                                      vote2_decision=args.vote2_decision)
    else:
        metrics = compute_chord_metrics(result_file, args.fps)
        if args.raw_file:
            raw_path = Path(args.raw_file)
            if raw_path.exists():
                raw = compute_chord_metrics(raw_path, args.fps)
                metrics["raw_root"]       = raw.get("root")
                metrics["raw_majmin"]     = raw.get("majmin")
                metrics["raw_mirex"]      = raw.get("mirex")
                metrics["raw_song_root"]  = raw.get("song_root")
                metrics["raw_song_majmin"]= raw.get("song_majmin")
                metrics["raw_song_mirex"] = raw.get("song_mirex")

    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
