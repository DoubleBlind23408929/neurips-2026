#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate group-aware score files for orbit verification.

Directory mapping
-----------------
For every group under::

    <feature-ids-root>/<group>/

this script reads exactly::

    <chord-results-root>/<group>/chord_song_level_gtbnd.tsv
    <relative-key-results-root>/<group>/key_song_relative.tsv

There is no cross-group fallback and no cross-mode fallback.

Exact lookup key
----------------
    (feature, lid, mode)

Output
------
For every ``epoch*_feature_ids.txt`` file, write or overwrite the companion::

    epoch*_score.txt

with::

    [major]: <RWC GT-boundary chord root>
    [minor]: <RWC GT-boundary chord root>
    [forth]: <RWC relative-key forth acc>

Legacy ``epoch*_scores.txt`` files are removed after the canonical file is
written, preventing old plural-named files from being read accidentally.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple


SCRIPT_NAME = "generate_grouped_orbit_scores.py"
SCRIPT_VERSION = "1.0"

Key = Tuple[str, str, str]  # feature, lid, mode

FEATURE_IDS_SUFFIX = "_feature_ids.txt"
SCORE_SUFFIX = "_score.txt"
LEGACY_SCORE_SUFFIX = "_scores.txt"

CHORD_FILENAME = "chord_song_level_gtbnd.tsv"
RELATIVE_KEY_FILENAME = "key_song_relative.tsv"

# Example:
# slakh2100_train_16_musicfm-2_3-1_1e-3_48_max
RUN_NAME_RE = re.compile(
    r"^[^_]+_([^-]+)_([^-]+)_([^-]+)-([0-9]+)_"
    r"([0-9]+-[0-9]+)_([^_]+)_([0-9]+)_max$"
)


def normalize_feature(value: str) -> str:
    return value.strip().lower()


def normalize_lid(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("Empty lid")
    return str(int(text))


def normalize_mode(value: str) -> str:
    return value.strip()


def parse_model_key(model_dir: Path) -> Key:
    """Parse exact ``(feature, lid, mode)`` from a run-directory name."""
    match = RUN_NAME_RE.match(model_dir.name)
    if match is None:
        raise ValueError(
            "Unrecognised model-directory name: "
            f"{model_dir.name}. Expected a name like "
            "slakh2100_train_16_musicfm-2_3-1_1e-3_48_max"
        )
    return (
        normalize_feature(match.group(3)),
        normalize_lid(match.group(4)),
        normalize_mode(match.group(5)),
    )


def require_columns(
    path: Path,
    fieldnames: Optional[Sequence[str]],
    required: Set[str],
) -> None:
    present = set(fieldnames or [])
    missing = required - present
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")


def insert_unique(
    table: Dict[Key, str],
    *,
    key: Key,
    value: str,
    path: Path,
) -> None:
    previous = table.get(key)
    if previous is not None and previous != value:
        raise ValueError(
            f"{path}: conflicting values for exact key {key}: "
            f"{previous!r} vs {value!r}"
        )
    table[key] = value


def load_chord_roots(path: Path) -> Dict[Key, str]:
    """Load ``dataset=rwc``, ``boundary_feature_rank=gt``, value ``root``."""
    scores: Dict[Key, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_columns(
            path,
            reader.fieldnames,
            {
                "feature",
                "lid",
                "mode",
                "dataset",
                "boundary_feature_rank",
                "root",
            },
        )
        for row in reader:
            if row["dataset"].strip().lower() != "rwc":
                continue
            if row["boundary_feature_rank"].strip().lower() != "gt":
                continue

            value = row["root"].strip()
            if not value:
                continue

            key = (
                normalize_feature(row["feature"]),
                normalize_lid(row["lid"]),
                normalize_mode(row["mode"]),
            )
            insert_unique(scores, key=key, value=value, path=path)
    return scores


def load_forth_acc(path: Path) -> Dict[Key, str]:
    """Load ``dataset=rwc``, ``degree=forth``, value ``acc``."""
    scores: Dict[Key, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_columns(
            path,
            reader.fieldnames,
            {"feature", "lid", "mode", "degree", "dataset", "acc"},
        )
        for row in reader:
            if row["dataset"].strip().lower() != "rwc":
                continue
            if row["degree"].strip().lower() != "forth":
                continue

            value = row["acc"].strip()
            if not value:
                continue

            key = (
                normalize_feature(row["feature"]),
                normalize_lid(row["lid"]),
                normalize_mode(row["mode"]),
            )
            insert_unique(scores, key=key, value=value, path=path)
    return scores


def find_feature_id_files(group_dir: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in group_dir.rglob("epoch*_feature_ids.txt")
        if "probe_feature_ids" not in path.name
    )


def companion_path(feature_ids_file: Path, suffix: str) -> Path:
    name = feature_ids_file.name
    if not name.endswith(FEATURE_IDS_SUFFIX):
        raise ValueError(f"Unexpected feature-id filename: {feature_ids_file}")
    prefix = name[: -len(FEATURE_IDS_SUFFIX)]
    return feature_ids_file.with_name(prefix + suffix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate exact-mode orbit score files using the matching result "
            "directory for each feature_ids group."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--feature-ids-root",
        type=Path,
        default=Path("store/results_latest/analysis/feature_ids"),
        help="Root containing <group>/<model>/epoch*_feature_ids.txt.",
    )
    parser.add_argument(
        "--chord-results-root",
        type=Path,
        default=Path("store/results_latest/analysis/chord_results_reb"),
        help="Root containing <group>/chord_song_level_gtbnd.tsv.",
    )
    parser.add_argument(
        "--relative-key-results-root",
        type=Path,
        default=Path("store/results_latest/analysis/relative_key_results"),
        help="Root containing <group>/key_song_relative.tsv.",
    )
    parser.add_argument(
        "--group",
        action="append",
        default=None,
        help=(
            "Process only this exact group name. May be supplied multiple "
            "times. By default all feature_ids groups are processed."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail immediately when a group TSV or an exact score key is "
            "missing. Without this flag, missing scores are written as N/A."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    feature_root = args.feature_ids_root.expanduser().resolve()
    chord_root = args.chord_results_root.expanduser().resolve()
    relative_root = args.relative_key_results_root.expanduser().resolve()

    if not feature_root.is_dir():
        raise NotADirectoryError(f"Feature-id root not found: {feature_root}")
    if not chord_root.is_dir():
        raise NotADirectoryError(f"Chord-results root not found: {chord_root}")
    if not relative_root.is_dir():
        raise NotADirectoryError(
            f"Relative-key-results root not found: {relative_root}"
        )

    requested_groups = set(args.group or [])
    group_dirs = sorted(path for path in feature_root.iterdir() if path.is_dir())
    if requested_groups:
        available = {path.name for path in group_dirs}
        unknown = requested_groups - available
        if unknown:
            raise FileNotFoundError(
                f"Requested feature_ids groups not found: {sorted(unknown)}"
            )
        group_dirs = [path for path in group_dirs if path.name in requested_groups]

    if not group_dirs:
        raise RuntimeError(f"No groups found under {feature_root}")

    print(f"[SCRIPT] {SCRIPT_NAME} v{SCRIPT_VERSION}")
    print(f"[ROOT] feature_ids={feature_root}")
    print(f"[ROOT] chord_results={chord_root}")
    print(f"[ROOT] relative_key_results={relative_root}")

    groups_processed = 0
    groups_skipped = 0
    files_written = 0
    missing_chord = 0
    missing_forth = 0

    for group_dir in group_dirs:
        group = group_dir.name
        chord_file = chord_root / group / CHORD_FILENAME
        relative_file = relative_root / group / RELATIVE_KEY_FILENAME

        missing_group_files = [
            str(path)
            for path in (chord_file, relative_file)
            if not path.is_file()
        ]
        if missing_group_files:
            message = (
                f"group={group}: missing result file(s): "
                + ", ".join(missing_group_files)
            )
            if args.strict:
                raise FileNotFoundError(message)
            print(f"[SKIP-GROUP] {message}")
            groups_skipped += 1
            continue

        chord_scores = load_chord_roots(chord_file)
        forth_scores = load_forth_acc(relative_file)
        feature_files = list(find_feature_id_files(group_dir))

        print(
            f"[GROUP] {group} | feature_files={len(feature_files)} | "
            f"chord_rows={len(chord_scores)} | forth_rows={len(forth_scores)}"
        )
        print(f"[TSV] chord={chord_file}")
        print(f"[TSV] relative_key={relative_file}")

        for feature_ids_file in feature_files:
            key = parse_model_key(feature_ids_file.parent)
            chord = chord_scores.get(key)
            forth = forth_scores.get(key)

            missing = []
            if chord is None:
                missing.append("chord root")
                missing_chord += 1
            if forth is None:
                missing.append("forth acc")
                missing_forth += 1

            if missing:
                message = (
                    f"{feature_ids_file}: missing {', '.join(missing)} "
                    f"for exact key={key} in group={group}"
                )
                if args.strict:
                    raise KeyError(message)
                print(f"[WARN] {message}")

            output_file = companion_path(feature_ids_file, SCORE_SUFFIX)
            legacy_file = companion_path(feature_ids_file, LEGACY_SCORE_SUFFIX)
            existed = output_file.exists()

            # Path.write_text opens with mode='w', so an existing file is
            # truncated and completely replaced on every run.
            output_file.write_text(
                f"[major]: {chord if chord is not None else 'N/A'}\n"
                f"[minor]: {chord if chord is not None else 'N/A'}\n"
                f"[forth]: {forth if forth is not None else 'N/A'}\n",
                encoding="utf-8",
            )

            if legacy_file.exists():
                legacy_file.unlink()
                print(f"[REMOVE-LEGACY] {legacy_file}")

            action = "OVERWRITE" if existed else "WRITE"
            print(
                f"[{action}] {output_file} | group={group} | key={key} | "
                f"major={chord if chord is not None else 'N/A'} | "
                f"minor={chord if chord is not None else 'N/A'} | "
                f"forth={forth if forth is not None else 'N/A'}"
            )
            files_written += 1

        groups_processed += 1

    print(
        f"[DONE] groups_processed={groups_processed} "
        f"groups_skipped={groups_skipped} files_written={files_written} "
        f"missing_chord={missing_chord} missing_forth={missing_forth}"
    )


if __name__ == "__main__":
    main()
