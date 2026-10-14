#!/usr/bin/env python3
"""Build two frozen train-only splits for percentile-clipping replication."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_sammed3d_t1ce_wt_inner_split import choose_exact_validation_subjects  # noqa: E402
from scripts.train_t1ce_wt_capg3d import sha256_file  # noqa: E402


SCHEMA = "aflow.brats2024.t1ce_wt_percentile_clip_replication_splits"
SOURCE_SHA256 = "6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65"
IDEA_SHA256 = "7504085184ea0e17d97268e9233ff4eb857f9f80c1b49e301aa1534416a24b0a"
PREREG_SHA256 = "cf4706a98e6053799f698d83dcd59efbfd87d7ff3880b6bea09624b4ed99e031"
SPLIT_SEEDS = {"a": 20261111, "b": 20261113}
TRAIN_SEEDS = {"a": 20261112, "b": 20261114}
BOOTSTRAP_SEEDS = {"a": 20261115, "b": 20261116}


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    case_ids = [str(row.get("case_id", "")) for row in rows]
    subject_ids = [str(row.get("subject_id", "")) for row in rows]
    if len(rows) != 972 or len(set(case_ids)) != 972 or len(set(subject_ids)) != 444:
        raise ValueError("source must contain 972 unique cases / 444 subjects")
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("source contains a non-train row")
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]], role: str, replicate: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({
                "case_id": str(row["case_id"]),
                "subject_id": str(row["subject_id"]),
                "role": role,
                "replicate": replicate,
            }, sort_keys=True) + "\n")


def summarize(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "cases": len(rows),
        "subjects": len({str(row["subject_id"]) for row in rows}),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    for path, expected in (
        (args.source, SOURCE_SHA256),
        (args.idea, IDEA_SHA256),
        (args.prereg, PREREG_SHA256),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"frozen input SHA-256 mismatch: {path}")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing split directory: {output}")
    rows = read_rows(args.source.resolve())
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subject[str(row["subject_id"])].append(row)
    output.mkdir(parents=True, exist_ok=False)
    records: dict[str, Any] = {}
    inner_cases: dict[str, set[str]] = {}
    inner_subjects: dict[str, set[str]] = {}
    counts = {subject: len(items) for subject, items in by_subject.items()}
    for replicate, split_seed in SPLIT_SEEDS.items():
        selected_subjects = choose_exact_validation_subjects(
            counts, seed=split_seed, target_cases=108
        )
        inner = [row for row in rows if str(row["subject_id"]) in selected_subjects]
        fit = [row for row in rows if str(row["subject_id"]) not in selected_subjects]
        fit_case_set = {str(row["case_id"]) for row in fit}
        inner_case_set = {str(row["case_id"]) for row in inner}
        fit_subject_set = {str(row["subject_id"]) for row in fit}
        inner_subject_set = {str(row["subject_id"]) for row in inner}
        if len(fit) != 864 or len(inner) != 108:
            raise RuntimeError(f"replicate {replicate} is not 864/108")
        if fit_case_set & inner_case_set or fit_subject_set & inner_subject_set:
            raise RuntimeError(f"replicate {replicate} leaks case/subject")
        fit_path = output / f"replicate_{replicate}_fit_cases.jsonl"
        inner_path = output / f"replicate_{replicate}_inner_cases.jsonl"
        write_rows(fit_path, fit, "percentile_clip_replication_fit", replicate)
        write_rows(inner_path, inner, "percentile_clip_replication_inner", replicate)
        records[replicate] = {
            "split_seed": split_seed,
            "train_seed": TRAIN_SEEDS[replicate],
            "bootstrap_seed": BOOTSTRAP_SEEDS[replicate],
            "fit": summarize(fit_path, fit),
            "inner": summarize(inner_path, inner),
            "checks": {"case_overlap": 0, "subject_overlap": 0, "union_cases": 972},
        }
        inner_cases[replicate] = inner_case_set
        inner_subjects[replicate] = inner_subject_set
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "frozen_before_cache_model_training_or_inner_access",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(args.source.resolve()), "sha256": SOURCE_SHA256, "cases": 972, "subjects": 444},
        "documents": {"idea_sha256": IDEA_SHA256, "prereg_sha256": PREREG_SHA256},
        "replications": records,
        "cross_replication": {
            "inner_case_overlap": len(inner_cases["a"] & inner_cases["b"]),
            "inner_subject_overlap": len(inner_subjects["a"] & inner_subjects["b"]),
            "same_inner_case_set": inner_cases["a"] == inner_cases["b"],
            "same_inner_subject_set": inner_subjects["a"] == inner_subjects["b"],
        },
        "access": {
            "train_only_inner_metrics": False,
            "fixed_validation108": False,
            "outer": False,
            "confirmation": False,
            "test270": False,
            "ucsd_external75": False,
        },
    }
    (output / "split_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--idea", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
