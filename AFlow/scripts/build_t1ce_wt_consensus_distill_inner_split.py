#!/usr/bin/env python3
"""Build the frozen fresh subject-clean split for consensus distillation."""
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

from scripts.build_sammed3d_t1ce_wt_inner_split import choose_exact_validation_subjects
from scripts.train_t1ce_wt_capg3d import sha256_file


SCHEMA = "aflow.brats2024.t1ce_wt_consensus_distill_inner_split"
SOURCE_SHA256 = "6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65"
IDEA_SHA256 = "97b048fd71c6b2ad7d4266497ce6a074f27cedd98c22e60bb926fad8ce1fc0d4"
PREREG_SHA256 = "120d1f00899d1d0d51910b063a4e9301b26b7e0f584fc1c78938c0a0546bdb73"
FROZEN_SEED = 20261030


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [str(row["case_id"]) for row in rows]
    subjects = [str(row["subject_id"]) for row in rows]
    if len(rows) != 972 or len(ids) != len(set(ids)) or len(set(subjects)) != 444:
        raise ValueError("expected frozen Full972 unique cases/subjects")
    return rows


def write_role(path: Path, rows: list[dict[str, Any]], role: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({
                "case_id": str(row["case_id"]),
                "subject_id": str(row["subject_id"]),
                "role": role,
            }, sort_keys=True) + "\n")


def summary(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "cases": len(rows),
        "subjects": len({str(row["subject_id"]) for row in rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--idea", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=FROZEN_SEED)
    parser.add_argument("--inner-cases", type=int, default=108)
    args = parser.parse_args()
    if args.seed != FROZEN_SEED or args.inner_cases != 108:
        raise ValueError("split seed/count differ from preregistration")
    if args.output_dir.exists():
        raise FileExistsError("refusing existing split directory")
    for path, expected in {
        args.source: SOURCE_SHA256,
        args.idea: IDEA_SHA256,
        args.prereg: PREREG_SHA256,
    }.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen input SHA mismatch: {path}")

    rows = read_rows(args.source)
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subject[str(row["subject_id"])].append(row)
    tune_subjects = choose_exact_validation_subjects(
        {subject: len(items) for subject, items in by_subject.items()},
        seed=args.seed,
        target_cases=args.inner_cases,
    )
    inner = [row for row in rows if str(row["subject_id"]) in tune_subjects]
    fit = [row for row in rows if str(row["subject_id"]) not in tune_subjects]
    fit_ids = {str(row["case_id"]) for row in fit}
    inner_ids = {str(row["case_id"]) for row in inner}
    fit_subjects = {str(row["subject_id"]) for row in fit}
    inner_subjects = {str(row["subject_id"]) for row in inner}
    if len(fit) != 864 or len(inner) != 108:
        raise RuntimeError("unexpected exact split case counts")
    if fit_ids & inner_ids or fit_subjects & inner_subjects:
        raise RuntimeError("case/subject leakage")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    fit_path = args.output_dir / "fit_cases.jsonl"
    inner_path = args.output_dir / "inner_tune_cases.jsonl"
    write_role(fit_path, fit, "consensus_distill_fit")
    write_role(inner_path, inner, "consensus_distill_inner_tune")
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "frozen_before_implementation_and_training",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": {
            "name": "seeded_subject_shuffle_exact_case_subset_sum",
            "seed": args.seed,
            "inner_cases": args.inner_cases,
        },
        "source": {"path": str(args.source), "sha256": SOURCE_SHA256, "cases": 972, "subjects": 444},
        "roles": {"fit": summary(fit_path, fit), "inner_tune": summary(inner_path, inner)},
        "documents": {"idea_sha256": IDEA_SHA256, "prereg_sha256": PREREG_SHA256},
        "checks": {"case_overlap": 0, "subject_overlap": 0, "union_cases": 972},
        "access": {
            "train_only_inner": False,
            "fixed_validation108": False,
            "outer": False,
            "confirmation": False,
            "test270": False
        }
    }
    manifest = args.output_dir / "split_manifest.json"
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
