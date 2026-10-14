#!/usr/bin/env python3
"""Build a common subject-clean inner tune for full756 versus full972 trajectories."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from scripts.build_sammed3d_t1ce_wt_inner_split import choose_exact_validation_subjects
from scripts.train_t1ce_wt_capg3d import sha256_file


SCHEMA = "aflow.brats2024.t1ce_wt_fulltrajectory_common_inner_split"


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty manifest")
    ids = [str(row["case_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case IDs")
    if any(not row.get("subject_id") for row in rows):
        raise ValueError("missing subject IDs")
    return rows


def write_role(path: Path, rows: list[dict[str, Any]], role: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({
                "case_id": str(row["case_id"]),
                "subject_id": str(row["subject_id"]),
                "role": role,
            }, sort_keys=True) + "\n")


def role_summary(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "cases": len(rows),
        "subjects": len({str(row["subject_id"]) for row in rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-source", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--control-sha256", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--inner-cases", type=int, default=108)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError("refusing existing split directory")
    if sha256_file(args.control_source) != args.control_sha256:
        raise ValueError("control source SHA mismatch")
    if sha256_file(args.candidate_source) != args.candidate_sha256:
        raise ValueError("candidate source SHA mismatch")
    control = read_rows(args.control_source)
    candidate = read_rows(args.candidate_source)
    if len(control) != 756 or len(candidate) != 972:
        raise ValueError("expected frozen 756/972 source counts")
    control_ids = {str(row["case_id"]) for row in control}
    candidate_ids = {str(row["case_id"]) for row in candidate}
    if not control_ids < candidate_ids or len(candidate_ids - control_ids) != 216:
        raise ValueError("candidate must be exact control superset plus216 cases")

    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in control:
        by_subject[str(row["subject_id"])].append(row)
    tune_subjects = choose_exact_validation_subjects(
        {subject: len(rows) for subject, rows in by_subject.items()},
        seed=args.seed,
        target_cases=args.inner_cases,
    )
    tune = [row for row in control if str(row["subject_id"]) in tune_subjects]
    control_fit = [row for row in control if str(row["subject_id"]) not in tune_subjects]
    candidate_fit = [row for row in candidate if str(row["subject_id"]) not in tune_subjects]
    if len(tune) != 108 or len(control_fit) != 648 or len(candidate_fit) != 864:
        raise RuntimeError("unexpected exact common-inner counts")
    tune_ids = {str(row["case_id"]) for row in tune}
    tune_subject_set = {str(row["subject_id"]) for row in tune}
    for fit in (control_fit, candidate_fit):
        if tune_ids & {str(row["case_id"]) for row in fit}:
            raise RuntimeError("case leakage")
        if tune_subject_set & {str(row["subject_id"]) for row in fit}:
            raise RuntimeError("subject leakage")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "control_fit": args.output_dir / "control_fit_cases.jsonl",
        "candidate_fit": args.output_dir / "candidate_fit_cases.jsonl",
        "common_inner_tune": args.output_dir / "common_inner_tune_cases.jsonl",
    }
    write_role(paths["control_fit"], control_fit, "inner_control_fit")
    write_role(paths["candidate_fit"], candidate_fit, "inner_candidate_fit")
    write_role(paths["common_inner_tune"], tune, "common_inner_tune")
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "frozen_before_inner_training",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": {
            "name": "seeded_subject_shuffle_exact_case_subset_sum_from_common756",
            "seed": args.seed,
            "inner_tune_cases": args.inner_cases,
        },
        "sources": {
            "control": {"sha256": args.control_sha256, "cases": 756, "subjects": 346},
            "candidate": {"sha256": args.candidate_sha256, "cases": 972, "subjects": 444},
        },
        "roles": {
            "control_fit": role_summary(paths["control_fit"], control_fit),
            "candidate_fit": role_summary(paths["candidate_fit"], candidate_fit),
            "common_inner_tune": role_summary(paths["common_inner_tune"], tune),
        },
        "checks": {
            "control_subset_candidate": True,
            "extra_candidate_cases": 216,
            "case_overlap_with_tune": 0,
            "subject_overlap_with_tune": 0,
        },
        "access": {
            "fixed_validation108": False,
            "outer": False,
            "confirmation": False,
            "test270": False,
        },
    }
    manifest = args.output_dir / "split_manifest.json"
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
