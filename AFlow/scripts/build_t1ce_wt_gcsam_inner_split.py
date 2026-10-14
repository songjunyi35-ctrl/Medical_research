#!/usr/bin/env python3
"""Create the preregistered fresh subject-clean Full972 GCSAM train-inner split."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from scripts.build_sammed3d_t1ce_wt_inner_split import choose_exact_validation_subjects
from scripts.train_t1ce_wt_capg3d import sha256_file


SCHEMA = "aflow.brats2024.t1ce_wt_gcsam_inner_split"
SOURCE_SHA256 = "6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65"
PREREG_SHA256 = "7fe9d8039ede538ff9c47cba001d2ed44ef4769d3aa0142568692797c37ef603"
IDEA_SHA256 = "0c3516c686e13d6cb86b947f2a4fff50a45a936ee8c1ebc939bd9ac349b4250a"
FROZEN_SEED = 20261001


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [str(row["case_id"]) for row in rows]
    if len(rows) != 972 or len(ids) != len(set(ids)):
        raise ValueError("expected972 unique Full972 cases")
    if any(not row.get("subject_id") for row in rows):
        raise ValueError("missing subject_id")
    if len({str(row["subject_id"]) for row in rows}) != 444:
        raise ValueError("expected444 Full972 subjects")
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
    frozen = {
        args.source: SOURCE_SHA256,
        args.idea: IDEA_SHA256,
        args.prereg: PREREG_SHA256,
    }
    for path, expected in frozen.items():
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
    tune = [row for row in rows if str(row["subject_id"]) in tune_subjects]
    fit = [row for row in rows if str(row["subject_id"]) not in tune_subjects]
    if len(tune) != 108 or len(fit) != 864:
        raise RuntimeError("unexpected exact GCSAM split counts")
    tune_ids = {str(row["case_id"]) for row in tune}
    fit_ids = {str(row["case_id"]) for row in fit}
    tune_subject_set = {str(row["subject_id"]) for row in tune}
    fit_subject_set = {str(row["subject_id"]) for row in fit}
    if tune_ids & fit_ids or tune_subject_set & fit_subject_set:
        raise RuntimeError("case/subject leakage in GCSAM split")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    fit_path = args.output_dir / "fit_cases.jsonl"
    tune_path = args.output_dir / "inner_tune_cases.jsonl"
    write_role(fit_path, fit, "gcsam_inner_fit")
    write_role(tune_path, tune, "gcsam_inner_tune")
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
        "roles": {"fit": summary(fit_path, fit), "inner_tune": summary(tune_path, tune)},
        "documents": {"idea_sha256": IDEA_SHA256, "prereg_sha256": PREREG_SHA256},
        "checks": {"case_overlap": 0, "subject_overlap": 0, "union_cases": 972},
        "access": {
            "train_only_inner": False,
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

