#!/usr/bin/env python3
"""Freeze a subject-clean fit/inner-tune split inside a case-role manifest."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from scripts.build_sammed3d_t1ce_wt_inner_split import choose_exact_validation_subjects
from scripts.train_t1ce_wt_capg3d import sha256_file


SCHEMA = "aflow.brats2024.t1ce_wt_intensity_inner_split"


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(not row.get("case_id") or not row.get("subject_id") for row in rows):
        raise ValueError("source role manifest is empty or malformed")
    case_ids = [str(row["case_id"]) for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("source role manifest contains duplicate cases")
    return rows


def write_role(path: Path, rows: list[dict[str, Any]], role: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({
                "case_id": str(row["case_id"]),
                "subject_id": str(row["subject_id"]),
                "role": role,
            }, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-role", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-cases", type=int, default=756)
    parser.add_argument("--inner-tune-cases", type=int, default=108)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_role.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite split: {output}")
    source_sha = sha256_file(source)
    if source_sha != args.expected_source_sha256:
        raise RuntimeError("source role SHA-256 mismatch")
    rows = read_rows(source)
    if len(rows) != args.expected_cases:
        raise RuntimeError(f"source has {len(rows)} cases, expected {args.expected_cases}")

    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subject[str(row["subject_id"])].append(row)
    tune_subjects = choose_exact_validation_subjects(
        {subject: len(items) for subject, items in by_subject.items()},
        seed=args.seed,
        target_cases=args.inner_tune_cases,
    )
    roles = {"fit": [], "inner_tune": []}
    for row in rows:
        role = "inner_tune" if str(row["subject_id"]) in tune_subjects else "fit"
        roles[role].append(row)
    expected_fit = args.expected_cases - args.inner_tune_cases
    if len(roles["fit"]) != expected_fit or len(roles["inner_tune"]) != args.inner_tune_cases:
        raise RuntimeError("exact case-count split failed")
    fit_cases = {str(row["case_id"]) for row in roles["fit"]}
    tune_cases = {str(row["case_id"]) for row in roles["inner_tune"]}
    fit_subjects = {str(row["subject_id"]) for row in roles["fit"]}
    actual_tune_subjects = {str(row["subject_id"]) for row in roles["inner_tune"]}
    if fit_cases & tune_cases or fit_subjects & actual_tune_subjects:
        raise RuntimeError("inner split leakage")

    output.mkdir(parents=True, exist_ok=False)
    fit_path = output / "fit_cases.jsonl"
    tune_path = output / "inner_tune_cases.jsonl"
    write_role(fit_path, roles["fit"], "fit")
    write_role(tune_path, roles["inner_tune"], "inner_tune")
    manifest = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_training",
        "purpose": "matched intensity-aware uncertainty residual development",
        "source": {
            "path": str(source),
            "sha256": source_sha,
            "cases": len(rows),
            "subjects": len(by_subject),
        },
        "algorithm": {
            "name": "seeded_subject_shuffle_exact_case_subset_sum",
            "seed": args.seed,
            "inner_tune_cases": args.inner_tune_cases,
        },
        "roles": {
            "fit": {
                "path": str(fit_path), "sha256": sha256_file(fit_path),
                "cases": len(roles["fit"]), "subjects": len(fit_subjects),
            },
            "inner_tune": {
                "path": str(tune_path), "sha256": sha256_file(tune_path),
                "cases": len(roles["inner_tune"]), "subjects": len(actual_tune_subjects),
            },
        },
        "checks": {"case_overlap": 0, "subject_overlap": 0, "union_cases": len(rows)},
        "forbidden_access": {
            "outer_selector": False,
            "confirmation": False,
            "test270": False,
        },
    }
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
