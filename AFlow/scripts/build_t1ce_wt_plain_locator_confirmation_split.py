#!/usr/bin/env python3
"""Freeze a new three-way subject-clean split inside the 972-case train cohort."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.build_sammed3d_t1ce_wt_inner_split import (
    choose_exact_validation_subjects,
)


SCHEMA = "aflow.brats2024.t1ce_wt_plain_locator_confirmation_split"
SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("case_id") or not row.get("subject_id"):
                raise ValueError(f"{path}:{line_number} lacks case_id or subject_id")
            rows.append(row)
    return rows


def write_role(path: Path, rows: list[dict[str, Any]], *, role: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            value = {
                "case_id": str(row["case_id"]),
                "subject_id": str(row["subject_id"]),
                "role": role,
            }
            handle.write(json.dumps(value, sort_keys=True) + "\n")


def subject_set(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["subject_id"]) for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-cases", type=int, default=972)
    parser.add_argument("--selector-cases", type=int, default=108)
    parser.add_argument("--confirmation-cases", type=int, default=108)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    source = args.source_manifest.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite split directory: {output}")
    source_sha256 = sha256_file(source)
    if source_sha256 != args.expected_source_sha256:
        raise RuntimeError(
            f"source manifest SHA-256 {source_sha256} != {args.expected_source_sha256}"
        )
    rows = read_jsonl(source)
    if len(rows) != args.expected_cases:
        raise RuntimeError(f"expected {args.expected_cases} cases, found {len(rows)}")
    case_ids = [str(row["case_id"]) for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("source manifest contains duplicate cases")

    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subject[str(row["subject_id"])].append(row)
    counts = {subject: len(items) for subject, items in by_subject.items()}
    confirmation_subjects = choose_exact_validation_subjects(
        counts, seed=args.seed, target_cases=args.confirmation_cases
    )
    remaining_counts = {
        subject: count
        for subject, count in counts.items()
        if subject not in confirmation_subjects
    }
    selector_subjects = choose_exact_validation_subjects(
        remaining_counts, seed=args.seed + 1, target_cases=args.selector_cases
    )

    roles: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "selector": [],
        "confirmation": [],
    }
    for row in rows:
        subject = str(row["subject_id"])
        if subject in confirmation_subjects:
            role = "confirmation"
        elif subject in selector_subjects:
            role = "selector"
        else:
            role = "train"
        roles[role].append(row)

    expected_train = args.expected_cases - args.selector_cases - args.confirmation_cases
    expected_role_counts = {
        "train": expected_train,
        "selector": args.selector_cases,
        "confirmation": args.confirmation_cases,
    }
    for role, expected in expected_role_counts.items():
        if len(roles[role]) != expected:
            raise RuntimeError(f"{role} has {len(roles[role])} cases, expected {expected}")
    for left in roles:
        for right in roles:
            if left >= right:
                continue
            if {str(row["case_id"]) for row in roles[left]} & {
                str(row["case_id"]) for row in roles[right]
            }:
                raise RuntimeError(f"case leakage between {left} and {right}")
            if subject_set(roles[left]) & subject_set(roles[right]):
                raise RuntimeError(f"subject leakage between {left} and {right}")

    output.mkdir(parents=True, exist_ok=False)
    role_paths = {
        "train": output / "train_cases.jsonl",
        "selector": output / "selector_cases.jsonl",
        "confirmation": output / "sealed_confirmation" / "confirmation_cases.jsonl",
    }
    for role, path in role_paths.items():
        write_role(path, roles[role], role=role)

    manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_training",
        "purpose": "plain locator protocol replication with untouched confirmation",
        "test_access": False,
        "source": {
            "manifest": str(source),
            "manifest_sha256": source_sha256,
            "cases": len(rows),
            "subjects": len(by_subject),
            "scope": "historical 972-case train cohort only",
        },
        "algorithm": {
            "name": "two_stage_seeded_subject_shuffle_exact_case_subset_sum",
            "seed": args.seed,
            "confirmation_selected_first": True,
            "selector_seed": args.seed + 1,
        },
        "roles": {
            role: {
                "cases": len(role_rows),
                "subjects": len(subject_set(role_rows)),
                "manifest": str(role_paths[role].relative_to(output)),
                "manifest_sha256": sha256_file(role_paths[role]),
            }
            for role, role_rows in roles.items()
        },
        "checks": {
            "case_overlap_all_pairs": 0,
            "subject_overlap_all_pairs": 0,
            "union_cases": sum(len(value) for value in roles.values()),
        },
        "forbidden": {
            "historical_validation_cases": "not read by this builder",
            "270_case_test_manifest": "not read by this builder",
            "270_case_test_nifti": "not read by this builder",
        },
    }
    manifest_path = output / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
