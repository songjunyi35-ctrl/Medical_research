#!/usr/bin/env python3
"""Freeze a new 648-fit/108-tune split inside the authorized 756-case cohort."""
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


SCHEMA = "aflow.brats2024.t1ce_wt_resolution_inner_split"
SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_authorized_train_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            case_id = str(value.get("case_id", ""))
            subject_id = str(value.get("subject_id", ""))
            if value.get("role") != "train":
                raise ValueError(f"{path}:{line_number} is not an authorized train row")
            if not case_id or subject_id != case_id.rsplit("-", 1)[0]:
                raise ValueError(f"{path}:{line_number} invalid case/subject identity")
            rows.append({"case_id": case_id, "subject_id": subject_id})
    case_ids = [row["case_id"] for row in rows]
    if not rows or len(case_ids) != len(set(case_ids)):
        raise ValueError("authorized train cohort is empty or has duplicate cases")
    return rows


def split_rows(
    rows: list[dict[str, str]], *, seed: int, tune_cases: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_subject[row["subject_id"]].append(row)
    tune_subjects = choose_exact_validation_subjects(
        {subject: len(items) for subject, items in by_subject.items()},
        seed=seed,
        target_cases=tune_cases,
    )
    fit: list[dict[str, str]] = []
    tune: list[dict[str, str]] = []
    for subject_id in sorted(by_subject):
        destination = tune if subject_id in tune_subjects else fit
        role = "inner_tune" if subject_id in tune_subjects else "fit"
        destination.extend(
            {
                "case_id": row["case_id"],
                "subject_id": row["subject_id"],
                "role": role,
            }
            for row in sorted(by_subject[subject_id], key=lambda item: item["case_id"])
        )
    return fit, tune


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_train_cases.resolve()
    if sha256_file(source) != args.expected_source_sha256:
        raise ValueError("authorized train cohort SHA-256 mismatch")
    rows = read_authorized_train_rows(source)
    if len(rows) != args.expected_source_cases:
        raise ValueError(
            f"authorized cohort has {len(rows)} cases, expected {args.expected_source_cases}"
        )
    fit, tune = split_rows(rows, seed=args.seed, tune_cases=args.tune_cases)
    if len(fit) != args.expected_source_cases - args.tune_cases:
        raise RuntimeError("unexpected fit case count")
    if len(tune) != args.tune_cases:
        raise RuntimeError("unexpected inner-tune case count")
    fit_cases = {row["case_id"] for row in fit}
    tune_cases = {row["case_id"] for row in tune}
    fit_subjects = {row["subject_id"] for row in fit}
    tune_subjects = {row["subject_id"] for row in tune}
    if fit_cases & tune_cases or fit_subjects & tune_subjects:
        raise RuntimeError("case or subject leakage in inner split")
    if fit_cases | tune_cases != {row["case_id"] for row in rows}:
        raise RuntimeError("inner split does not exactly cover the authorized cohort")

    output = args.output_dir.resolve()
    lowered = str(output).lower()
    if any(token in lowered for token in ("selector", "confirmation", "sealed", "test270")):
        raise ValueError("output path contains a forbidden role token")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite split directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    fit_path = output / "fit_cases.jsonl"
    tune_path = output / "inner_tune_cases.jsonl"
    write_jsonl(fit_path, fit)
    write_jsonl(tune_path, tune)
    manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_cache_build_and_training",
        "description": (
            "newly resampled subject-clean development split inside the same "
            "historical 756-case train cohort; not external or pristine validation"
        ),
        "seed": args.seed,
        "algorithm": "seeded_subject_shuffle_then_exact_tune_case_subset_sum",
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "cases": len(rows),
            "subjects": len({row["subject_id"] for row in rows}),
            "role_opened": "train_only",
        },
        "roles": {
            "fit": {
                "cases": len(fit),
                "subjects": len(fit_subjects),
                "path": fit_path.name,
                "sha256": sha256_file(fit_path),
            },
            "inner_tune": {
                "cases": len(tune),
                "subjects": len(tune_subjects),
                "path": tune_path.name,
                "sha256": sha256_file(tune_path),
            },
        },
        "checks": {
            "case_overlap": 0,
            "subject_overlap": 0,
            "union_cases": len(fit_cases | tune_cases),
        },
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train-cases", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-source-cases", type=int, default=756)
    parser.add_argument("--tune-cases", type=int, default=108)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output-dir", type=Path, required=True)
    manifest = build(parser.parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
