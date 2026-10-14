#!/usr/bin/env python3
"""Split the frozen outer-train cohort into subject-clean train/validation sets."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping


SCHEMA = "aflow.brats2024.sammed3d_t1ce_wt_inner_split"
SCHEMA_VERSION = 1
DEFAULT_SEED = 20260723
FOREGROUND_LABELS = (1, 2, 3)
EXCLUDED_LABELS = (4,)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def choose_exact_validation_subjects(
    case_counts: Mapping[str, int], *, seed: int, target_cases: int
) -> set[str]:
    if not 0 < target_cases < sum(case_counts.values()):
        raise ValueError("target_cases must leave non-empty train and validation sets")
    subjects = sorted(case_counts)
    random.Random(seed).shuffle(subjects)
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for subject_id in subjects:
        count = int(case_counts[subject_id])
        if count <= 0:
            raise ValueError(f"subject has no cases: {subject_id}")
        for subtotal in sorted(tuple(reachable), reverse=True):
            candidate = subtotal + count
            if candidate <= target_cases and candidate not in reachable:
                reachable[candidate] = reachable[subtotal] + (subject_id,)
        if target_cases in reachable:
            return set(reachable[target_cases])
    raise RuntimeError(
        f"no subject-clean subset contains exactly {target_cases} cases"
    )


def normalize_row(row: Mapping[str, object], split: str) -> dict:
    modalities = row.get("modalities")
    if not isinstance(modalities, Mapping) or not modalities.get("t1c"):
        raise ValueError(f"row lacks modalities.t1c: {row.get('case_id')}")
    required = ("case_id", "subject_id", "timepoint", "seg_path")
    missing = [field for field in required if not row.get(field)]
    if missing:
        raise ValueError(f"row lacks fields {missing}: {row.get('case_id')}")
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "case_id": str(row["case_id"]),
        "patient_id": str(row["subject_id"]),
        "subject_id": str(row["subject_id"]),
        "timepoint": str(row["timepoint"]),
        "split": split,
        "target": "WT",
        "modality": "t1ce",
        "image_path": str(modalities["t1c"]),
        "label_path": str(row["seg_path"]),
        "foreground_labels": list(FOREGROUND_LABELS),
        "excluded_labels": list(EXCLUDED_LABELS),
        "source_part": row.get("source_part"),
    }


def split_rows(
    rows: Iterable[dict], *, seed: int, validation_cases: int
) -> tuple[list[dict], list[dict]]:
    by_subject: dict[str, list[dict]] = defaultdict(list)
    seen_cases: set[str] = set()
    for row in rows:
        case_id = str(row.get("case_id", ""))
        subject_id = str(row.get("subject_id", ""))
        if not case_id or not subject_id:
            raise ValueError("every source row requires case_id and subject_id")
        if case_id in seen_cases:
            raise ValueError(f"duplicate case_id: {case_id}")
        if row.get("split") != "train":
            raise ValueError(f"source row is not from outer train: {case_id}")
        seen_cases.add(case_id)
        by_subject[subject_id].append(row)
    validation_subjects = choose_exact_validation_subjects(
        {subject: len(items) for subject, items in by_subject.items()},
        seed=seed,
        target_cases=validation_cases,
    )
    train = []
    validation = []
    for subject_id in sorted(by_subject):
        is_validation = subject_id in validation_subjects
        destination = validation if is_validation else train
        split = "validation" if is_validation else "train"
        destination.extend(
            normalize_row(row, split)
            for row in sorted(
                by_subject[subject_id], key=lambda item: str(item["case_id"])
            )
        )
    return train, validation


def subject_ids(rows: Iterable[dict]) -> set[str]:
    return {str(row["subject_id"]) for row in rows}


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-train-jsonl", required=True, type=Path)
    parser.add_argument("--outer-split-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--validation-ratio-within-train", type=float, default=0.10)
    parser.add_argument("--expected-outer-train-cases", type=int, default=1080)
    args = parser.parse_args()

    if not 0.0 < args.validation_ratio_within_train < 1.0:
        raise ValueError("validation ratio must be between zero and one")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    source_rows = read_jsonl(args.outer_train_jsonl)
    if len(source_rows) != args.expected_outer_train_cases:
        raise RuntimeError(
            f"expected {args.expected_outer_train_cases} outer-train cases, "
            f"found {len(source_rows)}"
        )
    validation_cases = round(
        len(source_rows) * args.validation_ratio_within_train
    )
    train_rows, validation_rows = split_rows(
        source_rows, seed=args.seed, validation_cases=validation_cases
    )
    train_subjects = subject_ids(train_rows)
    validation_subjects = subject_ids(validation_rows)
    if train_subjects & validation_subjects:
        raise RuntimeError("subject leakage between train and validation")
    if {row["case_id"] for row in train_rows} & {
        row["case_id"] for row in validation_rows
    }:
        raise RuntimeError("case leakage between train and validation")
    if len(train_rows) + len(validation_rows) != len(source_rows):
        raise RuntimeError("inner split does not cover the outer-train cohort")

    outer_manifest = json.loads(
        args.outer_split_manifest.read_text(encoding="utf-8")
    )
    outer_achieved = outer_manifest.get("achieved", {})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    train_path = args.output_dir / "train_t1ce_wt.jsonl"
    validation_path = args.output_dir / "validation_t1ce_wt.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, validation_rows)
    manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_before_training",
        "task": "T1ce_to_binary_WT_interactive_segmentation",
        "foreground_raw_labels": list(FOREGROUND_LABELS),
        "excluded_raw_labels": list(EXCLUDED_LABELS),
        "seed": args.seed,
        "split_algorithm": (
            "python_random_seeded_subject_shuffle_then_exact_validation_case_subset_sum"
        ),
        "group_unit": "subject_id_all_longitudinal_timepoints_together",
        "requested_ratio_within_outer_train": {
            "train": 1.0 - args.validation_ratio_within_train,
            "validation": args.validation_ratio_within_train,
        },
        "achieved": {
            "train": {
                "cases": len(train_rows),
                "subjects": len(train_subjects),
            },
            "validation": {
                "cases": len(validation_rows),
                "subjects": len(validation_subjects),
            },
            "outer_test": {
                "cases": outer_achieved.get("test", {}).get("cases"),
                "subjects": outer_achieved.get("test", {}).get("subjects"),
                "manifest_opened_by_builder": False,
                "nifti_opened_by_builder": False,
            },
        },
        "checks": {
            "case_overlap": 0,
            "subject_overlap": 0,
            "union_outer_train_cases": len(source_rows),
        },
        "source": {
            "outer_train_jsonl": str(args.outer_train_jsonl.resolve()),
            "outer_train_jsonl_sha256": sha256_file(args.outer_train_jsonl),
            "outer_split_manifest": str(args.outer_split_manifest.resolve()),
            "outer_split_manifest_sha256": sha256_file(args.outer_split_manifest),
        },
        "outputs": {
            train_path.name: sha256_file(train_path),
            validation_path.name: sha256_file(validation_path),
        },
    }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["achieved"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
