#!/usr/bin/env python3
"""Build the development-only T1ce split for the DPD-SAM reconstruction.

The existing outer test remains sealed. This script reads only the two existing
development manifests and the outer split ledger metadata, then writes train
and validation manifests containing one image path per case.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


SCHEMA = "aflow.brats2024.dpdsam_t1ce_binary_split"
SCHEMA_VERSION = 1
FOREGROUND_LABELS = (1, 2, 3)
EXCLUDED_LABELS = (4,)
DEFAULT_SEED = 20260721


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
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not an object")
                rows.append(value)
    return rows


def subject_rank(subject_id: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}:{subject_id}".encode("ascii")).hexdigest()
    return digest, subject_id


def normalize_development_row(row: dict, split: str) -> dict:
    required = ("case_id", "subject_id", "timepoint", "modalities", "seg_path")
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"development row lacks fields {missing}: {row.get('case_id')}")
    modalities = row["modalities"]
    if not isinstance(modalities, dict) or not modalities.get("t1c"):
        raise ValueError(f"row lacks modalities.t1c: {row['case_id']}")
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "case_id": str(row["case_id"]),
        "subject_id": str(row["subject_id"]),
        "timepoint": str(row["timepoint"]),
        "split": split,
        "modality": "t1ce",
        "image_path": str(modalities["t1c"]),
        "label_path": str(row["seg_path"]),
        "foreground_labels": list(FOREGROUND_LABELS),
        "excluded_labels": list(EXCLUDED_LABELS),
        "source_part": row.get("source_part"),
        "source_shape": row.get("shape"),
        "source_spacing_mm": row.get("spacing_mm"),
    }


def build_development_split(
    source_rows: list[dict],
    *,
    seed: int = DEFAULT_SEED,
    validation_subjects: int = 97,
) -> tuple[list[dict], list[dict]]:
    by_subject: dict[str, list[dict]] = defaultdict(list)
    seen_cases: set[str] = set()
    for row in source_rows:
        case_id = str(row.get("case_id", ""))
        subject_id = str(row.get("subject_id", ""))
        if not case_id or not subject_id:
            raise ValueError("every row requires case_id and subject_id")
        if case_id in seen_cases:
            raise ValueError(f"duplicate development case_id: {case_id}")
        seen_cases.add(case_id)
        by_subject[subject_id].append(row)
    if not 0 < validation_subjects < len(by_subject):
        raise ValueError("validation_subjects must leave non-empty train and val sets")

    ranked = sorted(by_subject, key=lambda value: subject_rank(value, seed))
    val_subject_ids = set(ranked[:validation_subjects])
    train_rows = []
    val_rows = []
    for subject_id in sorted(by_subject):
        destination = val_rows if subject_id in val_subject_ids else train_rows
        split = "validation" if subject_id in val_subject_ids else "train"
        destination.extend(
            normalize_development_row(row, split)
            for row in sorted(by_subject[subject_id], key=lambda value: value["case_id"])
        )
    return train_rows, val_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def _subject_ids(rows: list[dict]) -> set[str]:
    return {row["subject_id"] for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-train", required=True, type=Path)
    parser.add_argument("--development-val", required=True, type=Path)
    parser.add_argument("--outer-split-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--validation-subjects", type=int, default=97)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    outer = json.loads(args.outer_split_manifest.read_text(encoding="utf-8"))
    test_cases = set(outer["case_ids"]["test"])
    test_subjects = set(outer["patient_ids"]["test"])
    test_file_sha = outer["file_sha256"]["sealed_test/brats_test.jsonl"]

    source_paths = (args.development_train, args.development_val)
    source_rows = [row for path in source_paths for row in read_jsonl(path)]
    development_cases = {str(row["case_id"]) for row in source_rows}
    development_subjects = {str(row["subject_id"]) for row in source_rows}
    if development_cases & test_cases or development_subjects & test_subjects:
        raise RuntimeError("outer-test identity leaked into development manifests")

    train_rows, val_rows = build_development_split(
        source_rows,
        seed=args.seed,
        validation_subjects=args.validation_subjects,
    )
    if _subject_ids(train_rows) & _subject_ids(val_rows):
        raise RuntimeError("subject leakage between train and validation")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    train_path = args.output_dir / "train_t1ce_binary.jsonl"
    val_path = args.output_dir / "validation_t1ce_binary.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_before_t1ce_model_scoring",
        "task": "T1ce_to_binary_residual_tumor",
        "foreground_raw_labels": list(FOREGROUND_LABELS),
        "excluded_raw_labels": list(EXCLUDED_LABELS),
        "split_algorithm": "sha256(seed:subject_id)_ascending_first_n_validation",
        "seed": args.seed,
        "validation_subjects_requested": args.validation_subjects,
        "summary": {
            "train": {"cases": len(train_rows), "subjects": len(_subject_ids(train_rows))},
            "validation": {"cases": len(val_rows), "subjects": len(_subject_ids(val_rows))},
            "sealed_outer_test": {
                "cases": len(test_cases),
                "subjects": len(test_subjects),
                "opened_by_builder": False,
            },
        },
        "source_development_manifests": {
            str(path): sha256_file(path) for path in source_paths
        },
        "outer_split_manifest": {
            "path": str(args.outer_split_manifest),
            "sha256": sha256_file(args.outer_split_manifest),
            "sealed_test_jsonl_sha256_from_ledger": test_file_sha,
        },
        "output_sha256": {
            train_path.name: sha256_file(train_path),
            val_path.name: sha256_file(val_path),
        },
        "reproduction_status": (
            "subject_clean_ratio_aligned_reconstruction_not_bit_exact_paper_split"
        ),
    }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
