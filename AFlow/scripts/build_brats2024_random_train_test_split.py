#!/usr/bin/env python3
"""Build a reproducible subject-clean 80/20 split over all 1350 BraTS scans."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping


SCHEMA = "aflow.brats2024.random_subject_train_test_split"
SCHEMA_VERSION = 1
DEFAULT_SEED = 20260722
CASE_RE = re.compile(r"^(BraTS-GLI-\d{5})-(\d{3})$")
MODALITIES = ("t1n", "t1c", "t2w", "t2f")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def normalize_metadata(value: Mapping[str, object], source_path: Path) -> dict:
    case_id = str(value.get("case_id", ""))
    match = CASE_RE.fullmatch(case_id)
    if match is None:
        raise ValueError(f"invalid case id in {source_path}: {case_id!r}")
    subject_id = str(value.get("patient_id", ""))
    if subject_id != match.group(1):
        raise ValueError(f"case/subject mismatch in {source_path}: {case_id}/{subject_id}")
    source_files = value.get("source_files")
    if not isinstance(source_files, Mapping):
        raise ValueError(f"missing source_files in {source_path}")
    paths = {}
    for name in (*MODALITIES, "seg"):
        item = source_files.get(name)
        if not isinstance(item, Mapping) or not item.get("path"):
            raise ValueError(f"missing source_files.{name}.path in {source_path}")
        paths[name] = str(item["path"])
    return {
        "case_id": case_id,
        "subject_id": subject_id,
        "timepoint": match.group(2),
        "source_part": value.get("source_part"),
        "source_case_dir": str(value.get("source_case_dir", "")),
        "modalities": {name: paths[name] for name in MODALITIES},
        "seg_path": paths["seg"],
    }


def read_case_metadata(metadata_dir: Path) -> list[dict]:
    paths = sorted(metadata_dir.glob("BraTS-GLI-*.json"))
    if not paths:
        raise ValueError(f"no case metadata found in {metadata_dir}")
    rows = []
    seen = set()
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError(f"metadata is not an object: {path}")
        row = normalize_metadata(value, path)
        if row["case_id"] in seen:
            raise ValueError(f"duplicate case id: {row['case_id']}")
        seen.add(row["case_id"])
        rows.append(row)
    return rows


def choose_exact_test_subjects(
    case_counts: Mapping[str, int], *, seed: int, target_cases: int
) -> set[str]:
    """Randomize subjects, then solve an exact scan-count subset without leakage."""
    if not 0 < target_cases < sum(case_counts.values()):
        raise ValueError("target_cases must leave non-empty train and test sets")
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


def split_rows(rows: Iterable[dict], *, seed: int, test_cases: int) -> tuple[list[dict], list[dict]]:
    rows = list(rows)
    by_subject: dict[str, list[dict]] = defaultdict(list)
    seen_cases = set()
    for row in rows:
        case_id = str(row["case_id"])
        if case_id in seen_cases:
            raise ValueError(f"duplicate case id: {case_id}")
        seen_cases.add(case_id)
        by_subject[str(row["subject_id"])].append(row)
    test_subjects = choose_exact_test_subjects(
        {subject: len(items) for subject, items in by_subject.items()},
        seed=seed,
        target_cases=test_cases,
    )
    train = []
    test = []
    for subject_id in sorted(by_subject):
        destination = test if subject_id in test_subjects else train
        split = "test" if subject_id in test_subjects else "train"
        for source in sorted(by_subject[subject_id], key=lambda item: item["case_id"]):
            row = {
                "schema": SCHEMA,
                "schema_version": SCHEMA_VERSION,
                **source,
                "split": split,
            }
            destination.append(row)
    return train, test


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def subject_ids(rows: Iterable[dict]) -> set[str]:
    return {str(row["subject_id"]) for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-cases", type=int, default=1350)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    args = parser.parse_args()

    if not 0.0 < args.test_ratio < 1.0:
        raise ValueError("test-ratio must be between zero and one")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    source_rows = read_case_metadata(args.metadata_dir)
    if len(source_rows) != args.expected_cases:
        raise RuntimeError(
            f"expected {args.expected_cases} cases, found {len(source_rows)}"
        )
    target_test_cases = round(len(source_rows) * args.test_ratio)
    train_rows, test_rows = split_rows(
        source_rows, seed=args.seed, test_cases=target_test_cases
    )
    train_subjects = subject_ids(train_rows)
    test_subjects = subject_ids(test_rows)
    if train_subjects & test_subjects:
        raise RuntimeError("subject leakage between train and test")
    if {row["case_id"] for row in train_rows} & {row["case_id"] for row in test_rows}:
        raise RuntimeError("case leakage between train and test")
    if len(train_rows) + len(test_rows) != len(source_rows):
        raise RuntimeError("split union does not cover the source cohort")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    sealed_dir = args.output_dir / "sealed_test"
    sealed_dir.mkdir()
    train_path = args.output_dir / "train_cases.jsonl"
    test_path = sealed_dir / "test_cases.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(test_path, test_rows)
    seal = {
        "policy": "Do not use test images or labels until the final configuration is frozen.",
        "test_manifest": str(test_path.relative_to(args.output_dir)),
        "test_manifest_sha256": sha256_file(test_path),
        "test_nifti_opened_by_builder": False,
    }
    seal_path = sealed_dir / "SEAL.json"
    seal_path.write_text(
        json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    source_index = [
        {
            "case_id": row["case_id"],
            "subject_id": row["subject_id"],
            "source_part": row["source_part"],
            "source_case_dir": row["source_case_dir"],
        }
        for row in source_rows
    ]
    manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_random_subject_clean_80_20_before_training",
        "seed": args.seed,
        "split_algorithm": (
            "python_random_seeded_subject_shuffle_then_exact_test_case_subset_sum"
        ),
        "group_unit": "subject_id_all_longitudinal_timepoints_together",
        "requested_ratio": {"train": 0.8, "test": args.test_ratio},
        "achieved": {
            "train": {
                "cases": len(train_rows),
                "subjects": len(train_subjects),
                "case_ratio": len(train_rows) / len(source_rows),
            },
            "test": {
                "cases": len(test_rows),
                "subjects": len(test_subjects),
                "case_ratio": len(test_rows) / len(source_rows),
            },
        },
        "checks": {
            "case_overlap": 0,
            "subject_overlap": 0,
            "union_cases": len(source_rows),
            "test_nifti_opened_by_builder": False,
        },
        "source": {
            "metadata_dir": str(args.metadata_dir.resolve()),
            "case_index_sha256": canonical_sha256(source_index),
            "cases": len(source_rows),
            "subjects": len(subject_ids(source_rows)),
        },
        "outputs": {
            str(train_path.relative_to(args.output_dir)): sha256_file(train_path),
            str(test_path.relative_to(args.output_dir)): sha256_file(test_path),
            str(seal_path.relative_to(args.output_dir)): sha256_file(seal_path),
        },
    }
    manifest_path = args.output_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["achieved"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
