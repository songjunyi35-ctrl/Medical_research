#!/usr/bin/env python
"""Build the explicitly authorized N4 + GT-centred 128^3 test cache.

This wrapper keeps the development cache builder unchanged.  It accepts only
the frozen 270-case VA manifest, audits it against the exact development
manifests used by the residual refiner, and delegates every voxel transform to
``build_t1ce_wt_cache.py`` with the training-time arguments fixed in code.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_t1ce_wt_cache as legacy  # noqa: E402


AUTHORIZATION = "USER_AUTHORIZED_RESIDUAL_REFINER_TEST270_2026-08-13"
TEST_MANIFEST = ROOT / (
    "data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/"
    "sealed_test/test_cases_t1ce.jsonl"
)
TEST_MANIFEST_SHA256 = "a5ddc0ed2c0b269431758b14061da3c6163dd0c48133b60a2ef8efd7f23aa8f9"
DEVELOPMENT_TRAIN_MANIFEST = ROOT / (
    "data/datasets/brats2024_sammed3d_t1ce_wt_inner90_10_seed20260723_v1/"
    "train_t1ce_wt.jsonl"
)
DEVELOPMENT_TRAIN_SHA256 = "ea21e2306321b71d821146ed01198f52b612d390c0736f8703f81a259c06c23e"
DEVELOPMENT_VALIDATION_MANIFEST = ROOT / (
    "data/datasets/brats2024_sammed3d_t1ce_wt_inner90_10_seed20260723_v1/"
    "validation_t1ce_wt.jsonl"
)
DEVELOPMENT_VALIDATION_SHA256 = "fc750ab15266b8a53c6d9a7ff62bd043b10cc21733e9b68354a2a989fd1f488c"
TEST_COUNT = 270
TEST_SUBJECTS = 119


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_and_audit() -> tuple[list[dict], list[dict]]:
    bindings = (
        (TEST_MANIFEST, TEST_MANIFEST_SHA256),
        (DEVELOPMENT_TRAIN_MANIFEST, DEVELOPMENT_TRAIN_SHA256),
        (DEVELOPMENT_VALIDATION_MANIFEST, DEVELOPMENT_VALIDATION_SHA256),
    )
    for path, expected in bindings:
        actual = legacy.sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"manifest SHA mismatch for {path}: {actual} != {expected}")
    test_rows = _read_jsonl(TEST_MANIFEST)
    development_rows = _read_jsonl(DEVELOPMENT_TRAIN_MANIFEST) + _read_jsonl(
        DEVELOPMENT_VALIDATION_MANIFEST
    )
    if len(test_rows) != TEST_COUNT:
        raise RuntimeError(f"test manifest must contain {TEST_COUNT} cases")
    case_ids = [str(row["case_id"]) for row in test_rows]
    subject_ids = [str(row["subject_id"]) for row in test_rows]
    if len(set(case_ids)) != TEST_COUNT:
        raise RuntimeError("test manifest contains duplicate cases")
    if len(set(subject_ids)) != TEST_SUBJECTS:
        raise RuntimeError("unexpected test subject count")
    for row in test_rows:
        if row.get("split") != "test":
            raise RuntimeError("test manifest contains a non-test row")
        modalities = row.get("modalities")
        if not isinstance(modalities, dict) or set(modalities) != {"t1c"}:
            raise RuntimeError("test manifest must project to T1ce only")
        if not row.get("seg_path"):
            raise RuntimeError("test row lacks segmentation")
    dev_cases = {str(row["case_id"]) for row in development_rows}
    dev_subjects = {str(row["subject_id"]) for row in development_rows}
    if dev_cases.intersection(case_ids):
        raise RuntimeError("test/development case overlap")
    if dev_subjects.intersection(subject_ids):
        raise RuntimeError("test/development subject overlap")
    return test_rows, development_rows


def build(args: argparse.Namespace) -> dict:
    if args.authorization != AUTHORIZATION:
        raise RuntimeError("explicit residual-refiner test authorization is required")
    if args.manifest.resolve() != TEST_MANIFEST.resolve():
        raise RuntimeError("only the frozen 270-case VA test manifest is accepted")
    test_rows, _ = read_and_audit()
    delegated_args = argparse.Namespace(
        manifest=str(TEST_MANIFEST),
        output_dir=str(args.output_dir.resolve()),
        workers=int(args.workers),
        n4=True,
        n4_shrink=2,
        spatial="gt_crop",
        split="test",
        force=False,
        verify=0,
    )
    index = legacy.build(delegated_args)
    expected_preprocessing = {
        "spatial": "gt_crop",
        "normalization": "post-crop positive-voxel sample z-score",
        "uses_gt_for_localization": True,
        "n4_bias_field_correction": True,
        "n4_shrink_factor": 2,
    }
    for key, expected in expected_preprocessing.items():
        if index["preprocessing"].get(key) != expected:
            raise RuntimeError(f"preprocessing binding mismatch for {key}")
    if index.get("manifest_sha256") != TEST_MANIFEST_SHA256:
        raise RuntimeError("built cache is bound to the wrong manifest")
    if index.get("count") != TEST_COUNT or index.get("split") != "test":
        raise RuntimeError("built cache has the wrong split or count")
    index.update(
        {
            "authorization": AUTHORIZATION,
            "subject_ids": [str(row["subject_id"]) for row in test_rows],
            "development_overlap": {"case_ids": 0, "subject_ids": 0},
            "development_manifests": {
                "train_sha256": DEVELOPMENT_TRAIN_SHA256,
                "validation_sha256": DEVELOPMENT_VALIDATION_SHA256,
            },
            "authorized_wrapper": str(Path(__file__).resolve().relative_to(ROOT)),
            "authorized_wrapper_sha256": legacy.sha256_file(Path(__file__).resolve()),
            "delegated_builder": str(Path(legacy.__file__).resolve().relative_to(ROOT)),
            "delegated_builder_sha256": legacy.sha256_file(Path(legacy.__file__).resolve()),
        }
    )
    index_path = args.output_dir.resolve() / "index.json"
    temporary = index_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, index_path)
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=TEST_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


if __name__ == "__main__":
    build(parse_args())
