#!/usr/bin/env python
"""Build the one-time authorized official-1.5-mm cache for the sealed test.

This file deliberately does not broaden the development cache builder.  It
accepts exactly the frozen 270-case VA test manifest and delegates every voxel
transform to ``build_t1ce_wt_official15_cache.preprocess_row`` so test tensors
are constructed by the same implementation as train/validation.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_prompt import SCHEMA as INNER_SCHEMA, validate_row  # noqa: E402
from scripts import build_t1ce_wt_official15_cache as official  # noqa: E402

AUTHORIZATION = "USER_AUTHORIZED_270_TEST_2026-08-12"
TEST_MANIFEST = ROOT / (
    "data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/"
    "sealed_test/test_cases_t1ce.jsonl"
)
TEST_MANIFEST_SHA256 = "a5ddc0ed2c0b269431758b14061da3c6163dd0c48133b60a2ef8efd7f23aa8f9"
TRAIN_MANIFEST = ROOT / "data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl"
TRAIN_MANIFEST_SHA256 = "6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65"
VAL_MANIFEST = ROOT / "data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/validation_t1ce_wt.jsonl"
VAL_MANIFEST_SHA256 = "f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb"
TEST_COUNT = 270


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def adapt_test_row(row: dict) -> dict:
    modalities = row.get("modalities")
    if row.get("schema") != "aflow.brats2024.random_subject_train_test_split":
        raise ValueError("sealed test row has an unexpected schema")
    if row.get("split") != "test":
        raise ValueError("sealed test contains a non-test row")
    if not isinstance(modalities, dict) or set(modalities) != {"t1c"}:
        raise ValueError("sealed test row must project modalities to T1ce only")
    adapted = {
        "schema": INNER_SCHEMA,
        "schema_version": 1,
        "case_id": str(row["case_id"]),
        "subject_id": str(row["subject_id"]),
        "patient_id": str(row["subject_id"]),
        "image_path": str(modalities["t1c"]),
        "label_path": str(row["seg_path"]),
        "modality": "t1ce",
        "target": "WT",
        "foreground_labels": [1, 2, 3],
        "excluded_labels": [4],
        "split": "test",
        "source_part": row.get("source_part"),
        "timepoint": row.get("timepoint"),
    }
    validate_row(adapted)
    return adapted


def read_and_audit_rows(manifest: Path) -> tuple[list[dict], list[dict]]:
    if manifest.resolve() != TEST_MANIFEST.resolve():
        raise RuntimeError("only the frozen VA sealed-test manifest is accepted")
    if official.sha256_file(manifest) != TEST_MANIFEST_SHA256:
        raise RuntimeError("sealed-test manifest SHA-256 mismatch")
    if official.sha256_file(TRAIN_MANIFEST) != TRAIN_MANIFEST_SHA256:
        raise RuntimeError("train manifest SHA-256 mismatch")
    if official.sha256_file(VAL_MANIFEST) != VAL_MANIFEST_SHA256:
        raise RuntimeError("validation manifest SHA-256 mismatch")
    raw_rows = _jsonl(manifest)
    if len(raw_rows) != TEST_COUNT:
        raise RuntimeError(f"sealed test must contain {TEST_COUNT} rows")
    rows = [adapt_test_row(row) for row in raw_rows]
    case_ids = [row["case_id"] for row in rows]
    subjects = [row["subject_id"] for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("sealed test contains duplicate case IDs")
    development = _jsonl(TRAIN_MANIFEST) + _jsonl(VAL_MANIFEST)
    dev_cases = {str(row["case_id"]) for row in development}
    dev_subjects = {str(row["subject_id"]) for row in development}
    if dev_cases.intersection(case_ids):
        raise RuntimeError("sealed test overlaps development case IDs")
    if dev_subjects.intersection(subjects):
        raise RuntimeError("sealed test overlaps development subjects")
    return raw_rows, rows


def build(args: argparse.Namespace) -> dict:
    if args.authorization != AUTHORIZATION:
        raise RuntimeError("explicit sealed-test authorization token is required")
    manifest = args.manifest.resolve()
    raw_rows, rows = read_and_audit_rows(manifest)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty test cache: {output}")
    output.mkdir(parents=True, exist_ok=True)

    count = len(rows)
    shape = official.MODEL_SHAPE
    images_path = output / "images.f32"
    labels_path = output / "labels.u8"
    np.memmap(images_path, dtype=np.float32, mode="w+", shape=(count, *shape)).flush()
    np.memmap(labels_path, dtype=np.uint8, mode="w+", shape=(count, *shape)).flush()
    cfg = {"images_path": str(images_path), "labels_path": str(labels_path)}
    started = time.time()
    records: list[tuple[int, dict]] = []
    with mp.get_context("spawn").Pool(
        processes=args.workers,
        initializer=official._worker_init,
        initargs=(cfg,),
    ) as pool:
        for completed, record in enumerate(
            pool.imap_unordered(official._process, enumerate(rows), chunksize=1),
            start=1,
        ):
            records.append(record)
            if completed % 25 == 0 or completed == count:
                elapsed = max(time.time() - started, 1e-9)
                rate = completed / elapsed
                print(
                    f"{completed}/{count} {rate:.2f} case/s "
                    f"eta={(count - completed) / max(rate, 1e-9):.0f}s",
                    flush=True,
                )
    records.sort(key=lambda item: item[0])
    qc_rows = [record for _, record in records]
    retention = np.asarray([float(row["wt_crop_retention"]) for row in qc_rows])
    development_index = json.loads(
        (ROOT / "/tmp/aflow_t1ce_wt_official15_va_128/validation/index.json").read_text()
    ) if False else None
    # Keep this literal rather than importing a mutable training cache index.
    preprocessing = {
        "target": "T1ce single-channel / WT={1,2,3}; RC={4} excluded",
        "target_spacing_mm": list(official.TARGET_SPACING),
        "order": [
            "WT123 binary LabelMap",
            "TorchIO Resample 1.5mm isotropic",
            "TorchIO ToCanonical",
            "TorchIO CropOrPad(mask_name=label, target_shape=128^3)",
            "positive-voxel sample z-score applied to whole patch",
        ],
        "spatial": "gt_crop_1p5mm",
        "image_interpolation": "linear",
        "label_interpolation": "nearest (TorchIO LabelMap)",
        "normalization": "post-crop positive-voxel sample z-score",
        "uses_gt_for_localization": True,
        "skull_stripping": "not applied; BraTS volumes are already stripped",
        "n4_bias_field_correction": False,
        "augmentation": "none on validation/test; RandomFlip was train-only",
        "source": "uni-medical/SAM-Med3D prepare_data_from_nnUNet.py + infer_utils.py",
    }
    elapsed = time.time() - started
    index = {
        "schema": official.CACHE_SCHEMA,
        "schema_version": official.CACHE_SCHEMA_VERSION,
        "protocol_schema": official.PROTOCOL_SCHEMA,
        "manifest": str(manifest.relative_to(ROOT)),
        "manifest_sha256": TEST_MANIFEST_SHA256,
        "count": count,
        "model_shape": list(shape),
        "image_dtype": "float32",
        "label_dtype": "uint8",
        "preprocessing": preprocessing,
        "case_ids": [row["case_id"] for row in rows],
        "subject_ids": [row["subject_id"] for row in rows],
        "per_case_positive_voxels": {
            row["case_id"]: int(qc["wt_voxels_after_crop"])
            for row, qc in zip(rows, qc_rows)
        },
        "qc": {
            "all_source_spacing_mm": sorted(
                {tuple(row["source_spacing"]) for row in qc_rows}
            ),
            "all_resampled_spacing_mm": sorted(
                {tuple(row["resampled_spacing"]) for row in qc_rows}
            ),
            "crop_retention_min": float(retention.min()),
            "crop_retention_mean": float(retention.mean()),
            "cases_with_crop_loss": int((retention < 1.0).sum()),
            "cases_with_more_than_1pct_crop_loss": int((retention < 0.99).sum()),
            "per_case": qc_rows,
        },
        "build_seconds": round(elapsed, 2),
        "builder": str(Path(__file__).resolve().relative_to(ROOT)),
        "builder_sha256": official.sha256_file(Path(__file__).resolve()),
        "delegated_preprocess_builder": str(
            Path(official.__file__).resolve().relative_to(ROOT)
        ),
        "delegated_preprocess_builder_sha256": official.sha256_file(
            Path(official.__file__).resolve()
        ),
        "images_sha256": official.sha256_file(images_path),
        "labels_sha256": official.sha256_file(labels_path),
        "split": "test",
        "test_access": True,
        "authorization": AUTHORIZATION,
        "development_overlap": {
            "case_ids": 0,
            "subject_ids": 0,
            "train_manifest_sha256": TRAIN_MANIFEST_SHA256,
            "validation_manifest_sha256": VAL_MANIFEST_SHA256,
        },
        "raw_manifest_rows": len(raw_rows),
    }
    temporary = output / "index.json.tmp"
    temporary.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output / "index.json")
    print(f"test cache written: {output} ({elapsed:.1f}s)", flush=True)
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=TEST_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


if __name__ == "__main__":
    build(parse_args())
