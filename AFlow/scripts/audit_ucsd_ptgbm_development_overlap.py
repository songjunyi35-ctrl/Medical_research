#!/usr/bin/env python3
"""Audit UCSD/BraTS label identity using development-overlap cases only.

The 28 main-release cases belonging to the reused test270 are explicitly
excluded before any download URL is consumed.  This tool never runs a model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.prepare_ucsd_ptgbm_external_confirmation as prepare  # noqa: E402


SCHEMA = "aflow.ucsd_ptgbm.development_overlap_label_identity_audit"
EXPECTED_DEVELOPMENT_CASES = 81
EXPECTED_DEVELOPMENT_SUBJECTS = 63
EXPECTED_REUSED_TEST_CASES_EXCLUDED = 28
EXPECTED_REUSED_TEST_SUBJECTS_EXCLUDED = 20
EXPECTED_UCSD_SHAPE = (256, 256, 256)
EXPECTED_BRATS_SHAPE = (182, 218, 182)
EXPECTED_CENTER_OFFSET = (37, 19, 37)


class OverlapAuditError(RuntimeError):
    pass


def _rows(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    return prepare.read_jsonl(path.resolve(), expected_sha256)


def _resolve_brats_label(raw_root: Path, row: dict[str, Any]) -> Path:
    source_part = str(row.get("source_part", ""))
    if source_part not in {"Part1", "Part2", "Part3"}:
        raise OverlapAuditError(f"invalid source_part for {row.get('case_id')}")
    case_id = str(row["case_id"])
    path = raw_root / source_part.lower() / source_part / case_id / f"{case_id}-seg.nii"
    if not path.is_file():
        raise OverlapAuditError(f"missing development label: {path}")
    return path


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    mapping, _ = prepare.read_participants(args.participants.resolve())
    pairs, releases = prepare.read_release_assets([path.resolve() for path in args.release_json])
    train_rows = _rows(args.train_manifest, prepare.TRAIN_SHA256)
    test_rows = _rows(args.test_manifest, prepare.TEST_SHA256)
    train_by_case = {str(row["case_id"]): row for row in train_rows}
    test_cases = {str(row["case_id"]) for row in test_rows}
    test_subjects = {str(row["subject_id"]) for row in test_rows}
    selected = []
    excluded_test = []
    for ucsd_case_id in sorted(pairs):
        brats_case_id = mapping[ucsd_case_id]
        brats_subject_id = brats_case_id.rsplit("-", 1)[0]
        if brats_case_id in test_cases:
            excluded_test.append({"ucsd_case_id": ucsd_case_id, "brats_case_id": brats_case_id, "brats_subject_id": brats_subject_id})
        elif brats_case_id in train_by_case:
            row = train_by_case[brats_case_id]
            selected.append({
                "ucsd_case_id": ucsd_case_id,
                "brats_case_id": brats_case_id,
                "brats_subject_id": brats_subject_id,
                "official_label_asset": pairs[ucsd_case_id]["label"],
                "development_label_path": str(_resolve_brats_label(args.raw_root.resolve(), row)),
            })
    selected_subjects = {row["brats_subject_id"] for row in selected}
    excluded_test_subjects = {row["brats_subject_id"] for row in excluded_test}
    if len(selected) != EXPECTED_DEVELOPMENT_CASES or len(selected_subjects) != EXPECTED_DEVELOPMENT_SUBJECTS:
        raise OverlapAuditError("development-overlap counts differ from frozen observation")
    if len(excluded_test) != EXPECTED_REUSED_TEST_CASES_EXCLUDED or len(excluded_test_subjects) != EXPECTED_REUSED_TEST_SUBJECTS_EXCLUDED:
        raise OverlapAuditError("reused-test exclusion counts differ from frozen observation")
    if selected_subjects & test_subjects:
        raise OverlapAuditError("development audit selection has reused-test subject overlap")
    return {
        "schema": SCHEMA + ".plan", "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_receipts": releases,
        "development_cases": len(selected), "development_subjects": len(selected_subjects),
        "reused_test_cases_excluded": len(excluded_test), "reused_test_subjects_excluded": len(excluded_test_subjects),
        "reused_test_label_or_image_download_authorized": False,
        "selected": selected,
        "excluded_reused_test_ids": excluded_test,
    }


def _canonical_integer_label(path: Path) -> tuple[np.ndarray, list[float]]:
    image = nib.as_closest_canonical(nib.load(str(path), mmap=True))
    values = np.asanyarray(image.dataobj)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise OverlapAuditError(f"invalid label volume: {path}")
    rounded = np.rint(values)
    if not np.allclose(values, rounded, rtol=0.0, atol=1e-5):
        raise OverlapAuditError(f"non-integer label volume: {path}")
    result = np.ascontiguousarray(rounded, dtype=np.uint8)
    labels = {int(value) for value in np.unique(result)}
    if not labels <= {0, 1, 2, 3, 4}:
        raise OverlapAuditError(f"unexpected label values in {path}: {sorted(labels)}")
    return result, [float(value) for value in nib.affines.voxel_sizes(image.affine)]


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit report: {output}")
    plan = build_plan(args)
    download_root = args.download_dir.resolve()
    records = []
    started = time.time()
    for index, row in enumerate(plan["selected"], start=1):
        asset = row["official_label_asset"]
        official_path = download_root / row["ucsd_case_id"] / asset["name"]
        status = prepare._download_one(asset["url"], official_path, asset["sha256"], int(asset["size_bytes"]))
        official, official_spacing = _canonical_integer_label(official_path)
        development, development_spacing = _canonical_integer_label(Path(row["development_label_path"]))
        if official.shape != EXPECTED_UCSD_SHAPE or development.shape != EXPECTED_BRATS_SHAPE:
            raise OverlapAuditError(f"unexpected grid shape for {row['brats_case_id']}: {official.shape}/{development.shape}")
        slices = tuple(slice(offset, offset + size) for offset, size in zip(EXPECTED_CENTER_OFFSET, EXPECTED_BRATS_SHAPE))
        centered = np.ascontiguousarray(official[slices])
        exact = bool(np.array_equal(centered, development))
        records.append({
            "ucsd_case_id": row["ucsd_case_id"], "brats_case_id": row["brats_case_id"], "brats_subject_id": row["brats_subject_id"],
            "official_label_path": str(official_path), "official_label_sha256": prepare.sha256_file(official_path),
            "download_status": status, "official_shape": list(official.shape), "development_shape": list(development.shape),
            "official_spacing_mm": official_spacing, "development_spacing_mm": development_spacing,
            "center_offset": list(EXPECTED_CENTER_OFFSET), "centered_array_exact": exact,
            "official_labels": [int(value) for value in np.unique(official)], "development_labels": [int(value) for value in np.unique(development)],
        })
        print(f"[{index}/{len(plan['selected'])}] {row['brats_case_id']} exact={exact}", flush=True)
    exact_count = sum(row["centered_array_exact"] for row in records)
    report = {
        "schema": SCHEMA, "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "post-result provenance/format QC only; no model inference and no parameter selection",
        "development_cases": len(records), "development_subjects": len({row["brats_subject_id"] for row in records}),
        "reused_test_cases_excluded_before_download": plan["reused_test_cases_excluded"],
        "reused_test_subjects_excluded_before_download": plan["reused_test_subjects_excluded"],
        "reused_test_label_or_image_downloaded": False,
        "official_shape": list(EXPECTED_UCSD_SHAPE), "development_shape": list(EXPECTED_BRATS_SHAPE),
        "fixed_center_offset": list(EXPECTED_CENTER_OFFSET),
        "centered_arrays_exact_cases": exact_count,
        "all_centered_arrays_exact": exact_count == len(records),
        "elapsed_seconds": time.time() - started,
        "records": records,
        "model_inference": False,
        "result_dependent_external_rerun": False,
        "runner": str(Path(__file__).resolve().relative_to(ROOT)),
        "runner_sha256": prepare.sha256_file(Path(__file__).resolve()),
    }
    prepare.write_json_atomic(output, report)
    if not report["all_centered_arrays_exact"]:
        raise OverlapAuditError(f"only {exact_count}/{len(records)} centered labels are exact; report preserved at {output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--release-json", type=Path, action="append", required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({key: report[key] for key in ("development_cases", "development_subjects", "reused_test_cases_excluded_before_download", "centered_arrays_exact_cases", "all_centered_arrays_exact")}, indent=2))


if __name__ == "__main__":
    main()
