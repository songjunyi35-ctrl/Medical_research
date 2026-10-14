#!/usr/bin/env python3
"""Build the frozen UCSD75 cache from direct TCIA float32 files.

This entry point intentionally reuses the hash-bound historical per-case
preprocessing function without modifying it.  It changes only the source image
bytes and writes a new cache/index; the historical mirror cache is read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_cached import CACHE_SCHEMA, MODEL_SHAPE  # noqa: E402
import scripts.prepare_ucsd_ptgbm_external_confirmation as historical  # noqa: E402


SCHEMA = "aflow.ucsd75_official_float_source_correction.cache"
OLD_INVENTORY_SHA256 = "6cb1f5a5eb740c06c6e8b01e57c7e6aab6dd8a7472f02d8acdd42540e2d16c38"
DIRECT_PREACCESS_SHA256 = "608fe55084caf3bedba1bf946ca36eb69d4ff7ba5181621c15c49a54bcb3299c"
DIRECT_DOWNLOAD_SHA256 = "c0dc16b8e8a435d413a32140244a0f177b1a98c78cd916b83e6f03c360495b6a"
SOURCE_COMPARISON_SHA256 = "1dc05453fce200e11739dafbc1b73d33a5ef1bd491d3174b3ebb42504d101ac7"
HISTORICAL_BUILDER_SHA256 = "fde6bdbf9babd4e98b2e22d47537d4d70485bbff36a8983e11361d464fcfdf9e"
EXPECTED_CASES = 75
EXPECTED_SUBJECTS = 53


class SourceCorrectionError(RuntimeError):
    pass


def _read_frozen(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    path = path.resolve()
    if historical.sha256_file(path) != expected_sha256:
        raise SourceCorrectionError(f"{label} SHA-256 mismatch")
    return json.loads(path.read_text())


def resolve_rows(
    inventory_path: Path,
    direct_preaccess_path: Path,
    direct_download_path: Path,
    source_comparison_path: Path,
) -> list[dict[str, Any]]:
    """Resolve exact official files in the historical frozen case order."""
    inventory = _read_frozen(inventory_path, OLD_INVENTORY_SHA256, "old inventory")
    preaccess = _read_frozen(direct_preaccess_path, DIRECT_PREACCESS_SHA256, "direct preaccess")
    download = _read_frozen(direct_download_path, DIRECT_DOWNLOAD_SHA256, "direct download")
    comparison = _read_frozen(source_comparison_path, SOURCE_COMPARISON_SHA256, "source comparison")
    if historical.sha256_file(Path(historical.__file__).resolve()) != HISTORICAL_BUILDER_SHA256:
        raise SourceCorrectionError("historical preprocessing entry point SHA-256 mismatch")
    if preaccess.get("cases") != EXPECTED_CASES or preaccess.get("subjects") != EXPECTED_SUBJECTS:
        raise SourceCorrectionError("direct preaccess cohort mismatch")
    if not download.get("all_files_present_and_size_matched") or len(download.get("files", [])) != 2 * EXPECTED_CASES:
        raise SourceCorrectionError("direct download receipt is incomplete")
    if comparison.get("status") != "completed_source_identity_audit" or comparison.get("cases") != EXPECTED_CASES:
        raise SourceCorrectionError("source comparison status/count mismatch")
    if comparison.get("label_arrays_equal_count") != EXPECTED_CASES or not comparison.get("all_geometry_equal"):
        raise SourceCorrectionError("official labels/geometry did not pass frozen comparison")

    selected = inventory["selection"]["selected"]
    pre_by_brats = {row["brats_case_id"]: row for row in preaccess["rows"]}
    dl_by_name = {Path(row["path"]).name: row for row in download["files"]}
    cmp_by_ucsd = {row["case_id"]: row for row in comparison["comparisons"]}
    if len(selected) != EXPECTED_CASES or len(pre_by_brats) != EXPECTED_CASES or len(cmp_by_ucsd) != EXPECTED_CASES:
        raise SourceCorrectionError("frozen mapping cardinality mismatch")

    rows: list[dict[str, Any]] = []
    for old in selected:
        case_id = old["brats_case_id"]
        direct = pre_by_brats.get(case_id)
        if direct is None or direct["ucsd_case_id"] != old["ucsd_case_id"]:
            raise SourceCorrectionError(f"official mapping mismatch for {case_id}")
        cmp = cmp_by_ucsd.get(old["ucsd_case_id"])
        if cmp is None or not cmp["files"]["label"]["array_equal"]:
            raise SourceCorrectionError(f"frozen label comparison mismatch for {case_id}")
        if cmp["files"]["image"]["array_equal"]:
            raise SourceCorrectionError(f"official image unexpectedly equals mirror for {case_id}")
        local: dict[str, Any] = {
            "brats_case_id": case_id,
            "brats_subject_id": old["brats_subject_id"],
            "ucsd_case_id": old["ucsd_case_id"],
        }
        for kind in ("image", "label"):
            name = direct["files"][kind]["basename"]
            got = dl_by_name.get(name)
            if got is None or not got.get("exists") or got.get("bytes") != got.get("official_size"):
                raise SourceCorrectionError(f"missing/size-mismatched official {kind} for {case_id}")
            path = Path(got["path"]).resolve()
            expected = cmp["files"][kind]["official_sha256"]
            if got.get("sha256") != expected or historical.sha256_file(path) != expected:
                raise SourceCorrectionError(f"official {kind} SHA-256 mismatch for {case_id}")
            local[f"local_{kind}_path"] = str(path)
            local[f"official_{kind}_sha256"] = expected
        rows.append(local)
    if len({row["brats_case_id"] for row in rows}) != EXPECTED_CASES:
        raise SourceCorrectionError("case IDs are not unique")
    if len({row["brats_subject_id"] for row in rows}) != EXPECTED_SUBJECTS:
        raise SourceCorrectionError("subject count mismatch")
    return rows


def _reproduce_first(row: dict[str, Any], images: Path, labels: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="aflow-ucsd-float-repro-", dir="/tmp") as temporary:
        temp = Path(temporary)
        repro_image = temp / "images.f32"
        repro_label = temp / "labels.u8"
        np.memmap(repro_image, dtype=np.float32, mode="w+", shape=(1, *MODEL_SHAPE)).flush()
        np.memmap(repro_label, dtype=np.uint8, mode="w+", shape=(1, *MODEL_SHAPE)).flush()
        historical._worker_init({"images_path": str(repro_image), "labels_path": str(repro_label)})
        try:
            historical._prepare_case((0, row))
        finally:
            os.close(historical._CFG.pop("images_fd"))
            os.close(historical._CFG.pop("labels_fd"))
        voxels = int(np.prod(MODEL_SHAPE))
        original_image = np.memmap(images, dtype=np.float32, mode="r", shape=(EXPECTED_CASES, *MODEL_SHAPE))[0]
        original_label = np.memmap(labels, dtype=np.uint8, mode="r", shape=(EXPECTED_CASES, *MODEL_SHAPE))[0]
        rebuilt_image = np.memmap(repro_image, dtype=np.float32, mode="r", shape=(voxels,)).reshape(MODEL_SHAPE)
        rebuilt_label = np.memmap(repro_label, dtype=np.uint8, mode="r", shape=(voxels,)).reshape(MODEL_SHAPE)
        return {
            "case_id": row["brats_case_id"],
            "image_byte_exact": bool(np.array_equal(original_image, rebuilt_image)),
            "label_byte_exact": bool(np.array_equal(original_label, rebuilt_label)),
            "image_max_abs_error": float(np.max(np.abs(original_image - rebuilt_image))),
        }


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    rows = resolve_rows(args.inventory, args.direct_preaccess, args.direct_download, args.source_comparison)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty cache: {output}")
    output.mkdir(parents=True, exist_ok=True)
    images = output / "images.f32"
    labels = output / "labels.u8"
    np.memmap(images, dtype=np.float32, mode="w+", shape=(len(rows), *MODEL_SHAPE)).flush()
    np.memmap(labels, dtype=np.uint8, mode="w+", shape=(len(rows), *MODEL_SHAPE)).flush()
    cfg = {"images_path": str(images), "labels_path": str(labels)}
    started = time.time()
    results: list[tuple[int, dict[str, Any]]] = []
    with mp.get_context("spawn").Pool(
        processes=args.workers,
        initializer=historical._worker_init,
        initargs=(cfg,),
    ) as pool:
        for completed, result in enumerate(pool.imap_unordered(historical._prepare_case, enumerate(rows), chunksize=1), start=1):
            results.append(result)
            print(f"[{completed}/{len(rows)}] cached direct-TCIA float32 case", flush=True)
    results.sort(key=lambda item: item[0])
    qc = [item[1] for item in results]
    if [row["case_id"] for row in qc] != [row["brats_case_id"] for row in rows]:
        raise SourceCorrectionError("cache order differs from frozen order")
    if any(row["source_wt_empty"] != row["resized_wt_empty"] for row in qc):
        raise SourceCorrectionError("resize changed WT empty status")
    if any(not set(row["raw_labels"]) <= {0, 1, 2, 3, 4} for row in qc):
        raise SourceCorrectionError("label domain escaped frozen contract")
    reproduction = _reproduce_first(rows[0], images, labels)
    if not reproduction["image_byte_exact"] or not reproduction["label_byte_exact"]:
        raise SourceCorrectionError("first-case independent cache reproduction failed")

    index = {
        "schema": CACHE_SCHEMA,
        "schema_version": 1,
        "external_schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_version": "direct_TCIA_package_1019_float32",
        "source_correction_reuses_old_ucsd75": True,
        "independent_new_cohort": False,
        "count": len(rows),
        "subjects": len({row["brats_subject_id"] for row in rows}),
        "case_ids": [row["brats_case_id"] for row in rows],
        "subject_ids": [row["brats_subject_id"] for row in rows],
        "model_shape": list(MODEL_SHAPE),
        "image_dtype": "float32",
        "label_dtype": "uint8",
        "split": "external_reused_source_correction",
        "test_access": True,
        "external_confirmation": False,
        "not_pristine_blind": True,
        "preprocessing": {
            "order": ["canonical_RAS", "N4_shrink_2", "image_nonzero_bounding_box_crop", "source_nonzero_population_zscore", "resize_to_128_cubic_grid"],
            "spatial": "braincrop_resize",
            "output_resolution": 128,
            "image_interpolation": "trilinear_align_corners_false",
            "label_interpolation": "nearest-exact",
            "normalization": "source nonzero-voxel population z-score before resize",
            "uses_gt_for_localization": False,
            "n4_bias_field_correction": True,
            "n4_shrink_factor": 2,
        },
        "target": {"name": "WT", "foreground_raw_labels": [1, 2, 3], "excluded_raw_labels": [4]},
        "frozen_evidence": {
            "old_inventory_sha256": OLD_INVENTORY_SHA256,
            "direct_preaccess_sha256": DIRECT_PREACCESS_SHA256,
            "direct_download_sha256": DIRECT_DOWNLOAD_SHA256,
            "source_comparison_sha256": SOURCE_COMPARISON_SHA256,
            "historical_builder_sha256": HISTORICAL_BUILDER_SHA256,
        },
        "source_files": [
            {
                "case_id": row["brats_case_id"],
                "ucsd_case_id": row["ucsd_case_id"],
                "image_sha256": row["official_image_sha256"],
                "label_sha256": row["official_label_sha256"],
            }
            for row in rows
        ],
        "qc": {
            "cases": qc,
            "all_raw_labels": sorted({value for row in qc for value in row["raw_labels"]}),
            "source_wt_empty_cases": sum(row["source_wt_empty"] for row in qc),
            "resized_wt_empty_cases": sum(row["resized_wt_empty"] for row in qc),
            "all_finite": True,
            "image_label_geometry_exact_after_canonicalization": True,
            "official_source_labels_exact_to_historical_mirror": True,
            "first_case_reproduction": reproduction,
        },
        "images_sha256": historical.sha256_file(images),
        "labels_sha256": historical.sha256_file(labels),
        "builder": str(Path(__file__).resolve().relative_to(ROOT)),
        "builder_sha256": historical.sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": time.time() - started,
        "access": {"official_float_image_voxels": True, "official_label_voxels": True, "model_inference": False},
    }
    historical.write_json_atomic(output / "index.json", index)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--direct-preaccess", type=Path, required=True)
    parser.add_argument("--direct-download", type=Path, required=True)
    parser.add_argument("--source-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        raise ValueError("workers must be in [1,16]")
    result = build_cache(args)
    print(json.dumps({
        "status": "passed",
        "count": result["count"],
        "subjects": result["subjects"],
        "images_sha256": result["images_sha256"],
        "labels_sha256": result["labels_sha256"],
        "first_case_reproduction": result["qc"]["first_case_reproduction"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
