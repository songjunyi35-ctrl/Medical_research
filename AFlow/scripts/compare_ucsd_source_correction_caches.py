#!/usr/bin/env python3
"""Compare frozen official-float and historical mirror-uint8 UCSD75 caches."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_cached import MODEL_SHAPE  # noqa: E402
from scripts.prepare_ucsd_ptgbm_external_confirmation import sha256_file, write_json_atomic  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-cache", type=Path, required=True)
    parser.add_argument("--expected-official-index-sha256", required=True)
    parser.add_argument("--mirror-cache", type=Path, required=True)
    parser.add_argument("--expected-mirror-index-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite cache comparison")
    official_index_path = args.official_cache.resolve() / "index.json"
    mirror_index_path = args.mirror_cache.resolve() / "index.json"
    if sha256_file(official_index_path) != args.expected_official_index_sha256:
        raise RuntimeError("official index SHA mismatch")
    if sha256_file(mirror_index_path) != args.expected_mirror_index_sha256:
        raise RuntimeError("mirror index SHA mismatch")
    official_index = json.loads(official_index_path.read_text())
    mirror_index = json.loads(mirror_index_path.read_text())
    if official_index["case_ids"] != mirror_index["case_ids"] or official_index["subject_ids"] != mirror_index["subject_ids"]:
        raise RuntimeError("cache order mismatch")
    count = len(official_index["case_ids"])
    official_images = np.memmap(args.official_cache.resolve() / "images.f32", dtype=np.float32, mode="r", shape=(count, *MODEL_SHAPE))
    mirror_images = np.memmap(args.mirror_cache.resolve() / "images.f32", dtype=np.float32, mode="r", shape=(count, *MODEL_SHAPE))
    official_labels = np.memmap(args.official_cache.resolve() / "labels.u8", dtype=np.uint8, mode="r", shape=(count, *MODEL_SHAPE))
    mirror_labels = np.memmap(args.mirror_cache.resolve() / "labels.u8", dtype=np.uint8, mode="r", shape=(count, *MODEL_SHAPE))
    old_qc = {row["case_id"]: row for row in mirror_index["qc"]["cases"]}
    new_qc = {row["case_id"]: row for row in official_index["qc"]["cases"]}
    rows = []
    for slot, case_id in enumerate(official_index["case_ids"]):
        image_delta = np.asarray(official_images[slot], dtype=np.float32) - np.asarray(mirror_images[slot], dtype=np.float32)
        label_equal = bool(np.array_equal(official_labels[slot], mirror_labels[slot]))
        rows.append({
            "case_id": case_id,
            "subject_id": official_index["subject_ids"][slot],
            "image_array_equal": bool(np.array_equal(official_images[slot], mirror_images[slot])),
            "image_mean_abs_error": float(np.mean(np.abs(image_delta), dtype=np.float64)),
            "image_max_abs_error": float(np.max(np.abs(image_delta))),
            "label_array_equal": label_equal,
            "label_unequal_voxels": int(np.count_nonzero(official_labels[slot] != mirror_labels[slot])),
            "crop_origin_equal": new_qc[case_id]["crop_origin"] == old_qc[case_id]["crop_origin"],
            "crop_extent_equal": new_qc[case_id]["crop_extent"] == old_qc[case_id]["crop_extent"],
            "official_resized_wt_voxels": int(new_qc[case_id]["resized_wt_voxels"]),
            "mirror_resized_wt_voxels": int(old_qc[case_id]["resized_wt_voxels"]),
        })
    result = {
        "schema": "aflow.ucsd75_official_float_source_correction.cache_comparison",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_inference": False,
        "prediction_metrics": False,
        "cases": count,
        "subjects": len(set(official_index["subject_ids"])),
        "official_index_sha256": args.expected_official_index_sha256,
        "mirror_index_sha256": args.expected_mirror_index_sha256,
        "summary": {
            "image_array_equal_cases": sum(row["image_array_equal"] for row in rows),
            "label_array_equal_cases": sum(row["label_array_equal"] for row in rows),
            "crop_origin_equal_cases": sum(row["crop_origin_equal"] for row in rows),
            "crop_extent_equal_cases": sum(row["crop_extent_equal"] for row in rows),
            "image_mean_abs_error_case_median": statistics.median(row["image_mean_abs_error"] for row in rows),
            "image_max_abs_error_overall": max(row["image_max_abs_error"] for row in rows),
            "label_unequal_voxels_total": sum(row["label_unequal_voxels"] for row in rows),
            "resized_wt_voxel_delta_total": sum(row["official_resized_wt_voxels"] - row["mirror_resized_wt_voxels"] for row in rows),
        },
        "per_case": rows,
        "comparator": str(Path(__file__).resolve().relative_to(ROOT)),
        "comparator_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_atomic(args.output.resolve(), result)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
