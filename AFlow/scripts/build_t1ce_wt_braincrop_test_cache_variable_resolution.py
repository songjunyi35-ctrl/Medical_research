#!/usr/bin/env python3
"""Build the explicitly authorized true-160^3 cache for internal test270."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import SimpleITK as sitk


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.build_t1ce_wt_braincrop_cache_variable_resolution as base  # noqa: E402


TEST_CACHE_SCHEMA = "aflow.sammed3d.t1ce_wt_braincrop_variable_resolution_test_cache"
AUTHORIZATION = "USER_AUTHORIZED_EPOCH200_TRUE160_TEST270_2026-08-18"


def read_test_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            case_id = str(value.get("case_id", ""))
            subject_id = str(value.get("subject_id", ""))
            modalities = value.get("modalities")
            if value.get("split") != "test":
                raise ValueError(f"{path}:{line_number} is not a test row")
            if not isinstance(modalities, dict) or set(modalities) != {"t1c"}:
                raise ValueError(f"{path}:{line_number} violates T1ce-only projection")
            if subject_id != case_id.rsplit("-", 1)[0] or not value.get("seg_path"):
                raise ValueError(f"{path}:{line_number} invalid case/subject/label")
            image_path = Path(str(modalities["t1c"]))
            label_path = Path(str(value["seg_path"]))
            if not image_path.is_file() or not label_path.is_file():
                raise FileNotFoundError(f"missing authorized raw test pair for {case_id}")
            rows.append(
                {
                    "case_id": case_id,
                    "subject_id": subject_id,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "source_part": str(value.get("source_part", "unknown")),
                }
            )
    case_ids = [row["case_id"] for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("test manifest contains duplicate case IDs")
    return rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.authorization != AUTHORIZATION:
        raise RuntimeError("explicit epoch200 true-160 test270 authorization is required")
    manifest = args.test_manifest.resolve()
    if base.sha256_file(manifest) != args.expected_manifest_sha256:
        raise ValueError("test270 manifest SHA-256 mismatch")
    rows = read_test_rows(manifest)
    if len(rows) != args.expected_cases:
        raise ValueError(f"test manifest has {len(rows)} rows, expected {args.expected_cases}")
    if len({row["subject_id"] for row in rows}) != args.expected_subjects:
        raise ValueError("test manifest subject count mismatch")
    shape = base.cubic_shape(args.resolution)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty test cache: {output}")
    output.mkdir(parents=True, exist_ok=True)
    images_path = output / "images.f32"
    labels_path = output / "labels.u8"
    np.memmap(images_path, dtype=np.float32, mode="w+", shape=(len(rows), *shape)).flush()
    np.memmap(labels_path, dtype=np.uint8, mode="w+", shape=(len(rows), *shape)).flush()
    cfg = {
        "n4": True,
        "n4_shrink": 2,
        "model_shape": list(shape),
        "images_path": str(images_path),
        "labels_path": str(labels_path),
    }
    started = time.time()
    results: list[tuple[int, dict[str, Any]]] = []
    with mp.get_context("spawn").Pool(
        processes=args.workers, initializer=base._worker_init, initargs=(cfg,)
    ) as pool:
        for completed, result in enumerate(
            pool.imap_unordered(base._process, enumerate(rows), chunksize=1), start=1
        ):
            results.append(result)
            if completed % 25 == 0 or completed == len(rows):
                elapsed = max(time.time() - started, 1e-9)
                rate = completed / elapsed
                print(
                    f"{completed}/{len(rows)} rate={rate:.3f} case/s "
                    f"eta={(len(rows) - completed) / max(rate, 1e-9):.0f}s",
                    flush=True,
                )
    results.sort(key=lambda item: item[0])
    metadata = [item[1] for item in results]

    images = np.memmap(
        images_path, dtype=np.float32, mode="r", shape=(len(rows), *shape)
    )
    labels = np.memmap(
        labels_path, dtype=np.uint8, mode="r", shape=(len(rows), *shape)
    )
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    base._CFG.update({"n4": True, "n4_shrink": 2, "model_shape": shape})
    verify_slots = np.linspace(
        0, len(rows) - 1, num=min(args.verify_cases, len(rows)), dtype=int
    ).tolist()
    verification = []
    for slot in verify_slots:
        image, label, reproduced = base._prepare(rows[slot])
        result = {
            "slot": slot,
            "case_id": rows[slot]["case_id"],
            "image_exact": bool(np.array_equal(image, np.asarray(images[slot]))),
            "label_exact": bool(np.array_equal(label, np.asarray(labels[slot]))),
            "image_max_abs_difference": float(
                np.max(np.abs(image - np.asarray(images[slot])))
            ),
            "source_positive_voxels_exact": (
                reproduced["source_positive_voxels"]
                == metadata[slot]["source_positive_voxels"]
            ),
        }
        if not all(
            result[key]
            for key in ("image_exact", "label_exact", "source_positive_voxels_exact")
        ):
            raise RuntimeError(f"raw test reproduction failed: {result}")
        verification.append(result)
        print(f"exact raw test reproduction: {result['case_id']}", flush=True)

    source_positive = {
        item["case_id"]: item["source_positive_voxels"] for item in metadata
    }
    resized_positive = {
        item["case_id"]: item["resized_positive_voxels"] for item in metadata
    }
    index = {
        "schema": TEST_CACHE_SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": AUTHORIZATION,
        "created_from_raw_nifti": True,
        "manifest": str(manifest.relative_to(ROOT)),
        "manifest_sha256": base.sha256_file(manifest),
        "count": len(rows),
        "subjects": len({row["subject_id"] for row in rows}),
        "case_ids": [item["case_id"] for item in metadata],
        "model_shape": list(shape),
        "image_dtype": "float32",
        "label_dtype": "uint8",
        "split": "authorized_internal_test270",
        "test_access": True,
        "outer_selector_access": False,
        "confirmation_access": False,
        "preprocessing": {
            "order": [
                "canonical_RAS",
                "N4_shrink_2",
                "image_nonzero_bounding_box_crop",
                "source_nonzero_population_zscore",
                "resize_to_160_cubic_grid"
            ],
            "spatial": "braincrop_resize",
            "output_resolution": args.resolution,
            "image_interpolation": "trilinear_align_corners_false",
            "label_interpolation": "nearest-exact",
            "uses_gt_for_localization": False,
            "source_is_existing_128_cache": False
        },
        "per_case_source_positive_voxels": source_positive,
        "source_positive_voxels_sha256": base.stable_sha256(source_positive),
        "per_case_resized_positive_voxels": resized_positive,
        "crop_geometry": {
            item["case_id"]: {
                "origin": item["origin"],
                "extent": item["extent"],
                "native_shape": item["native_shape"],
                "source_part": item["source_part"]
            }
            for item in metadata
        },
        "raw_reproduction": {
            "simpleitk_global_threads": 1,
            "cases_verified": len(verification),
            "all_exact": True,
            "results": verification
        },
        "images_sha256": base.sha256_file(images_path),
        "labels_sha256": base.sha256_file(labels_path),
        "builder": "scripts/build_t1ce_wt_braincrop_test_cache_variable_resolution.py",
        "builder_sha256": base.sha256_file(Path(__file__).resolve()),
        "build_seconds": round(time.time() - started, 3)
    }
    (output / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-cases", type=int, default=270)
    parser.add_argument("--expected-subjects", type=int, default=119)
    parser.add_argument("--resolution", type=int, choices=(160,), default=160)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--verify-cases", type=int, default=4)
    parser.add_argument("--authorization", required=True)
    index = build(parser.parse_args())
    print(json.dumps({
        "count": index["count"],
        "subjects": index["subjects"],
        "resolution": index["preprocessing"]["output_resolution"],
        "images_sha256": index["images_sha256"],
        "labels_sha256": index["labels_sha256"]
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
