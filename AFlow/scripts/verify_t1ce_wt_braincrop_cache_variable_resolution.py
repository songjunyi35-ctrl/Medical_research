#!/usr/bin/env python3
"""Reproduce selected variable-resolution cache cases with worker-identical N4."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

import scripts.build_t1ce_wt_braincrop_cache_variable_resolution as builder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=4)
    args = parser.parse_args()
    for path in (args.cache, args.manifest):
        lowered = str(path.resolve()).lower()
        if path == args.cache and any(
            token in lowered for token in ("selector", "confirmation", "sealed", "test270")
        ):
            raise ValueError(f"forbidden data role in cache path: {path}")
    if builder.sha256_file(args.manifest) != args.expected_manifest_sha256:
        raise ValueError("authorized train manifest SHA-256 mismatch")
    index_path = args.cache / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if any(
        index.get(key) is not False
        for key in ("outer_selector_access", "confirmation_access", "test_access")
    ):
        raise ValueError("cache declares forbidden data access")
    shape = tuple(int(value) for value in index["model_shape"])
    images = np.memmap(
        args.cache / "images.f32",
        dtype=np.float32,
        mode="r",
        shape=(index["count"], *shape),
    )
    labels = np.memmap(
        args.cache / "labels.u8",
        dtype=np.uint8,
        mode="r",
        shape=(index["count"], *shape),
    )
    source_rows = builder.read_train_cohort(args.manifest)
    rows_by_case = {row["case_id"]: row for row in source_rows}
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    builder._CFG.update({"n4": True, "n4_shrink": 2, "model_shape": shape})
    count = min(int(args.cases), int(index["count"]))
    picks = np.linspace(0, index["count"] - 1, num=count, dtype=int).tolist()
    results = []
    for slot in picks:
        case_id = index["case_ids"][slot]
        image_path, label_path, source_part = builder.resolve_case_paths(
            args.raw_root.resolve(), case_id
        )
        image, label, metadata = builder._prepare(
            {
                **rows_by_case[case_id],
                "image_path": str(image_path),
                "label_path": str(label_path),
                "source_part": source_part,
            }
        )
        cached_image = np.asarray(images[slot])
        cached_label = np.asarray(labels[slot])
        max_abs = float(np.max(np.abs(image - cached_image)))
        image_exact = bool(np.array_equal(image, cached_image))
        label_exact = bool(np.array_equal(label, cached_label))
        source_volume_exact = (
            metadata["source_positive_voxels"]
            == index["per_case_source_positive_voxels"][case_id]
        )
        results.append(
            {
                "slot": slot,
                "case_id": case_id,
                "image_exact": image_exact,
                "label_exact": label_exact,
                "image_max_abs_difference": max_abs,
                "source_positive_voxels_exact": source_volume_exact,
            }
        )
        if not image_exact or not label_exact or not source_volume_exact:
            raise RuntimeError(f"raw reproduction failed: {results[-1]}")
        print(f"exact raw reproduction: {case_id}", flush=True)
    report = {
        "schema": "aflow.sammed3d.t1ce_wt_variable_resolution_raw_reproduction",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "cache_index_sha256": builder.sha256_file(index_path),
        "builder_sha256": builder.sha256_file(Path(builder.__file__).resolve()),
        "verifier_sha256": builder.sha256_file(Path(__file__).resolve()),
        "simpleitk_global_threads": 1,
        "cases_verified": len(results),
        "results": results,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    output = args.cache / "RAW_REPRODUCTION.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
