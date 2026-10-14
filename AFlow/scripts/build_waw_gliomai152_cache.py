#!/usr/bin/env python3
"""Build the frozen canonical-N4 128^3 WAW-GliomAI152 cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from sammed3d import t1ce_binary
from scripts.build_no_n4_clean_cache import arrays
from scripts import waw_gliomai152_protocol as p


CFG: dict = {}
ORIGINAL_CASE_ID = t1ce_binary._case_id


def init_worker(config: dict) -> None:
    CFG.update(config)
    CFG["allowed"] = set(config["allowed"])
    torch.set_num_threads(1)
    import SimpleITK as sitk

    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    t1ce_binary._case_id = lambda value: value if value in CFG["allowed"] else ORIGINAL_CASE_ID(value)
    CFG["image_fd"] = os.open(config["images"], os.O_WRONLY)
    CFG["label_fd"] = os.open(config["labels"], os.O_WRONLY)


def array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def source_map() -> dict[str, dict]:
    receipt = json.loads(p.SOURCE_RECEIPT.read_text())
    if receipt["status"] != "passed" or receipt["selected"]["cases"] != p.CASES:
        raise ValueError("source receipt contract")
    return {row["case_id"]: row for row in receipt["selected"]["files"]}


def prepare(row: dict, sources: dict[str, dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    source = sources[row["case_id"]]
    image_path = Path(source["image_path"])
    mask_path = Path(source["mask_path"])
    p.checked(image_path, source["image_sha256"])
    p.checked(mask_path, source["mask_sha256"])
    image_nifti = nib.as_closest_canonical(nib.load(str(image_path), mmap=True))
    mask_nifti = nib.as_closest_canonical(nib.load(str(mask_path), mmap=True))
    affine_error = float(np.max(np.abs(image_nifti.affine - mask_nifti.affine)))
    if image_nifti.shape != mask_nifti.shape or affine_error > 1e-5:
        raise ValueError(f"geometry: {row['case_id']}")
    image = np.ascontiguousarray(np.asanyarray(image_nifti.dataobj))
    raw = np.ascontiguousarray(np.asanyarray(mask_nifti.dataobj))
    rounded = np.rint(raw)
    values = sorted(int(value) for value in np.unique(rounded))
    if (
        not np.isfinite(image).all()
        or not (image > 0).any()
        or not np.isfinite(raw).all()
        or not np.allclose(raw, rounded, rtol=0, atol=1e-5)
        or not set(values).issubset({0, 1, 2, 3})
    ):
        raise ValueError(f"array/domain: {row['case_id']} {values}")
    binary = np.ascontiguousarray(rounded > 0, dtype=np.uint8)
    if not binary.any() or not (binary == 0).any():
        raise ValueError(f"invalid WT: {row['case_id']}")
    output_image, output_label, geometry = arrays(
        row["case_id"],
        image,
        binary,
        nib.affines.voxel_sizes(image_nifti.affine),
        use_n4=True,
    )
    details = {
        "native_shape": list(image_nifti.shape),
        "spacing_mm": [float(value) for value in nib.affines.voxel_sizes(image_nifti.affine)],
        "axis_codes": list(nib.aff2axcodes(image_nifti.affine)),
        "affine_error": affine_error,
        "raw_values": values,
        "raw_wt": int(binary.sum()),
        "resized_wt": int(output_label.sum()),
        "geometry": geometry,
        "image_sha256": array_sha(output_image),
        "label_sha256": array_sha(output_label),
    }
    return output_image, output_label, details


def worker(task: tuple[int, dict, dict]) -> tuple[int, str, str, dict]:
    slot, row, sources = task
    image, label, details = prepare(row, sources)
    voxels = 128**3
    if os.pwrite(CFG["image_fd"], image.tobytes(), slot * voxels * 4) != voxels * 4:
        raise IOError("short image cache write")
    if os.pwrite(CFG["label_fd"], label.tobytes(), slot * voxels) != voxels:
        raise IOError("short label cache write")
    return slot, row["case_id"], row["subject_id"], details


def probe() -> None:
    if p.FIRST_PROBE.exists():
        raise FileExistsError(p.FIRST_PROBE)
    rows = p.cases()
    sources = source_map()
    torch.set_num_threads(1)
    import SimpleITK as sitk

    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    first_id = rows[0]["case_id"]
    t1ce_binary._case_id = lambda value: value if value == first_id else ORIGINAL_CASE_ID(value)
    image, label, details = prepare(rows[0], sources)
    receipt = {
        "schema": "aflow.waw_gliomai152.first_case_probe",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "case_id": first_id,
        **details,
        "output_image_shape": list(image.shape),
        "output_label_values": sorted(int(value) for value in np.unique(label)),
        "model_inference": False,
        "other_cases_preprocessed": False,
        "runner_sha256": p.sha256(Path(__file__)),
    }
    p.write_new(p.FIRST_PROBE, receipt)
    print(json.dumps(receipt, indent=2))


def build(workers: int) -> None:
    rows = p.cases()
    sources = source_map()
    first = json.loads(p.FIRST_PROBE.read_text())
    if first.get("status") != "passed" or first["case_id"] != rows[0]["case_id"]:
        raise ValueError("first-case probe contract")
    if p.CACHE.exists() or p.CACHE_RECEIPT.exists():
        raise FileExistsError("cache output already exists")
    required = p.CASES * 128**3 * 5
    if shutil.disk_usage(p.CACHE.parent).free < required + 2 * 1024**3:
        raise RuntimeError("insufficient disk with 2 GiB reserve")
    p.CACHE.mkdir()
    images_path = p.CACHE / "images.f32"
    labels_path = p.CACHE / "labels.u8"
    for path, size in ((images_path, required * 4 // 5), (labels_path, required // 5)):
        with path.open("xb") as handle:
            handle.truncate(size)
    config = {
        "allowed": [row["case_id"] for row in rows],
        "images": str(images_path),
        "labels": str(labels_path),
    }
    begin = time.monotonic()
    results = []
    tasks = [(slot, row, sources) for slot, row in enumerate(rows)]
    with mp.get_context("spawn").Pool(workers, initializer=init_worker, initargs=(config,)) as pool:
        for result in pool.imap_unordered(worker, tasks, chunksize=1):
            results.append(result)
            if len(results) % 10 == 0 or len(results) == p.CASES:
                print(f"cached WAW {len(results)}/{p.CASES}", flush=True)
    results.sort()
    details = {case_id: detail for _, case_id, _, detail in results}
    images = np.memmap(images_path, dtype=np.float32, mode="r", shape=(p.CASES, 128, 128, 128))
    labels = np.memmap(labels_path, dtype=np.uint8, mode="r", shape=(p.CASES, 128, 128, 128))
    first_details = details[rows[0]["case_id"]]
    gates = {
        "all_finite_binary_nonempty": all(
            np.isfinite(images[index]).all()
            and np.all((labels[index] == 0) | (labels[index] == 1))
            and labels[index].any()
            for index in range(p.CASES)
        ),
        "first_probe_image_exact": first_details["image_sha256"] == first["image_sha256"],
        "first_probe_label_exact": first_details["label_sha256"] == first["label_sha256"],
        "all_wt_nonempty_before_and_after": all(
            detail["raw_wt"] > 0 and detail["resized_wt"] > 0 for detail in details.values()
        ),
        "global_raw_union_exact": sorted(
            {value for detail in details.values() for value in detail["raw_values"]}
        )
        == [0, 1, 2, 3],
    }
    if not all(gates.values()):
        raise RuntimeError(f"cache gates failed: {gates}")
    index = {
        "schema": "aflow.waw_gliomai152.canonical_n4_cache",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "public_external_to_project_waw_preop",
        "cases": p.CASES,
        "subjects": p.CASES,
        "case_ids": [row["case_id"] for row in rows],
        "subject_ids": [row["subject_id"] for row in rows],
        "model_shape": [128, 128, 128],
        "image_dtype": "float32",
        "label_dtype": "uint8",
        "images_sha256": p.sha256(images_path),
        "labels_sha256": p.sha256(labels_path),
        "details": details,
        "raw_union": [0, 1, 2, 3],
        "target": "preoperative WT=all positive labels {1,2,3}",
        "preprocessing": {
            "canonical_ras": True,
            "n4_shrink": 2,
            "n4_iterations": [50, 50, 50, 50],
            "simpleitk_threads": 1,
            "image_only_crop": True,
            "source_nonzero_zscore": True,
            "resolution": 128,
        },
        "gates": gates,
        "source_receipt_sha256": p.SOURCE_RECEIPT_SHA,
        "first_probe_sha256": p.sha256(p.FIRST_PROBE),
        "builder_sha256": p.sha256(Path(__file__)),
        "elapsed_seconds": time.monotonic() - begin,
        "model_inference": False,
    }
    p.write_new(p.CACHE / "index.json", index)
    receipt = {
        "schema": "aflow.waw_gliomai152.cache_receipt",
        "schema_version": 1,
        "status": "passed",
        "all_gates_passed": True,
        "cases": p.CASES,
        "subjects": p.CASES,
        "index_sha256": p.sha256(p.CACHE / "index.json"),
        "images_sha256": index["images_sha256"],
        "labels_sha256": index["labels_sha256"],
        "raw_union": index["raw_union"],
        "gates": gates,
        "elapsed_seconds": index["elapsed_seconds"],
        "model_inference": False,
    }
    p.write_new(p.CACHE_RECEIPT, receipt)
    print(json.dumps(receipt, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("probe", "build"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    p.docs()
    probe() if args.action == "probe" else build(args.workers)


if __name__ == "__main__":
    main()
