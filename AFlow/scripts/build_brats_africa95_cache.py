#!/usr/bin/env python3
"""Build the canonical-N4 128^3 cache for frozen BraTS-Africa95."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import tempfile
import time

import nibabel as nib
import numpy as np
import torch

from sammed3d import t1ce_binary
from sammed3d.t1ce_wt_cached import MODEL_SHAPE
from scripts.build_no_n4_clean_cache import arrays as prepare_arrays
from scripts import brats_africa95_protocol as p


CFG = {}
ORIGINAL_CASE_ID = t1ce_binary._case_id


def init(config: dict) -> None:
    CFG.update(config); CFG["allowed"] = set(config["allowed"])
    torch.set_num_threads(1)
    import SimpleITK as sitk
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    def external(value):
        return value if isinstance(value, str) and value in CFG["allowed"] else ORIGINAL_CASE_ID(value)
    t1ce_binary._case_id = external
    CFG["image_fd"] = os.open(config["images"], os.O_WRONLY)
    CFG["label_fd"] = os.open(config["labels"], os.O_WRONLY)


def array_sha(value: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def prepare(row: dict, source: dict[str, dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    case = row["case_id"]; root = p.SOURCE / "copy_a" / case
    image_item = source[f"{case}/{case}-t1c.nii.gz"]
    label_item = source[f"{case}/{case}-seg.nii.gz"]
    image_path = root / f"{case}-t1c.nii.gz"; label_path = root / f"{case}-seg.nii.gz"
    if p.sha(image_path) != image_item["sha256"] or p.sha(label_path) != label_item["sha256"]:
        raise ValueError(case + " source SHA changed")
    image_nii = nib.as_closest_canonical(nib.load(str(image_path), mmap=True))
    label_nii = nib.as_closest_canonical(nib.load(str(label_path), mmap=True))
    affine_error = float(np.max(np.abs(image_nii.affine - label_nii.affine)))
    if image_nii.shape != label_nii.shape or affine_error > 1e-5:
        raise ValueError(case + " image/label geometry mismatch")
    image = np.ascontiguousarray(np.asanyarray(image_nii.dataobj))
    raw = np.ascontiguousarray(np.asanyarray(label_nii.dataobj))
    if image.ndim != 3 or not np.isfinite(image).all() or not (image > 0).any():
        raise ValueError(case + " invalid image")
    rounded = np.rint(raw)
    values = sorted(int(x) for x in np.unique(rounded))
    if not np.isfinite(raw).all() or not np.allclose(raw, rounded, rtol=0, atol=1e-5) or not set(values) <= {0, 1, 2, 3, 4}:
        raise ValueError(case + f" invalid label domain {values}")
    binary = np.ascontiguousarray(rounded > 0, dtype=np.uint8)
    if not (binary == 0).any() or not binary.any():
        raise ValueError(case + " WT must contain background and foreground")
    out_image, out_label, geometry = prepare_arrays(case, image, binary, nib.affines.voxel_sizes(image_nii.affine), use_n4=True)
    if not np.isfinite(out_image).all() or not out_label.any():
        raise ValueError(case + " invalid cache tensor")
    return out_image, out_label, {"native_shape": list(image_nii.shape), "spacing": [float(x) for x in nib.affines.voxel_sizes(image_nii.affine)], "axis_codes": list(nib.aff2axcodes(image_nii.affine)), "affine_max_abs": affine_error, "raw_label_values": values, "raw_wt_voxels": int(binary.sum()), "resized_wt_voxels": int(out_label.sum()), "geometry": geometry, "image_sha256": array_sha(out_image), "label_sha256": array_sha(out_label)}


def worker(task):
    slot, row, source = task
    image, label, detail = prepare(row, source)
    voxels = int(np.prod(MODEL_SHAPE))
    if os.pwrite(CFG["image_fd"], image.tobytes(), slot * voxels * 4) != voxels * 4 or os.pwrite(CFG["label_fd"], label.tobytes(), slot * voxels) != voxels:
        raise IOError("short cache write")
    return slot, row["case_id"], detail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--workers", type=int, default=8); args = parser.parse_args()
    p.documents(); rows = p.cohort_rows()
    if not p.SOURCE_DOWNLOAD.exists(): raise FileNotFoundError("source receipt missing")
    download = json.loads(p.SOURCE_DOWNLOAD.read_text())
    if download.get("status") != "passed" or download.get("files_copy_exact") != p.FILES or download.get("voxel_arrays_accessed") is not False:
        raise ValueError("source receipt contract mismatch")
    if not p.FIRST_PROBE.exists(): raise FileNotFoundError("frozen first-case probe receipt missing")
    first_probe = json.loads(p.FIRST_PROBE.read_text())
    if first_probe.get("status") != "passed" or first_probe.get("case_id") != rows[0]["case_id"] or first_probe.get("model_inference") is not False:
        raise ValueError("first-case probe contract mismatch")
    if p.CACHE.exists() or p.CACHE_RECEIPT.exists(): raise FileExistsError("cache already started")
    required = p.CASES * int(np.prod(MODEL_SHAPE)) * 5
    if shutil.disk_usage(p.CACHE.parent).free < required + 5 * 1024**3: raise RuntimeError("cache disk reserve")
    source = {x["relative_path"]: x for x in download["comparisons"]}
    p.CACHE.mkdir()
    images_path = p.CACHE / "images.f32"; labels_path = p.CACHE / "labels.u8"
    for path, size in ((images_path, required * 4 // 5), (labels_path, required // 5)):
        with path.open("xb") as handle: handle.truncate(size)
    started = time.monotonic(); results = []
    config = {"allowed": [x["case_id"] for x in rows], "images": str(images_path), "labels": str(labels_path)}
    with mp.get_context("spawn").Pool(args.workers, initializer=init, initargs=(config,)) as pool:
        for item in pool.imap_unordered(worker, [(slot, row, source) for slot, row in enumerate(rows)], chunksize=1):
            results.append(item)
            if len(results) % 10 == 0 or len(results) == p.CASES: print(f"built Africa95 {len(results)}/{p.CASES}", flush=True)
    results.sort(); details = {case: detail for _, case, detail in results}
    images = np.memmap(images_path, dtype=np.float32, mode="r", shape=(p.CASES, *MODEL_SHAPE))
    labels = np.memmap(labels_path, dtype=np.uint8, mode="r", shape=(p.CASES, *MODEL_SHAPE))
    valid = all(np.isfinite(images[i]).all() and np.all((labels[i] == 0) | (labels[i] == 1)) and labels[i].any() for i in range(p.CASES))
    # Rebuild the frozen first case independently in the parent process.
    torch.set_num_threads(1)
    import SimpleITK as sitk
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    t1ce_binary._case_id = lambda value: value if value == rows[0]["case_id"] else ORIGINAL_CASE_ID(value)
    rebuilt_image, rebuilt_label, _ = prepare(rows[0], source)
    reproduction = {"case_id": rows[0]["case_id"], "image_byte_exact": bool(np.array_equal(rebuilt_image, images[0])), "label_byte_exact": bool(np.array_equal(rebuilt_label, labels[0])), "image_max_abs_error": float(np.max(np.abs(rebuilt_image - images[0])))}
    gates = {"all_memmaps_valid": valid, "first_image_exact": reproduction["image_byte_exact"], "first_label_exact": reproduction["label_byte_exact"], "first_probe_image_exact": details[rows[0]["case_id"]]["image_sha256"] == first_probe["output_image_sha256"], "first_probe_label_exact": details[rows[0]["case_id"]]["label_sha256"] == first_probe["output_label_sha256"], "image_bytes_exact": images_path.stat().st_size == required * 4 // 5, "label_bytes_exact": labels_path.stat().st_size == required // 5, "all_source_wt_nonempty": all(x["resized_wt_voxels"] > 0 and x["raw_wt_voxels"] > 0 for x in details.values())}
    if not all(gates.values()): raise RuntimeError("cache gates failed before index")
    index = {"schema": "aflow.brats_africa95.cache", "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "split": "public_postchallenge_external95", "cases": p.CASES, "subjects": p.CASES, "case_ids": [x["case_id"] for x in rows], "subject_ids": [x["subject_id"] for x in rows], "model_shape": list(MODEL_SHAPE), "image_dtype": "float32", "label_dtype": "uint8", "images_sha256": p.sha(images_path), "labels_sha256": p.sha(labels_path), "preprocessing": {"source": "official TCIA BraTS-Africa T1CE", "canonical_ras": True, "n4_bias_field_correction": True, "n4_shrink_factor": 2, "spatial": "positive-support crop then resize128", "normalization": "source positive-support population z-score", "uses_gt_for_localization": False}, "target": "preoperative WT=all raw label values >0", "allowed_raw_values": [0, 1, 2, 3, 4], "details": details, "gates": gates, "reproduction": reproduction, "first_probe_sha256": p.sha(p.FIRST_PROBE), "source_download_sha256": p.sha(p.SOURCE_DOWNLOAD), "cohort_sha256": p.COHORT_SHA, "builder_sha256": p.sha(Path(__file__)), "elapsed_seconds": time.monotonic() - started, "access": {"source_arrays": True, "label_values": True, "model_inference": False}}
    p.write_new(p.CACHE / "index.json", index)
    receipt = {"schema": "aflow.brats_africa95.cache_receipt", "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "passed", "all_gates_passed": True, "gates": gates, "cache": str(p.CACHE), "index_sha256": p.sha(p.CACHE / "index.json"), "images_sha256": index["images_sha256"], "labels_sha256": index["labels_sha256"], "reproduction": reproduction, "raw_label_value_union": sorted({v for x in details.values() for v in x["raw_label_values"]}), "elapsed_seconds": index["elapsed_seconds"], "label_values_accessed": True, "model_inference": False}
    p.write_new(p.CACHE_RECEIPT, receipt); print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__": main()
