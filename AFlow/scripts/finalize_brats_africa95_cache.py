#!/usr/bin/env python3
"""Finalize the completed Africa95 memmap only after a single-thread exact probe."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np

from scripts import brats_africa95_protocol as p


FAILED_IMAGES_SHA = "f95959ac50335185d3ec89a777bbef8ac85d54328ebbd53bc434a3a69967ff82"
FAILED_LABELS_SHA = "9e937ca9481c9f0661e666f478ba0f860b87d1bf689a54d09d3f70cc3e90b05b"


def main() -> None:
    p.documents(); rows = p.cohort_rows()
    if (p.CACHE / "index.json").exists() or p.CACHE_RECEIPT.exists(): raise FileExistsError("cache already finalized")
    p.checked(p.CACHE / "images.f32", FAILED_IMAGES_SHA); p.checked(p.CACHE / "labels.u8", FAILED_LABELS_SHA)
    probe = json.loads(p.FIRST_PROBE.read_text())
    if probe.get("status") != "passed" or probe.get("case_id") != rows[0]["case_id"] or probe.get("model_inference") is not False: raise ValueError("V2 probe contract")
    source_download = json.loads(p.SOURCE_DOWNLOAD.read_text()); source = {x["relative_path"]: x for x in source_download["comparisons"]}
    images = np.memmap(p.CACHE / "images.f32", dtype=np.float32, mode="r", shape=(p.CASES, 128, 128, 128)); labels = np.memmap(p.CACHE / "labels.u8", dtype=np.uint8, mode="r", shape=(p.CASES, 128, 128, 128))
    details = {}; started = time.monotonic()
    for slot, row in enumerate(rows):
        case = row["case_id"]; root = p.SOURCE / "copy_a" / case; image_path = root / f"{case}-t1c.nii.gz"; label_path = root / f"{case}-seg.nii.gz"
        if p.sha(image_path) != source[f"{case}/{case}-t1c.nii.gz"]["sha256"] or p.sha(label_path) != source[f"{case}/{case}-seg.nii.gz"]["sha256"]: raise ValueError(case + " source SHA")
        image_nii = nib.as_closest_canonical(nib.load(str(image_path), mmap=True)); label_nii = nib.as_closest_canonical(nib.load(str(label_path), mmap=True)); affine = float(np.max(np.abs(image_nii.affine - label_nii.affine)))
        if image_nii.shape != label_nii.shape or affine > 1e-5: raise ValueError(case + " geometry")
        image = np.asanyarray(image_nii.dataobj); raw = np.asanyarray(label_nii.dataobj); rounded = np.rint(raw); values = sorted(int(x) for x in np.unique(rounded))
        if not np.isfinite(image).all() or not (image > 0).any() or not np.isfinite(raw).all() or not np.allclose(raw, rounded, rtol=0, atol=1e-5) or not set(values) <= {0,1,2,3,4} or not (rounded > 0).any(): raise ValueError(case + " array/domain")
        if not np.isfinite(images[slot]).all() or not np.all((labels[slot] == 0) | (labels[slot] == 1)) or not labels[slot].any(): raise ValueError(case + " memmap")
        details[case] = {"native_shape": list(image_nii.shape), "spacing": [float(x) for x in nib.affines.voxel_sizes(image_nii.affine)], "axis_codes": list(nib.aff2axcodes(image_nii.affine)), "affine_max_abs": affine, "raw_label_values": values, "raw_wt_voxels": int((rounded > 0).sum()), "resized_wt_voxels": int(labels[slot].sum())}
    import hashlib
    first_image_sha = hashlib.sha256(np.ascontiguousarray(images[0]).tobytes()).hexdigest(); first_label_sha = hashlib.sha256(np.ascontiguousarray(labels[0]).tobytes()).hexdigest()
    gates = {"all95_source_and_memmap_revalidated": True, "v2_probe_image_byte_exact": first_image_sha == probe["output_image_sha256"], "v2_probe_label_byte_exact": first_label_sha == probe["output_label_sha256"], "all95_wt_nonempty": all(x["raw_wt_voxels"] > 0 and x["resized_wt_voxels"] > 0 for x in details.values())}
    if not all(gates.values()): raise RuntimeError("finalizer gates failed")
    index = {"schema": "aflow.brats_africa95.cache", "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "split": "public_postchallenge_external95", "cases": p.CASES, "subjects": p.CASES, "case_ids": [x["case_id"] for x in rows], "subject_ids": [x["subject_id"] for x in rows], "model_shape": [128,128,128], "image_dtype": "float32", "label_dtype": "uint8", "images_sha256": FAILED_IMAGES_SHA, "labels_sha256": FAILED_LABELS_SHA, "preprocessing": {"source": "official TCIA BraTS-Africa T1CE", "canonical_ras": True, "n4_bias_field_correction": True, "n4_shrink_factor": 2, "simpleitk_threads": 1, "spatial": "positive-support crop then resize128", "normalization": "source positive-support population z-score", "uses_gt_for_localization": False}, "target": "preoperative WT=all raw label values >0", "allowed_raw_values": [0,1,2,3,4], "details": details, "gates": gates, "first_probe_v2_sha256": p.sha(p.FIRST_PROBE), "source_download_sha256": p.sha(p.SOURCE_DOWNLOAD), "cohort_sha256": p.COHORT_SHA, "original_builder_sha256": p.sha(Path(__file__).with_name("build_brats_africa95_cache.py")), "finalizer_sha256": p.sha(Path(__file__)), "elapsed_finalizer_seconds": time.monotonic() - started, "access": {"source_arrays": True, "label_values": True, "model_inference": False}}
    p.write_new(p.CACHE / "index.json", index)
    receipt = {"schema": "aflow.brats_africa95.cache_receipt", "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "passed_after_thread_matched_probe", "all_gates_passed": True, "gates": gates, "cache": str(p.CACHE), "index_sha256": p.sha(p.CACHE / "index.json"), "images_sha256": FAILED_IMAGES_SHA, "labels_sha256": FAILED_LABELS_SHA, "raw_label_value_union": sorted({v for x in details.values() for v in x["raw_label_values"]}), "original_failed_execution_preserved": "CACHE_EXECUTION_01_FAILURE.json", "model_inference": False}
    p.write_new(p.CACHE_RECEIPT, receipt); print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__": main()
