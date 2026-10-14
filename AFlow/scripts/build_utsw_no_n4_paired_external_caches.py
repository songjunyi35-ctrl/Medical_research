"""Build paired canonical-N4/no-N4 caches for frozen UTSW362."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch

from sammed3d import t1ce_binary
from scripts.audit_utsw_primary_fets_no_n4_input_bridge import validate_source_arrays
from scripts.build_no_n4_clean_cache import arrays as shared_arrays
from scripts import utsw_no_n4_paired_external_protocol as p


CFG: dict[str, object] = {}
ORIGINAL_CASE_ID = t1ce_binary._case_id


def sha_array(array: np.ndarray) -> str:
    return __import__("hashlib").sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def initialize(config: dict) -> None:
    CFG.update(config)
    CFG["allowed"] = set(config["allowed"])
    torch.set_num_threads(1)
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)

    def external_case_id(value: object) -> str:
        if isinstance(value, str) and value in CFG["allowed"]:
            return value
        return ORIGINAL_CASE_ID(value)

    t1ce_binary._case_id = external_case_id
    for arm in ("canonical", "no_n4"):
        CFG[f"{arm}_image_fd"] = os.open(config[f"{arm}_images"], os.O_WRONLY)
        CFG[f"{arm}_label_fd"] = os.open(config[f"{arm}_labels"], os.O_WRONLY)


def pair_gates(canonical_image: np.ndarray, canonical_label: np.ndarray, canonical_geometry: dict,
               no_n4_image: np.ndarray, no_n4_label: np.ndarray, no_n4_geometry: dict) -> dict:
    return {
        "labels_exact": bool(np.array_equal(canonical_label, no_n4_label)),
        "geometry_exact": canonical_geometry == no_n4_geometry,
        "images_different": not bool(np.array_equal(canonical_image, no_n4_image)),
    }


def worker(task: tuple[int, dict, str, str]) -> tuple[int, str, dict]:
    slot, row, expected_image_sha, expected_label_sha = task
    subject = row["subject_id"]
    image_path = p.SOURCE / "copy_a" / subject / "brain_t1ce.nii.gz"
    label_path = p.SOURCE / "copy_a" / subject / "rtumorseg_manual_correction.nii.gz"
    if p.sha(image_path) != expected_image_sha or p.sha(label_path) != expected_label_sha:
        raise ValueError(f"{subject} source file SHA changed")
    image_nii = nib.as_closest_canonical(nib.load(str(image_path), mmap=True))
    label_nii = nib.as_closest_canonical(nib.load(str(label_path), mmap=True))
    affine_delta = float(np.max(np.abs(np.asarray(image_nii.affine) - np.asarray(label_nii.affine))))
    if image_nii.shape != label_nii.shape or affine_delta > 1e-6:
        raise ValueError(f"{subject} source geometry mismatch")
    source_image = np.ascontiguousarray(np.asanyarray(image_nii.dataobj))
    source_label = np.ascontiguousarray(np.asanyarray(label_nii.dataobj))
    binary, source = validate_source_arrays(source_image, source_label)
    if not set(source["label_values"]) <= {0.0, 1.0, 2.0, 4.0}:
        raise ValueError(f"{subject} unexpected UTSW label domain: {source['label_values']}")
    spacing = nib.affines.voxel_sizes(image_nii.affine)
    no_image, no_label, no_geometry = shared_arrays(subject, source_image, binary, spacing, use_n4=False)
    canonical_image, canonical_label, canonical_geometry = shared_arrays(subject, source_image, binary, spacing, use_n4=True)
    gates = pair_gates(canonical_image, canonical_label, canonical_geometry, no_image, no_label, no_geometry)
    if not gates["labels_exact"] or not gates["geometry_exact"]:
        raise RuntimeError(f"{subject} paired cache gate failed: {gates}")
    voxels = 128 ** 3
    outputs = {
        "canonical": (canonical_image, canonical_label),
        "no_n4": (no_image, no_label),
    }
    for arm, (image, label) in outputs.items():
        if os.pwrite(int(CFG[f"{arm}_image_fd"]), image.tobytes(), slot * voxels * 4) != voxels * 4:
            raise IOError(f"short {arm} image write")
        if os.pwrite(int(CFG[f"{arm}_label_fd"]), label.tobytes(), slot * voxels) != voxels:
            raise IOError(f"short {arm} label write")
    return slot, subject, {
        "native_shape": list(image_nii.shape),
        "native_affine_max_abs": affine_delta,
        "axis_codes": list(nib.aff2axcodes(image_nii.affine)),
        "spacing": [float(value) for value in spacing],
        "raw_label_values": source["label_values"],
        "raw_wt_voxels": source["source_wt_voxels"],
        "geometry": no_geometry,
        "canonical_image_sha256": sha_array(canonical_image),
        "no_n4_image_sha256": sha_array(no_image),
        "label_sha256": sha_array(no_label),
        "pair_gates": gates,
    }


def validate_memmaps(root: Path) -> dict:
    images = np.memmap(root / "images.f32", mode="r", dtype=np.float32, shape=(p.CASES, *p.MODEL_SHAPE))
    labels = np.memmap(root / "labels.u8", mode="r", dtype=np.uint8, shape=(p.CASES, *p.MODEL_SHAPE))
    finite = True
    binary = True
    positive = True
    for slot in range(p.CASES):
        finite = finite and bool(np.isfinite(images[slot]).all())
        binary = binary and bool(np.all((labels[slot] == 0) | (labels[slot] == 1)))
        positive = positive and bool(labels[slot].any())
    return {"all_images_finite": finite, "all_labels_binary": binary, "all_targets_positive": positive}


def main(workers: int) -> None:
    from scripts.freeze_utsw_no_n4_paired_external import check_implementation

    check_implementation()
    p.verify_preaccess()
    download_path = p.RECORD / "SOURCE_DOWNLOAD.json"
    download = json.loads(download_path.read_text())
    if download.get("status") != "passed" or download.get("files_copy_exact") != p.SOURCE_FILES:
        raise ValueError("UTSW dual source download did not pass")
    if download.get("preaccess_sha256") != p.sha(p.PREACCESS) or download.get("voxel_arrays_accessed") is not False:
        raise ValueError("UTSW source receipt access binding mismatch")
    if p.CACHE_ROOT.exists():
        raise FileExistsError(p.CACHE_ROOT)
    per_cache = p.CASES * 128 ** 3 * 5
    if shutil.disk_usage(p.CACHE_ROOT.parent).free < 2 * per_cache + 5 * 1024**3:
        raise RuntimeError("insufficient /tmp reserve for paired UTSW caches")
    rows = p.cohort()
    source_rows = {item["relative_path"]: item for item in download["comparisons"]}
    tasks = []
    for slot, row in enumerate(rows):
        image_key = f"{row['subject_id']}/brain_t1ce.nii.gz"
        label_key = f"{row['subject_id']}/rtumorseg_manual_correction.nii.gz"
        tasks.append((slot, row, source_rows[image_key]["sha256"], source_rows[label_key]["sha256"]))
    p.CACHE_ROOT.mkdir()
    for arm in ("canonical", "no_n4"):
        root = p.CACHE_ROOT / arm
        root.mkdir()
        for name, size in (("images.f32", p.CASES * 128 ** 3 * 4), ("labels.u8", p.CASES * 128 ** 3)):
            with (root / name).open("xb") as handle:
                handle.truncate(size)
    config = {"allowed": [row["subject_id"] for row in rows]}
    for arm in ("canonical", "no_n4"):
        config[f"{arm}_images"] = str(p.CACHE_ROOT / arm / "images.f32")
        config[f"{arm}_labels"] = str(p.CACHE_ROOT / arm / "labels.u8")
    start = time.monotonic()
    results = []
    with mp.get_context("spawn").Pool(workers, initializer=initialize, initargs=(config,)) as pool:
        for item in pool.imap_unordered(worker, tasks, chunksize=1):
            results.append(item)
            if len(results) % 25 == 0:
                print(f"built paired UTSW {len(results)}/{p.CASES}", flush=True)
    results.sort()
    details = {subject: value for _, subject, value in results}
    case_ids = [row["subject_id"] for row in rows]
    no_n4_bridge = json.loads(p.NO_N4_BRIDGE.read_text())
    canonical_bridge_copy_a = json.loads(p.CANONICAL_BRIDGE_COPY_A.read_text())
    fixed_checks = {}
    for subject in ("BT0001", "BT0002", "BT0003"):
        value = details[subject]
        fixed_checks[subject] = {
            "no_n4_image_exact": value["no_n4_image_sha256"] == no_n4_bridge["copies"]["copy_a"][subject]["output_image_sha256"],
            "no_n4_label_exact": value["label_sha256"] == no_n4_bridge["copies"]["copy_a"][subject]["output_label_sha256"],
            "no_n4_geometry_exact": value["geometry"] == no_n4_bridge["comparisons"][subject]["geometry"],
            "canonical_image_exact": value["canonical_image_sha256"] == canonical_bridge_copy_a["cases"][subject]["output_image_sha256"],
            "canonical_label_exact": value["label_sha256"] == canonical_bridge_copy_a["cases"][subject]["output_label_sha256"],
            "canonical_geometry_exact": value["geometry"] == canonical_bridge_copy_a["cases"][subject]["geometry"],
        }
    indices = {}
    for arm in ("canonical", "no_n4"):
        root = p.CACHE_ROOT / arm
        index = {
            "schema": "aflow.utsw_no_n4_paired_external.cache",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "arm": arm, "split": "external_cross_stage", "external_confirmation": True,
            "cases": p.CASES, "subjects": p.CASES, "case_ids": case_ids, "subject_ids": case_ids,
            "model_shape": list(p.MODEL_SHAPE), "image_dtype": "float32", "label_dtype": "uint8",
            "images_sha256": p.sha(root / "images.f32"), "labels_sha256": p.sha(root / "labels.u8"),
            "geometry": {subject: details[subject]["geometry"] for subject in case_ids},
            "preprocessing": {
                "source": "official primary FeTS brain_t1ce", "canonical_ras": True,
                "n4_bias_field_correction": arm == "canonical",
                "n4_fit_mask": "positive source voxels" if arm == "canonical" else None,
                "spatial": "positive-support crop then resize128",
                "normalization": "source positive-support population z-score",
                "uses_gt_for_localization": False,
            },
            "target": "UTSW WT = label>0", "source_download_sha256": p.sha(download_path),
            "preaccess_sha256": p.sha(p.PREACCESS), "builder_sha256": p.sha(Path(__file__)),
            "elapsed_seconds": time.monotonic() - start,
        }
        p.write_new(root / "index.json", index)
        indices[arm] = index
    memmap_gates = {arm: validate_memmaps(p.CACHE_ROOT / arm) for arm in ("canonical", "no_n4")}
    gates = {
        "all_362_label_geometry_pair_gates": all(
            value["pair_gates"]["labels_exact"] and value["pair_gates"]["geometry_exact"]
            for value in details.values()
        ),
        "images_files_different": indices["canonical"]["images_sha256"] != indices["no_n4"]["images_sha256"],
        "labels_file_exact": indices["canonical"]["labels_sha256"] == indices["no_n4"]["labels_sha256"],
        "geometry_all_exact": indices["canonical"]["geometry"] == indices["no_n4"]["geometry"],
        "fixed_three_archived_exact": all(all(row.values()) for row in fixed_checks.values()),
        "canonical_memmap_valid": all(memmap_gates["canonical"].values()),
        "no_n4_memmap_valid": all(memmap_gates["no_n4"].values()),
        "file_sizes_exact": all(
            (p.CACHE_ROOT / arm / "images.f32").stat().st_size == p.CASES * 128 ** 3 * 4
            and (p.CACHE_ROOT / arm / "labels.u8").stat().st_size == p.CASES * 128 ** 3
            for arm in ("canonical", "no_n4")
        ),
    }
    receipt = {
        "schema": "aflow.utsw_no_n4_paired_external.cache_receipt",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(gates.values()) else "failed",
        "gates": gates, "all_gates_passed": all(gates.values()),
        "indices": {
            arm: {"path": str(p.CACHE_ROOT / arm / "index.json"), "sha256": p.sha(p.CACHE_ROOT / arm / "index.json"),
                  "images_sha256": indices[arm]["images_sha256"], "labels_sha256": indices[arm]["labels_sha256"]}
            for arm in ("canonical", "no_n4")
        },
        "fixed_three_checks": fixed_checks, "memmap_gates": memmap_gates,
        "elapsed_seconds": time.monotonic() - start,
        "label_voxels_accessed": True, "model_inference": False,
    }
    p.write_new(p.RECORD / "CACHE_RECEIPT.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not receipt["all_gates_passed"]:
        raise RuntimeError("UTSW paired cache gates failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    main(parser.parse_args().workers)
