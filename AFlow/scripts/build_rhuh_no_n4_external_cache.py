"""Build the frozen RHUH80 128^3 external cache after official label download."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from scripts import rhuh_no_n4_external_protocol as p
from scripts.audit_rhuh_no_n4_input_bridge import bridge

MODEL_SHAPE = (128, 128, 128)
CFG: dict[str, object] = {}


def initialize(config: dict) -> None:
    torch.set_num_threads(1)
    CFG.update(config)
    CFG["image_fd"] = os.open(str(config["images"]), os.O_WRONLY)
    CFG["label_fd"] = os.open(str(config["labels"]), os.O_WRONLY)


def integer_labels(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError("RHUH segmentation must be finite 3D")
    rounded = np.rint(array)
    if not np.allclose(array, rounded, rtol=0, atol=1e-5):
        raise ValueError("RHUH segmentation is not integer-valued")
    labels = np.ascontiguousarray(rounded, dtype=np.int16)
    values = {int(value) for value in np.unique(labels)}
    if not values <= {0, 1, 2, 3, 4} or not (values - {0}):
        raise ValueError(f"RHUH segmentation label domain invalid: {sorted(values)}")
    return labels


def prepare(item: dict, label_item: dict, expected_image_sha: str) -> tuple[np.ndarray, np.ndarray, dict]:
    image_path = Path(item["path"])
    label_path = Path(label_item["path"])
    if p.sha(image_path) != item["sha256"] or p.sha(label_path) != label_item["sha256"]:
        raise ValueError("RHUH raw file hash changed")
    image_nii = nib.as_closest_canonical(nib.load(image_path))
    label_nii = nib.as_closest_canonical(nib.load(label_path))
    if image_nii.shape != label_nii.shape or not np.allclose(image_nii.affine, label_nii.affine, rtol=0, atol=1e-5):
        raise ValueError("RHUH image/label geometry mismatch")
    raw_image = np.ascontiguousarray(np.asanyarray(image_nii.dataobj), dtype=np.float32)
    raw_labels = integer_labels(np.asanyarray(label_nii.dataobj))
    output_image, _, metadata = bridge(raw_image)
    if metadata["output_image_sha256"] != expected_image_sha:
        raise ValueError("RHUH no-N4 image differs from frozen input QC")
    origin, stop = metadata["bbox_origin"], metadata["bbox_stop"]
    slices = tuple(slice(low, high) for low, high in zip(origin, stop))
    target = torch.from_numpy(np.ascontiguousarray(raw_labels[slices] > 0)).reshape(1, 1, *raw_labels[slices].shape)
    resized = F.interpolate(target.float(), size=MODEL_SHAPE, mode="nearest-exact")[0, 0].bool()
    output_label = np.ascontiguousarray(resized.numpy(), dtype=np.uint8)
    if not output_label.any():
        raise ValueError("RHUH WT became empty at 128^3")
    geometry = {
        "native_shape": list(image_nii.shape),
        "native_affine_max_abs_delta": float(np.max(np.abs(image_nii.affine - label_nii.affine))),
        "bbox_origin": origin,
        "bbox_stop": stop,
        "axis_codes": list(nib.aff2axcodes(image_nii.affine)),
        "spacing": [float(value) for value in nib.affines.voxel_sizes(image_nii.affine)],
        "raw_label_values": sorted(int(value) for value in np.unique(raw_labels)),
        "raw_wt_voxels": int((raw_labels > 0).sum()),
        "model_wt_voxels": int(output_label.sum()),
    }
    return output_image, output_label, geometry


def worker(task: tuple[int, dict, dict, str]) -> tuple[int, str, dict]:
    slot, image, label, expected = task
    output_image, output_label, geometry = prepare(image, label, expected)
    voxels = 128 ** 3
    if os.pwrite(int(CFG["image_fd"]), output_image.tobytes(), slot * voxels * 4) != voxels * 4:
        raise IOError("short RHUH image cache write")
    if os.pwrite(int(CFG["label_fd"]), output_label.tobytes(), slot * voxels) != voxels:
        raise IOError("short RHUH label cache write")
    return slot, image["case_id"], geometry


def main(workers: int) -> None:
    from scripts.freeze_rhuh_no_n4_external import check_implementation

    check_implementation()
    p.verify_preaccess()
    download = json.loads((p.RECORD / "LABEL_DOWNLOAD.json").read_text())
    if download.get("all_official_match") is not True or download.get("preaccess_sha256") != p.sha(p.PREACCESS):
        raise ValueError("RHUH labels were not downloaded under current preaccess")
    images = json.loads((p.REVIEW / "RHUH_IMAGE_DOWNLOAD.json").read_text())["files"]
    labels = download["files"]
    input_rows = json.loads((p.RECORD / "INPUT_QC.json").read_text())["rows"]
    if [row["case_id"] for row in images] != [row["case_id"] for row in labels] or len(images) != 80:
        raise ValueError("RHUH image/label order mismatch")
    expected = {row["case_id"]: row["output_image_sha256"] for row in input_rows}
    output = p.CACHE
    if output.exists():
        raise FileExistsError("RHUH external cache already exists")
    bytes_needed = 80 * 128 ** 3 * 5
    if shutil.disk_usage(output.parent).free < bytes_needed + 2 * 1024 ** 3:
        raise RuntimeError("insufficient /tmp reserve for RHUH cache")
    output.mkdir()
    for name, size in (("images.f32", bytes_needed * 4 // 5), ("labels.u8", bytes_needed // 5)):
        with (output / name).open("xb") as handle:
            handle.truncate(size)
    tasks = [(slot, image, label, expected[image["case_id"]]) for slot, (image, label) in enumerate(zip(images, labels))]
    start = time.monotonic()
    with mp.get_context("spawn").Pool(
        workers, initializer=initialize,
        initargs=({"images": str(output / "images.f32"), "labels": str(output / "labels.u8")},),
    ) as pool:
        rows = list(pool.imap_unordered(worker, tasks, chunksize=1))
    rows.sort()
    geometry = {case_id: value for _, case_id, value in rows}
    case_ids = [item["case_id"] for item in images]
    subject_ids = [case_id.rsplit("_", 1)[0] for case_id in case_ids]
    index = {
        "schema": "aflow.rhuh_no_n4_external.cache",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": "external_confirmation",
        "external_confirmation": True,
        "cases": 80,
        "subjects": 40,
        "case_ids": case_ids,
        "subject_ids": subject_ids,
        "model_shape": [128, 128, 128],
        "image_dtype": "float32",
        "label_dtype": "uint8",
        "images_sha256": p.sha(output / "images.f32"),
        "labels_sha256": p.sha(output / "labels.u8"),
        "geometry": geometry,
        "preprocessing": {
            "background": "six-face mode decode",
            "n4_bias_field_correction": False,
            "spatial": "image-only positive-support crop then resize128",
            "normalization": "source positive-support population z-score",
            "uses_gt_for_localization": False,
        },
        "target": "WT = label>0",
        "input_qc_sha256": p.INPUT_QC_SHA,
        "label_download_sha256": p.sha(p.RECORD / "LABEL_DOWNLOAD.json"),
        "preaccess_sha256": p.sha(p.PREACCESS),
        "builder_sha256": p.sha(Path(__file__)),
        "elapsed_seconds": time.monotonic() - start,
    }
    p.write_new(output / "index.json", index)
    # Full memmap checks catch short/sparse writes and require all targets positive.
    image_map = np.memmap(output / "images.f32", mode="r", dtype=np.float32, shape=(80, 128, 128, 128))
    label_map = np.memmap(output / "labels.u8", mode="r", dtype=np.uint8, shape=(80, 128, 128, 128))
    gates = {
        "all_images_finite": bool(np.isfinite(image_map).all()),
        "all_labels_binary": bool(np.all((label_map == 0) | (label_map == 1))),
        "all_80_targets_positive": bool(np.all(label_map.reshape(80, -1).sum(1) > 0)),
        "image_bytes_exact": (output / "images.f32").stat().st_size == 80 * 128 ** 3 * 4,
        "label_bytes_exact": (output / "labels.u8").stat().st_size == 80 * 128 ** 3,
    }
    receipt = {
        "schema": "aflow.rhuh_no_n4_external.cache_receipt",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(gates.values()) else "failed",
        "cache": str(output),
        "cache_index_sha256": p.sha(output / "index.json"),
        "images_sha256": index["images_sha256"],
        "labels_sha256": index["labels_sha256"],
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "label_voxel_access": True,
        "model_inference": False,
        "elapsed_seconds": time.monotonic() - start,
    }
    p.write_new(p.RECORD / "CACHE_RECEIPT.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not receipt["all_gates_passed"]:
        raise RuntimeError("RHUH external cache gates failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    main(parser.parse_args().workers)
