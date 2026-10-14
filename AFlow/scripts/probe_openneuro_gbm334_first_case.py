#!/usr/bin/env python3
"""Download and preprocess only the frozen first OpenNeuro-GBM334 case."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from sammed3d import t1ce_binary
from scripts.build_no_n4_clean_cache import arrays
from scripts import openneuro_gbm334_protocol as p


ORIGINAL_CASE_ID = t1ce_binary._case_id


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def download(entry: dict, path: Path, retries: int = 3) -> None:
    for attempt in range(retries):
        try:
            digest = hashlib.sha256()
            count = 0
            request = urllib.request.Request(entry["url"], headers={"User-Agent": "AFlow-GBM334/1.0"})
            with urllib.request.urlopen(request, timeout=180) as source, path.open("wb") as target:
                while True:
                    block = source.read(8 << 20)
                    if not block:
                        break
                    target.write(block)
                    digest.update(block)
                    count += len(block)
            if count != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"download checksum: {path.name}")
            return
        except Exception:
            path.unlink(missing_ok=True)
            if attempt + 1 == retries:
                raise
            time.sleep(1 + attempt)


def prepare(row: dict, directory: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    image_path = directory / "image.nii.gz"
    label_path = directory / "label.nii.gz"
    download(row["image"], image_path)
    download(row["label"], label_path)
    image_nifti = nib.as_closest_canonical(nib.load(str(image_path), mmap=False))
    label_nifti = nib.as_closest_canonical(nib.load(str(label_path), mmap=False))
    affine_error = float(np.max(np.abs(image_nifti.affine - label_nifti.affine)))
    if image_nifti.shape != label_nifti.shape or affine_error > 1e-5:
        raise ValueError(f"geometry mismatch: {image_nifti.shape} {label_nifti.shape} {affine_error}")
    image = np.ascontiguousarray(np.asarray(image_nifti.dataobj))
    raw = np.ascontiguousarray(np.asarray(label_nifti.dataobj))
    rounded = np.rint(raw)
    values = sorted(int(value) for value in np.unique(rounded))
    if (
        not np.isfinite(image).all()
        or not (image > 0).any()
        or not np.isfinite(raw).all()
        or not np.allclose(raw, rounded, rtol=0, atol=1e-5)
        or not set(values).issubset({0, 1, 2, 3})
    ):
        raise ValueError(f"array domain: {values}")
    binary = np.ascontiguousarray(rounded > 0, dtype=np.uint8)
    if not binary.any() or not (binary == 0).any():
        raise ValueError("invalid WT")
    output_image, output_label, geometry = arrays(
        row["subject_id"],
        image,
        binary,
        nib.affines.voxel_sizes(image_nifti.affine),
        use_n4=True,
    )
    origin = geometry["origin"]
    stop = [a + b for a, b in zip(origin, geometry["extent"])]
    crop = tuple(slice(a, b) for a, b in zip(origin, stop))
    outside_wt = int(binary.sum() - binary[crop].sum())
    details = {
        "native_shape": list(image_nifti.shape),
        "spacing_mm": [float(x) for x in nib.affines.voxel_sizes(image_nifti.affine)],
        "axis_codes": list(nib.aff2axcodes(image_nifti.affine)),
        "affine_error": affine_error,
        "raw_values": values,
        "raw_wt": int(binary.sum()),
        "wt_outside_image_positive_crop": outside_wt,
        "resized_wt": int(output_label.sum()),
        "crop_geometry": geometry,
        "image_array_sha256": array_sha(output_image),
        "label_array_sha256": array_sha(output_label),
        "image_source_sha256": p.sha256(image_path),
        "label_source_sha256": p.sha256(label_path),
    }
    if outside_wt or not output_label.any() or not np.isfinite(output_image).all():
        raise ValueError(f"postprocess gates: {details}")
    return output_image, output_label, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=p.RECORD / "FIRST_CASE_PROBE.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = p.cases()
    row = rows[0]
    torch.set_num_threads(1)
    import SimpleITK as sitk

    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    t1ce_binary._case_id = lambda value: value if value == row["subject_id"] else ORIGINAL_CASE_ID(value)
    with tempfile.TemporaryDirectory(prefix="aflow_gbm334_probe_", dir="/tmp") as first_dir:
        image, label, first = prepare(row, Path(first_dir))
    with tempfile.TemporaryDirectory(prefix="aflow_gbm334_repeat_", dir="/tmp") as repeat_dir:
        _, _, second = prepare(row, Path(repeat_dir))
    gates = {
        "source_sha_exact": first["image_source_sha256"] == row["image"]["sha256"]
        and first["label_source_sha256"] == row["label"]["sha256"],
        "geometry_exact": first["affine_error"] <= 1e-5,
        "label_domain": set(first["raw_values"]).issubset({0, 1, 2, 3}),
        "zero_wt_crop_loss": first["wt_outside_image_positive_crop"] == 0,
        "output_finite_binary_nonempty": np.isfinite(image).all()
        and set(np.unique(label)).issubset({0, 1})
        and bool(label.any()),
        "repeat_image_byte_exact": first["image_array_sha256"] == second["image_array_sha256"],
        "repeat_label_byte_exact": first["label_array_sha256"] == second["label_array_sha256"],
    }
    receipt = {
        "schema": "aflow.openneuro_gbm334.first_case_probe",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(gates.values()) else "failed",
        "subject_id": row["subject_id"],
        "snapshot": "ds007045:1.0.1",
        "gates": gates,
        "details": first,
        "repeat_details": second,
        "other_cases_payload_accessed": False,
        "checkpoint_loaded": False,
        "model_forwards": 0,
        "runner_sha256": p.sha256(Path(__file__)),
    }
    p.write_new(args.output, receipt)
    print(json.dumps(receipt, indent=2))
    if receipt["status"] != "passed":
        raise RuntimeError(gates)


if __name__ == "__main__":
    main()
