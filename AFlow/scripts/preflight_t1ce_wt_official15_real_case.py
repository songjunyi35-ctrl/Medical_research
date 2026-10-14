#!/usr/bin/env python
"""Voxelwise real-train-case check of the official15 cache transformation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import torch
import torchio as tio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_t1ce_wt_official15_cache import preprocess_row  # noqa: E402


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    row = rows[0]
    if row.get("split") != "train":
        raise ValueError("real-case preflight accepts a train manifest only")

    actual_image, actual_label, qc = preprocess_row(row)

    image_nii = nib.load(row["image_path"], mmap=True)
    label_nii = nib.load(row["label_path"], mmap=True)
    image = np.ascontiguousarray(np.asanyarray(image_nii.dataobj), dtype=np.float32)
    raw_label = np.asanyarray(label_nii.dataobj)
    wt = np.isin(raw_label, [1, 2, 3]).astype(np.uint8)
    subject = tio.Subject(
        image=tio.ScalarImage(
            tensor=torch.from_numpy(image)[None], affine=image_nii.affine
        ),
        label=tio.LabelMap(
            tensor=torch.from_numpy(wt)[None], affine=label_nii.affine
        ),
    )
    deterministic = tio.CropOrPad(
        mask_name="label", target_shape=(128, 128, 128)
    )(
        tio.ToCanonical()(
            tio.Resample(
                target=(1.5, 1.5, 1.5), image_interpolation="linear"
            )(subject)
        )
    )
    positive_mask = deterministic.image.data > 0
    expected_image = tio.ZNormalization(masking_method=lambda x: x > 0)(
        deterministic.image
    ).data.numpy()
    expected_label = (deterministic.label.data > 0).to(torch.uint8).numpy()

    image_max_abs_diff = float(np.max(np.abs(actual_image - expected_image)))
    labels_equal = bool(np.array_equal(actual_label, expected_label))
    normalized_positive = torch.from_numpy(actual_image)[positive_mask]
    result = {
        "schema": "aflow.sammed3d.t1ce_wt_official15_real_case_preflight",
        "case_id": row["case_id"],
        "split": "train",
        "manifest": str(manifest),
        "image_voxelwise_equal_atol_1e-6": image_max_abs_diff <= 1e-6,
        "image_max_abs_diff": image_max_abs_diff,
        "label_voxelwise_equal": labels_equal,
        "image_sha256": array_sha256(actual_image),
        "label_sha256": array_sha256(actual_label),
        "normalized_source_positive_mean": float(normalized_positive.mean()),
        "normalized_source_positive_sample_std": float(
            normalized_positive.std(unbiased=True)
        ),
        "qc": qc,
        "n4_bias_field_correction": False,
        "test_access": False,
    }
    if not result["image_voxelwise_equal_atol_1e-6"] or not labels_equal:
        raise RuntimeError(json.dumps(result, indent=2))
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
