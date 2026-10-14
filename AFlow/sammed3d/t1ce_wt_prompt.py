"""T1ce/WT preprocessing for interactive SAM-Med3D fine-tuning."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import nibabel as nib
import numpy as np
import torch
import torchio as tio


SCHEMA = "aflow.brats2024.sammed3d_t1ce_wt_inner_split"
MODEL_SHAPE = (128, 128, 128)
ALLOWED_RAW_LABELS = frozenset({0, 1, 2, 3, 4})
WT_RAW_LABELS = frozenset({1, 2, 3})


def validate_row(row: Mapping[str, object]) -> None:
    if row.get("schema") != SCHEMA:
        raise ValueError(f"unsupported T1ce/WT row schema: {row.get('schema')!r}")
    if row.get("modality") != "t1ce" or row.get("target") != "WT":
        raise ValueError("T1ce/WT row must declare modality=t1ce and target=WT")
    if tuple(row.get("foreground_labels", ())) != tuple(sorted(WT_RAW_LABELS)):
        raise ValueError("T1ce/WT row must declare foreground_labels=[1,2,3]")
    if tuple(row.get("excluded_labels", ())) != (4,):
        raise ValueError("T1ce/WT row must declare excluded_labels=[4]")
    for field in ("case_id", "subject_id", "image_path", "label_path"):
        if not isinstance(row.get(field), str) or not str(row[field]).strip():
            raise ValueError(f"T1ce/WT row requires non-empty {field}")


def wt_target(segmentation: np.ndarray | torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(segmentation)
    if values.ndim != 3:
        raise ValueError(f"segmentation must be 3D, got {tuple(values.shape)}")
    if values.is_floating_point():
        rounded = values.round()
        if not torch.isfinite(values).all() or not torch.allclose(
            values, rounded, rtol=0.0, atol=1e-5
        ):
            raise ValueError("segmentation must contain finite integer labels")
        values = rounded.to(torch.int64)
    else:
        values = values.to(torch.int64)
    labels = {int(value) for value in torch.unique(values).tolist()}
    if not labels <= ALLOWED_RAW_LABELS:
        raise ValueError(f"segmentation contains unexpected labels: {sorted(labels)}")
    return (values == 1) | (values == 2) | (values == 3)


def load_official_crop(row: Mapping[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw cropped image/WT tensors; trainer applies the upstream z-score."""
    validate_row(row)
    image_path = Path(str(row["image_path"]))
    label_path = Path(str(row["label_path"]))
    image_nifti = nib.load(str(image_path), mmap=True)
    label_nifti = nib.load(str(label_path), mmap=True)
    image = np.asanyarray(image_nifti.dataobj)
    label = np.asanyarray(label_nifti.dataobj)
    if image.shape != label.shape:
        raise ValueError(f"{row['case_id']} image/label shape mismatch")
    if not np.allclose(
        image_nifti.affine, label_nifti.affine, rtol=0.0, atol=1e-5
    ):
        raise ValueError(f"{row['case_id']} image/label affine mismatch")
    target = wt_target(label)
    if not bool(target.any()):
        raise ValueError(f"{row['case_id']} has empty WT foreground")
    subject = tio.Subject(
        image=tio.ScalarImage(
            tensor=torch.from_numpy(np.ascontiguousarray(image))[None].float(),
            affine=np.asarray(image_nifti.affine, dtype=np.float64),
        ),
        label=tio.LabelMap(
            tensor=target[None].to(torch.uint8),
            affine=np.asarray(label_nifti.affine, dtype=np.float64),
        ),
    )
    transformed = tio.Compose(
        [
            tio.ToCanonical(),
            tio.CropOrPad(mask_name="label", target_shape=MODEL_SHAPE),
        ]
    )(subject)
    cropped_image = transformed.image.data.float().contiguous()
    cropped_target = transformed.label.data.float().contiguous()
    if tuple(cropped_image.shape) != (1, *MODEL_SHAPE):
        raise RuntimeError("T1ce official crop did not produce [1,128,128,128]")
    if tuple(cropped_target.shape) != (1, *MODEL_SHAPE):
        raise RuntimeError("WT official crop did not produce [1,128,128,128]")
    if not bool(cropped_target.any()):
        raise RuntimeError(f"{row['case_id']} WT crop is unexpectedly empty")
    return cropped_image, cropped_target
