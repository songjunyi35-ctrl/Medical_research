"""Configurable T1ce/WT preprocessing for SAM-Med3D fine-tuning."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import torchio as tio

from sammed3d.t1ce_binary import prepare_t1ce_supervised_arrays
from sammed3d.t1ce_wt_prompt import (
    MODEL_SHAPE,
    load_official_crop,
    validate_row,
)


GT_CROP = "gt_crop"
WHOLE_RESIZE = "whole_resize"
NO_AUGMENTATION = "none"
AFFINE_ELASTIC_NOISE = "affine_elastic_noise"
RANDOM_FLIP = "random_flip"
SPATIAL_PREPROCESSING_CHOICES = (GT_CROP, WHOLE_RESIZE)
TRAIN_AUGMENTATION_CHOICES = (
    NO_AUGMENTATION,
    AFFINE_ELASTIC_NOISE,
    RANDOM_FLIP,
)


@dataclass(frozen=True)
class T1ceWtPreprocessConfig:
    spatial_preprocessing: str = GT_CROP
    train_augmentation: str = NO_AUGMENTATION
    model_shape: tuple[int, int, int] = MODEL_SHAPE

    def __post_init__(self) -> None:
        if self.spatial_preprocessing not in SPATIAL_PREPROCESSING_CHOICES:
            raise ValueError(
                "spatial_preprocessing must be one of "
                f"{SPATIAL_PREPROCESSING_CHOICES}"
            )
        if self.train_augmentation not in TRAIN_AUGMENTATION_CHOICES:
            raise ValueError(
                f"train_augmentation must be one of {TRAIN_AUGMENTATION_CHOICES}"
            )
        if len(self.model_shape) != 3 or any(
            int(size) <= 0 for size in self.model_shape
        ):
            raise ValueError("model_shape must contain three positive integers")

    def to_protocol_dict(self) -> dict[str, object]:
        if self.spatial_preprocessing == WHOLE_RESIZE:
            geometry = {
                "name": WHOLE_RESIZE,
                "target_shape": list(self.model_shape),
                "image_interpolation": "trilinear align_corners=False",
                "label_interpolation": "nearest-exact",
                "uses_gt_for_localization": False,
            }
            normalization = (
                "source nonzero-voxel population z-score before resize; "
                "background remains zero; no batch re-normalization"
            )
        elif self.train_augmentation == RANDOM_FLIP:
            augmentation = {
                "name": RANDOM_FLIP,
                "applies_to": "train split only",
                "validation_augmentation": False,
                "random_flip": {
                    "axes": [0, 1, 2],
                    "probability": 0.5,
                    "source": "SAM-Med3D official train.py",
                },
            }
        else:
            geometry = {
                "name": GT_CROP,
                "target_shape": list(self.model_shape),
                "implementation": "TorchIO CropOrPad(mask_name=label)",
                "uses_gt_for_localization": True,
            }
            normalization = (
                "post-crop positive-voxel sample z-score matching the v1 "
                "TorchIO convention; no batch re-normalization"
            )
        augmentation: dict[str, object]
        if self.train_augmentation == AFFINE_ELASTIC_NOISE:
            augmentation = {
                "name": AFFINE_ELASTIC_NOISE,
                "applies_to": "train split only",
                "validation_augmentation": False,
                "random_affine": {
                    "probability": 0.5,
                    "scales": 0,
                    "degrees": 15,
                    "translation": 10,
                    "image_interpolation": "linear",
                    "label_interpolation": "nearest",
                },
                "random_elastic_deformation": {
                    "probability": 0.2,
                    "num_control_points": 7,
                    "max_displacement": 5,
                    "locked_borders": 2,
                    "image_interpolation": "linear",
                    "label_interpolation": "nearest",
                },
                "random_gaussian_noise": {
                    "probability": 0.15,
                    "mean": 0,
                    "std": [0, 0.1],
                },
            }
        else:
            augmentation = {
                "name": NO_AUGMENTATION,
                "applies_to": "none",
                "validation_augmentation": False,
            }
        return {
            "orientation": "canonical RAS",
            "geometry": geometry,
            "normalization": normalization,
            "augmentation": augmentation,
        }


def _load_canonical_nifti(path: Path, description: str) -> nib.Nifti1Image:
    if not path.is_file():
        raise ValueError(f"missing {description}: {path}")
    try:
        image = nib.as_closest_canonical(nib.load(str(path), mmap=True))
    except (OSError, ValueError) as exc:
        raise ValueError(f"failed to load {description} {path}: {exc}") from exc
    if len(image.shape) != 3:
        raise ValueError(f"{description} must be 3D, got {image.shape}")
    if not np.isfinite(image.affine).all():
        raise ValueError(f"{description} affine must be finite")
    return image


def load_whole_volume_resize(
    row: Mapping[str, object],
    *,
    model_shape: Sequence[int] = MODEL_SHAPE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load, canonicalize, normalize once, and resize the complete volume."""
    validate_row(row)
    case_id = str(row["case_id"])
    image_nifti = _load_canonical_nifti(
        Path(str(row["image_path"])), f"{case_id} T1ce"
    )
    label_nifti = _load_canonical_nifti(
        Path(str(row["label_path"])), f"{case_id} segmentation"
    )
    if image_nifti.shape != label_nifti.shape:
        raise ValueError(f"{case_id} image/label shape mismatch")
    if not np.allclose(
        image_nifti.affine, label_nifti.affine, rtol=0.0, atol=1e-5
    ):
        raise ValueError(f"{case_id} canonical image/label affine mismatch")
    image = np.ascontiguousarray(np.asanyarray(image_nifti.dataobj))
    label = np.ascontiguousarray(np.asanyarray(label_nifti.dataobj))
    spacing = nib.affines.voxel_sizes(image_nifti.affine)
    view = prepare_t1ce_supervised_arrays(
        case_id,
        image,
        label,
        spacing,
        output_shape=model_shape,
    )
    target = view.target.float().contiguous()
    if not bool(target.any()):
        raise RuntimeError(f"{case_id} WT is empty after whole-volume resize")
    return view.image.float().contiguous(), target


def positive_sample_zscore(image: torch.Tensor) -> torch.Tensor:
    """Match the legacy post-crop positive-mask sample-standard-deviation z-score."""
    output = image.float().clone()
    mask = output > 0
    if not bool(mask.any()):
        raise ValueError("T1ce image has no positive voxels for normalization")
    values = output[mask]
    std = values.std(unbiased=True)
    if not bool(torch.isfinite(std)) or float(std) <= 1e-6:
        raise ValueError("T1ce positive-voxel standard deviation is invalid")
    return (output - values.mean()) / std


def load_preprocessed_case(
    row: Mapping[str, object],
    config: T1ceWtPreprocessConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if config.spatial_preprocessing == WHOLE_RESIZE:
        return load_whole_volume_resize(row, model_shape=config.model_shape)
    image, target = load_official_crop(row)
    if tuple(config.model_shape) != MODEL_SHAPE:
        raise ValueError("gt_crop only supports the frozen 128^3 model shape")
    return positive_sample_zscore(image), target.float().contiguous()


def build_training_augmentation(
    config: T1ceWtPreprocessConfig,
) -> tio.Compose | None:
    if config.train_augmentation == NO_AUGMENTATION:
        return None
    if config.train_augmentation == RANDOM_FLIP:
        return tio.Compose([tio.RandomFlip(axes=(0, 1, 2))])
    return tio.Compose(
        [
            tio.RandomAffine(
                scales=0,
                degrees=15,
                translation=10,
                default_pad_value=0,
                default_pad_label=0,
                image_interpolation="linear",
                label_interpolation="nearest",
                p=0.5,
            ),
            tio.RandomElasticDeformation(
                num_control_points=7,
                max_displacement=5,
                locked_borders=2,
                image_interpolation="linear",
                label_interpolation="nearest",
                p=0.2,
            ),
            tio.RandomNoise(mean=0, std=(0, 0.1), p=0.15),
        ]
    )


def apply_training_augmentation(
    image: torch.Tensor,
    target: torch.Tensor,
    transform: tio.Compose,
) -> tuple[torch.Tensor, torch.Tensor]:
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=image),
        target=tio.LabelMap(tensor=target),
    )
    transformed = transform(subject)
    transformed_image = transformed.image.data.float().contiguous()
    transformed_target = (transformed.target.data > 0).float().contiguous()
    if not bool(transformed_target.any()):
        raise RuntimeError("training augmentation removed the complete WT target")
    return transformed_image, transformed_target


class T1ceWtTrainingDataset(Dataset):
    """Dataset that applies stochastic transforms only to declared train rows."""

    def __init__(
        self,
        rows: list[dict],
        *,
        config: T1ceWtPreprocessConfig,
    ) -> None:
        if not rows:
            raise ValueError("T1ce/WT dataset cannot be empty")
        for row in rows:
            validate_row(row)
        splits = {str(row.get("split")) for row in rows}
        if len(splits) != 1 or not splits <= {"train", "validation"}:
            raise ValueError(
                "T1ce/WT dataset rows must have one common train/validation split"
            )
        self.rows = rows
        self.config = config
        self.split = next(iter(splits))
        self.augmentation = (
            build_training_augmentation(config)
            if self.split == "train"
            else None
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        image, target = load_preprocessed_case(row, self.config)
        if self.augmentation is not None:
            image, target = apply_training_augmentation(
                image, target, self.augmentation
            )
        expected_shape = (1, *self.config.model_shape)
        if tuple(image.shape) != expected_shape:
            raise RuntimeError(
                f"{row['case_id']} image shape {tuple(image.shape)} != {expected_shape}"
            )
        if tuple(target.shape) != expected_shape:
            raise RuntimeError(
                f"{row['case_id']} target shape {tuple(target.shape)} != {expected_shape}"
            )
        return {
            "image": image,
            "label": target,
            "case_id": str(row["case_id"]),
            "target": "WT",
        }


def identity_preprocessed_batch(image: torch.Tensor) -> torch.Tensor:
    """The dataset already normalized each volume exactly once."""
    return image
