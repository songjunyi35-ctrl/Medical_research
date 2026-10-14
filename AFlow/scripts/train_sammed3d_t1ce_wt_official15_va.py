#!/usr/bin/env python
"""Fail-closed VA-HDFS entry point for the official 1.5 mm T1ce/WT arm.

This wrapper deliberately leaves the byte-pinned historical DPD-SAM-aligned
trainer unchanged.  It reuses that trainer's optimization loop while binding
it to the migrated development manifests and to the deterministic official
SAM-Med3D preprocessing cache:

  Resample(1.5 mm) -> ToCanonical -> GT CropOrPad(128^3) -> positive z-score,
  with no N4 and train-only RandomFlip.

The sealed test split is not accepted or read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_sammed3d_t1ce_wt_dpdsam_aligned as base  # noqa: E402

SCHEMA = "aflow.sammed3d.t1ce_wt_official15_va_training"
RUN_IDENTITY = (
    "sammed3d_official15_va_gtcrop_fullunfreeze_a100x2_"
    "train972_val108_e200_trainseed20260805_r20260812"
)
BASE_TRAINER_SHA256 = (
    "31c114fe24e6ad419c3eac29a50775a5ab9d7dedf758ae907c3d7f417829c20c"
)
BUILDER_SHA256 = (
    "ee882e0eed4704b398568dd2a36683ac1312cb429c5be85de1636816599114c9"
)
TRAIN_MANIFEST_SHA256 = (
    "6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65"
)
VAL_MANIFEST_SHA256 = (
    "f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb"
)
TRAIN_COUNT = 972
VAL_COUNT = 108
OFFICIAL_CACHE_PROTOCOL = "aflow.sammed3d.t1ce_wt_official15_cache"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_protocol_dict() -> dict[str, object]:
    """Return the explicit protocol record, avoiding the legacy config bug."""
    return {
        "orientation": "canonical RAS after 1.5 mm paired resampling",
        "geometry": {
            "name": "gt_crop_1p5mm",
            "target_spacing_mm": [1.5, 1.5, 1.5],
            "target_shape": [128, 128, 128],
            "implementation": "TorchIO CropOrPad(mask_name=label)",
            "image_interpolation": "linear",
            "label_interpolation": "nearest (TorchIO LabelMap)",
            "uses_gt_for_localization": True,
        },
        "normalization": (
            "post-crop positive-voxel sample z-score applied to the whole patch; "
            "no N4 and no batch re-normalization"
        ),
        "augmentation": {
            "name": "random_flip",
            "applies_to": "train split only",
            "validation_augmentation": False,
            "random_flip": {
                "axes": [0, 1, 2],
                "per_axis_probability": 0.5,
                "source": "SAM-Med3D official train.py / TorchIO 1.2.0",
            },
        },
        "cache_protocol_schema": OFFICIAL_CACHE_PROTOCOL,
        "cache_builder_sha256": BUILDER_SHA256,
        "reused_training_loop_sha256": BASE_TRAINER_SHA256,
        "n4_bias_field_correction": False,
    }


def validate_cache_contract(cache_root: Path) -> dict[str, dict[str, Any]]:
    """Reject a cache unless every scientific provenance field is official15."""
    expected = {
        "train": (TRAIN_MANIFEST_SHA256, TRAIN_COUNT),
        "validation": (VAL_MANIFEST_SHA256, VAL_COUNT),
    }
    indexes: dict[str, dict[str, Any]] = {}
    expected_order = [
        "WT123 binary LabelMap",
        "TorchIO Resample 1.5mm isotropic",
        "TorchIO ToCanonical",
        "TorchIO CropOrPad(mask_name=label, target_shape=128^3)",
        "positive-voxel sample z-score applied to whole patch",
    ]
    for split, (manifest_sha, count) in expected.items():
        path = cache_root / split / "index.json"
        if not path.is_file():
            raise FileNotFoundError(f"official15 cache index missing: {path}")
        index = json.loads(path.read_text())
        preprocessing = index.get("preprocessing", {})
        checks = {
            "schema": index.get("schema") == "aflow.sammed3d.t1ce_wt_preprocess_cache",
            "protocol_schema": index.get("protocol_schema") == OFFICIAL_CACHE_PROTOCOL,
            "manifest_sha256": index.get("manifest_sha256") == manifest_sha,
            "count": int(index.get("count", -1)) == count,
            "split": index.get("split") == split,
            "model_shape": index.get("model_shape") == [128, 128, 128],
            "builder_sha256": index.get("builder_sha256") == BUILDER_SHA256,
            "target_spacing": preprocessing.get("target_spacing_mm") == [1.5, 1.5, 1.5],
            "order": preprocessing.get("order") == expected_order,
            "spatial": preprocessing.get("spatial") == "gt_crop_1p5mm",
            "uses_gt": preprocessing.get("uses_gt_for_localization") is True,
            "no_n4": preprocessing.get("n4_bias_field_correction") is False,
            "augmentation": preprocessing.get("augmentation")
            == "NOT cached; official RandomFlip applied by train dataset",
            "test_access": index.get("test_access") is False,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"{split} cache violates official15 contract: {failed}")
        indexes[split] = index
    return indexes


def validate_official_args(
    args: argparse.Namespace, *, world_size: int
) -> None:
    expected_common = {
        "spatial_preprocessing": "gt_crop",
        "train_augmentation": "random_flip",
        "unfreeze": "full",
        "encoder_lr": 1e-4,
        "head_lr": 1e-5,
        "weight_decay": 0.1,
        "lr_milestones": [120, 180],
        "lr_gamma": 0.1,
    }
    failed = [
        name for name, expected in expected_common.items()
        if getattr(args, name) != expected
    ]
    if world_size != 2:
        failed.append("world_size=2")
    if args.mode == "formal":
        formal = {
            "epochs": 200,
            "batch_size": 6,
            "accumulation_steps": 1,
            "min_epochs_before_stop": 201,
            "patience": 201,
        }
        failed.extend(
            name for name, expected in formal.items() if getattr(args, name) != expected
        )
    else:
        smoke = {"batch_size": 2, "accumulation_steps": 3}
        failed.extend(
            name for name, expected in smoke.items() if getattr(args, name) != expected
        )
    if failed:
        raise ValueError(f"arguments violate frozen official15 A100x2 protocol: {failed}")


def bind_official_protocol() -> None:
    actual = sha256_file(Path(base.__file__).resolve())
    if actual != BASE_TRAINER_SHA256:
        raise RuntimeError(f"reused trainer SHA {actual} != {BASE_TRAINER_SHA256}")
    base.TRAIN_MANIFEST_SHA256 = TRAIN_MANIFEST_SHA256
    base.VAL_MANIFEST_SHA256 = VAL_MANIFEST_SHA256
    base.TRAIN_COUNT = TRAIN_COUNT
    base.VAL_COUNT = VAL_COUNT
    base.SCHEMA = SCHEMA
    base.RUN_IDENTITY = RUN_IDENTITY
    base.T1ceWtPreprocessConfig.to_protocol_dict = lambda _self: official_protocol_dict()
    # The base loop records Path(__file__) as its runner.  Point that record at
    # this wrapper while retaining the reused-loop SHA inside the protocol dict.
    base.__file__ = str(Path(__file__).resolve())


def main(argv: list[str] | None = None) -> None:
    args = base.parse_args(argv)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    validate_official_args(args, world_size=world)
    cache_root = Path(args.cache_root)
    if not cache_root.is_absolute():
        cache_root = ROOT / cache_root
    validate_cache_contract(cache_root)
    bind_official_protocol()
    base.main(argv)


if __name__ == "__main__":
    main()
