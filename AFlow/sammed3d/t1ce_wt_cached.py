"""Memmap-backed T1ce/WT dataset.

`T1ceWtTrainingDataset` recomputes the full deterministic preprocessing chain
(canonical NIfTI load, nonzero z-score, trilinear/nearest-exact resize to 128^3)
on every __getitem__, i.e. once per case *per epoch*. That is what starved the
GPU: measured utilization was 32.5% while two dataloader workers each consumed
~2.5 CPU cores.

This dataset reads the identical tensors from a precomputed cache built by
`scripts/build_t1ce_wt_cache.py` and applies only the genuinely per-epoch part,
the stochastic train augmentation. The cache is proven bit-identical to the live
pipeline when N4 is disabled.

Reads go through np.memmap, so the OS page cache serves repeat epochs from RAM
without a per-worker copy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from sammed3d.t1ce_wt_training import (
    T1ceWtPreprocessConfig,
    apply_training_augmentation,
    build_training_augmentation,
)

CACHE_SCHEMA = "aflow.sammed3d.t1ce_wt_preprocess_cache"
MODEL_SHAPE = (128, 128, 128)


class CachedT1ceWtDataset(Dataset):
    """Serve cached 128^3 volumes, augmenting train rows only."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        split: str,
        config: T1ceWtPreprocessConfig,
        expected_manifest_sha256: str | None = None,
        expected_count: int | None = None,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError(f"split must be train or validation, got {split!r}")
        cache_dir = Path(cache_dir)
        index_path = cache_dir / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"cache index missing (incomplete build?): {index_path}"
            )
        index = json.loads(index_path.read_text())
        if index.get("schema") != CACHE_SCHEMA:
            raise ValueError(f"unexpected cache schema: {index.get('schema')!r}")
        if tuple(index["model_shape"]) != MODEL_SHAPE:
            raise ValueError(f"cache model_shape != {MODEL_SHAPE}")
        if index.get("test_access") is not False:
            raise ValueError("cache must declare test_access=false")
        if expected_manifest_sha256 is not None and (
            index["manifest_sha256"] != expected_manifest_sha256
        ):
            raise ValueError(
                f"cache manifest SHA {index['manifest_sha256']} != "
                f"expected {expected_manifest_sha256}"
            )
        count = int(index["count"])
        if expected_count is not None and count != expected_count:
            raise ValueError(f"cache has {count} cases, expected {expected_count}")

        images_path = cache_dir / "images.f32"
        labels_path = cache_dir / "labels.u8"
        expect_img = count * int(np.prod(MODEL_SHAPE)) * 4
        expect_lbl = count * int(np.prod(MODEL_SHAPE))
        if images_path.stat().st_size != expect_img:
            raise ValueError(f"{images_path} size {images_path.stat().st_size} != {expect_img}")
        if labels_path.stat().st_size != expect_lbl:
            raise ValueError(f"{labels_path} size {labels_path.stat().st_size} != {expect_lbl}")

        self.cache_dir = cache_dir
        self.index = index
        self.count = count
        self.case_ids: list[str] = list(index["case_ids"])
        if len(self.case_ids) != count:
            raise ValueError("cache index case_ids length != count")
        self.split = split
        self.config = config
        self.augmentation = (
            build_training_augmentation(config) if split == "train" else None
        )
        self._images: np.memmap | None = None
        self._labels: np.memmap | None = None

    # memmaps are opened lazily so each dataloader worker gets its own handle
    # rather than inheriting one across the fork/spawn boundary.
    def _ensure_open(self) -> None:
        if self._images is None:
            self._images = np.memmap(
                self.cache_dir / "images.f32", dtype=np.float32, mode="r",
                shape=(self.count, *MODEL_SHAPE),
            )
        if self._labels is None:
            self._labels = np.memmap(
                self.cache_dir / "labels.u8", dtype=np.uint8, mode="r",
                shape=(self.count, *MODEL_SHAPE),
            )

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        self._ensure_open()
        assert self._images is not None and self._labels is not None
        image = torch.from_numpy(np.array(self._images[index], dtype=np.float32))[None]
        target = torch.from_numpy(
            np.array(self._labels[index], dtype=np.uint8).astype(np.float32)
        )[None]
        if self.augmentation is not None:
            image, target = apply_training_augmentation(image, target, self.augmentation)
        expected = (1, *MODEL_SHAPE)
        if tuple(image.shape) != expected or tuple(target.shape) != expected:
            raise RuntimeError(f"cached case {index} has unexpected shape")
        if not bool(target.any()):
            raise RuntimeError(f"cached case {index} has empty WT target")
        return {
            "image": image,
            "label": target,
            "case_id": self.case_ids[index],
            "target": "WT",
        }

    def provenance(self) -> dict:
        return {
            "cache_dir": str(self.cache_dir),
            "manifest": self.index["manifest"],
            "manifest_sha256": self.index["manifest_sha256"],
            "images_sha256": self.index["images_sha256"],
            "labels_sha256": self.index["labels_sha256"],
            "count": self.count,
            "preprocessing": self.index["preprocessing"],
        }
