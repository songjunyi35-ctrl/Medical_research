"""Memmap dataset for image-only brain-crop caches on a declared cubic grid.

This module is intentionally separate from :mod:`sammed3d.t1ce_wt_cached`.
Historical 128^3 results bind that module and its fixed ``MODEL_SHAPE``.  The
resolution experiment needs the same byte-level cache contract while allowing
the index to declare 128^3 or 160^3 without changing a historical entry point.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


CACHE_SCHEMA = "aflow.sammed3d.t1ce_wt_braincrop_variable_resolution_cache"
SPATIAL = "braincrop_resize"
FORMAL_RESOLUTIONS = (128, 160)


def cubic_shape(resolution: int) -> tuple[int, int, int]:
    value = int(resolution)
    if value <= 0 or value % 16 != 0:
        raise ValueError("resolution must be a positive multiple of 16")
    return (value, value, value)


def _resolve_case_slots(
    available_case_ids: list[str], requested_case_ids: list[str] | None
) -> tuple[list[int], list[str]]:
    if len(available_case_ids) != len(set(available_case_ids)):
        raise ValueError("source cache contains duplicate case_id values")
    if requested_case_ids is None:
        return list(range(len(available_case_ids))), list(available_case_ids)
    if len(requested_case_ids) != len(set(requested_case_ids)):
        raise ValueError("requested case view contains duplicate case_id values")
    source_slots = {case_id: slot for slot, case_id in enumerate(available_case_ids)}
    missing = [case_id for case_id in requested_case_ids if case_id not in source_slots]
    if missing:
        raise ValueError(
            "requested cases are absent from source cache: " + ", ".join(missing[:3])
        )
    return [source_slots[case_id] for case_id in requested_case_ids], list(requested_case_ids)


class VariableResolutionBrainCropDataset(Dataset):
    """Serve a frozen case view from a 128^3 or 160^3 brain-crop cache."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        expected_resolution: int,
        expected_manifest_sha256: str,
        case_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        index_path = self.cache_dir / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"cache index missing: {index_path}")
        self.index = json.loads(index_path.read_text(encoding="utf-8"))
        if self.index.get("schema") != CACHE_SCHEMA:
            raise ValueError(f"unexpected cache schema: {self.index.get('schema')!r}")
        if self.index.get("test_access") is not False:
            raise ValueError("resolution experiment requires test_access=false")
        if self.index.get("outer_selector_access") is not False:
            raise ValueError("resolution experiment requires outer_selector_access=false")
        if self.index.get("confirmation_access") is not False:
            raise ValueError("resolution experiment requires confirmation_access=false")
        if self.index.get("split") != "historical_train_cohort_only":
            raise ValueError("cache is not restricted to the historical train cohort")
        if self.index.get("manifest_sha256") != expected_manifest_sha256:
            raise ValueError("cache manifest SHA-256 does not match the frozen cohort")

        self.resolution = int(expected_resolution)
        self.model_shape = cubic_shape(self.resolution)
        if tuple(self.index.get("model_shape", ())) != self.model_shape:
            raise ValueError(
                f"cache model_shape {self.index.get('model_shape')} != {self.model_shape}"
            )
        preprocessing = self.index.get("preprocessing", {})
        if preprocessing.get("spatial") != SPATIAL:
            raise ValueError("cache is not image-only braincrop_resize")
        if preprocessing.get("uses_gt_for_localization") is not False:
            raise ValueError("cache must not use GT for localization")
        if int(preprocessing.get("output_resolution", -1)) != self.resolution:
            raise ValueError("preprocessing output_resolution mismatch")

        self._total = int(self.index["count"])
        available = [str(value) for value in self.index["case_ids"]]
        if len(available) != self._total:
            raise ValueError("cache count and case_ids length differ")
        slots, selected = _resolve_case_slots(available, case_ids)
        if limit is not None:
            if int(limit) <= 0:
                raise ValueError("limit must be positive")
            slots = slots[: int(limit)]
            selected = selected[: int(limit)]
        self._slots = slots
        self.case_ids = selected
        self.count = len(slots)
        if self.count == 0:
            raise ValueError("selected cache view is empty")

        source_volumes = self.index.get("per_case_source_positive_voxels")
        if not isinstance(source_volumes, dict):
            raise ValueError("cache lacks resolution-independent source WT volumes")
        if any(int(source_volumes.get(case_id, 0)) <= 0 for case_id in available):
            raise ValueError("cache has missing or nonpositive source WT volumes")

        voxels = int(np.prod(self.model_shape))
        images_path = self.cache_dir / "images.f32"
        labels_path = self.cache_dir / "labels.u8"
        expected_images = self._total * voxels * np.dtype(np.float32).itemsize
        expected_labels = self._total * voxels * np.dtype(np.uint8).itemsize
        if images_path.stat().st_size != expected_images:
            raise ValueError("images.f32 byte size does not match cache index")
        if labels_path.stat().st_size != expected_labels:
            raise ValueError("labels.u8 byte size does not match cache index")
        self._images: np.memmap | None = None
        self._labels: np.memmap | None = None

    def _open(self) -> None:
        if self._images is None:
            self._images = np.memmap(
                self.cache_dir / "images.f32",
                dtype=np.float32,
                mode="r",
                shape=(self._total, *self.model_shape),
            )
            self._labels = np.memmap(
                self.cache_dir / "labels.u8",
                dtype=np.uint8,
                mode="r",
                shape=(self._total, *self.model_shape),
            )

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        self._open()
        assert self._images is not None and self._labels is not None
        source_slot = self._slots[index]
        image = torch.from_numpy(
            np.array(self._images[source_slot], dtype=np.float32)
        )[None]
        label = torch.from_numpy(
            np.array(self._labels[source_slot], dtype=np.uint8).astype(np.float32)
        )[None]
        expected = (1, *self.model_shape)
        if tuple(image.shape) != expected or tuple(label.shape) != expected:
            raise RuntimeError("cached tensor shape does not match declared resolution")
        if not torch.isfinite(image).all() or not bool(label.any()):
            raise RuntimeError("cached case is non-finite or has an empty WT target")
        case_id = self.case_ids[index]
        return {
            "image": image,
            "label": label,
            "case_id": case_id,
            "subject_id": case_id.rsplit("-", 1)[0],
        }

    def source_volume_vector(self) -> np.ndarray:
        """Return native-grid WT volumes; identical for the 128 and 160 arms."""

        values = self.index["per_case_source_positive_voxels"]
        return np.asarray([float(values[case_id]) for case_id in self.case_ids])

    def sampling_weights(self) -> torch.Tensor:
        volumes = self.source_volume_vector()
        median = float(np.median(volumes))
        weights = np.sqrt(median / np.maximum(volumes, 1.0))
        return torch.as_tensor(np.clip(weights, 0.5, 4.0), dtype=torch.double)

    def provenance(self) -> dict:
        return {
            "cache_dir": str(self.cache_dir),
            "manifest": self.index["manifest"],
            "manifest_sha256": self.index["manifest_sha256"],
            "count": self._total,
            "model_shape": list(self.model_shape),
            "images_sha256": self.index["images_sha256"],
            "labels_sha256": self.index["labels_sha256"],
            "source_positive_voxels_sha256": self.index[
                "source_positive_voxels_sha256"
            ],
            "preprocessing": self.index["preprocessing"],
        }
