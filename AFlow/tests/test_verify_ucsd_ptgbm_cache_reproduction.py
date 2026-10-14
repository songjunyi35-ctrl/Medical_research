from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.verify_ucsd_ptgbm_cache_reproduction import compare_tensors


def _write(path: Path, images: np.ndarray, labels: np.ndarray) -> None:
    path.mkdir()
    image_map = np.memmap(path / "images.f32", dtype=np.float32, mode="w+", shape=images.shape)
    label_map = np.memmap(path / "labels.u8", dtype=np.uint8, mode="w+", shape=labels.shape)
    image_map[:] = images; label_map[:] = labels
    image_map.flush(); label_map.flush()


def test_compare_tensors_detects_exact_and_mismatch(tmp_path: Path) -> None:
    images = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    labels = (images % 2).astype(np.uint8)
    left = tmp_path / "left"; right = tmp_path / "right"
    _write(left, images, labels); _write(right, images, labels)
    exact = compare_tensors(left, right, 2, (2, 2, 2))
    assert exact["all_tensors_byte_exact"] is True
    changed = np.memmap(right / "images.f32", dtype=np.float32, mode="r+", shape=images.shape)
    changed[1, 0, 0, 0] += 0.5; changed.flush()
    mismatch = compare_tensors(left, right, 2, (2, 2, 2))
    assert mismatch["all_tensors_byte_exact"] is False
    assert mismatch["images.f32"]["byte_exact_cases"] == 1
    assert mismatch["images.f32"]["mismatch_slots"] == [1]
    assert mismatch["images.f32"]["max_abs_delta"] == 0.5
