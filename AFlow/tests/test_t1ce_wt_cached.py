"""Contract tests for the memmap-backed T1ce/WT cached dataset."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sammed3d.t1ce_wt_cached import CACHE_SCHEMA, MODEL_SHAPE, CachedT1ceWtDataset
from sammed3d.t1ce_wt_training import T1ceWtPreprocessConfig

VOXELS = int(np.prod(MODEL_SHAPE))


def make_cache(
    root: Path,
    *,
    count: int = 3,
    manifest_sha: str = "a" * 64,
    schema: str = CACHE_SCHEMA,
    test_access: bool = False,
    truncate_images: bool = False,
    declared_count: int | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    images = rng.standard_normal((count, *MODEL_SHAPE)).astype(np.float32)
    labels = np.zeros((count, *MODEL_SHAPE), dtype=np.uint8)
    # every case must carry a non-empty WT target
    labels[:, 60:68, 60:68, 60:68] = 1
    img_bytes = images.tobytes()
    if truncate_images:
        img_bytes = img_bytes[: len(img_bytes) - 4]
    (root / "images.f32").write_bytes(img_bytes)
    (root / "labels.u8").write_bytes(labels.tobytes())
    (root / "index.json").write_text(json.dumps({
        "schema": schema,
        "schema_version": 1,
        "manifest": "data/fake/train.jsonl",
        "manifest_sha256": manifest_sha,
        "count": count if declared_count is None else declared_count,
        "model_shape": list(MODEL_SHAPE),
        "image_dtype": "float32",
        "label_dtype": "uint8",
        "preprocessing": {"n4_bias_field_correction": True},
        "case_ids": [f"CASE-{i:03d}" for i in range(count if declared_count is None else declared_count)],
        "images_sha256": "b" * 64,
        "labels_sha256": "c" * 64,
        "test_access": test_access,
    }))
    return root


@pytest.fixture()
def config() -> T1ceWtPreprocessConfig:
    return T1ceWtPreprocessConfig(
        spatial_preprocessing="whole_resize",
        train_augmentation="affine_elastic_noise",
    )


def test_validation_split_returns_expected_shapes(tmp_path, config):
    cache = make_cache(tmp_path / "validation")
    ds = CachedT1ceWtDataset(cache, split="validation", config=config)
    assert len(ds) == 3
    item = ds[1]
    assert tuple(item["image"].shape) == (1, *MODEL_SHAPE)
    assert tuple(item["label"].shape) == (1, *MODEL_SHAPE)
    assert item["case_id"] == "CASE-001"
    assert item["target"] == "WT"
    assert torch.isfinite(item["image"]).all()
    assert bool(item["label"].any())


def test_validation_split_is_not_augmented(tmp_path, config):
    cache = make_cache(tmp_path / "validation")
    ds = CachedT1ceWtDataset(cache, split="validation", config=config)
    assert ds.augmentation is None
    # deterministic across repeated reads
    assert torch.equal(ds[0]["image"], ds[0]["image"])


def test_train_split_builds_augmentation(tmp_path, config):
    cache = make_cache(tmp_path / "train")
    ds = CachedT1ceWtDataset(cache, split="train", config=config)
    assert ds.augmentation is not None


def test_cached_values_match_source_bytes(tmp_path, config):
    cache = make_cache(tmp_path / "validation")
    ds = CachedT1ceWtDataset(cache, split="validation", config=config)
    raw = np.memmap(cache / "images.f32", dtype=np.float32, mode="r",
                    shape=(3, *MODEL_SHAPE))
    assert torch.equal(ds[2]["image"][0], torch.from_numpy(np.array(raw[2])))


def test_rejects_unknown_split(tmp_path, config):
    cache = make_cache(tmp_path / "validation")
    with pytest.raises(ValueError, match="split must be"):
        CachedT1ceWtDataset(cache, split="test", config=config)


def test_rejects_missing_index(tmp_path, config):
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="cache index missing"):
        CachedT1ceWtDataset(root, split="train", config=config)


def test_rejects_foreign_schema(tmp_path, config):
    cache = make_cache(tmp_path / "v", schema="something.else")
    with pytest.raises(ValueError, match="unexpected cache schema"):
        CachedT1ceWtDataset(cache, split="train", config=config)


def test_rejects_manifest_sha_mismatch(tmp_path, config):
    cache = make_cache(tmp_path / "v", manifest_sha="a" * 64)
    with pytest.raises(ValueError, match="cache manifest SHA"):
        CachedT1ceWtDataset(
            cache, split="train", config=config, expected_manifest_sha256="d" * 64
        )


def test_rejects_count_mismatch(tmp_path, config):
    cache = make_cache(tmp_path / "v", count=3)
    with pytest.raises(ValueError, match="expected 972"):
        CachedT1ceWtDataset(cache, split="train", config=config, expected_count=972)


def test_rejects_truncated_image_file(tmp_path, config):
    cache = make_cache(tmp_path / "v", truncate_images=True)
    with pytest.raises(ValueError, match="images.f32 size"):
        CachedT1ceWtDataset(cache, split="train", config=config)


def test_rejects_test_access_true(tmp_path, config):
    cache = make_cache(tmp_path / "v", test_access=True)
    with pytest.raises(ValueError, match="test_access=false"):
        CachedT1ceWtDataset(cache, split="train", config=config)


def test_rejects_case_id_count_mismatch(tmp_path, config):
    # declared_count inflates index metadata beyond the real byte payload
    cache = make_cache(tmp_path / "v", count=3, declared_count=4)
    with pytest.raises(ValueError, match="size"):
        CachedT1ceWtDataset(cache, split="train", config=config)


def test_provenance_exposes_binding_fields(tmp_path, config):
    cache = make_cache(tmp_path / "v")
    ds = CachedT1ceWtDataset(cache, split="train", config=config)
    prov = ds.provenance()
    for key in ("cache_dir", "manifest_sha256", "images_sha256",
                "labels_sha256", "count", "preprocessing"):
        assert key in prov
    assert prov["count"] == 3
