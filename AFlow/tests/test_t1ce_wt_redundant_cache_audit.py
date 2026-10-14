from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import scripts.audit_t1ce_wt_redundant_cache as audit


def _cache(path: Path, case_ids: list[str], values: list[float], split: str) -> str:
    path.mkdir()
    images = np.memmap(path / "images.f32", dtype=np.float32, mode="w+", shape=(len(case_ids), 2, 2, 2))
    labels = np.memmap(path / "labels.u8", dtype=np.uint8, mode="w+", shape=(len(case_ids), 2, 2, 2))
    for slot, value in enumerate(values):
        images[slot] = value
        labels[slot] = int(value) % 2
    images.flush(); labels.flush()
    index = {"split": split, "test_access": False, "count": len(case_ids), "case_ids": case_ids, "model_shape": [2, 2, 2], "preprocessing": {"spatial": "braincrop_resize", "uses_gt_for_localization": False, "n4_bias_field_correction": True, "n4_shrink_factor": 2}}
    (path / "index.json").write_text(json.dumps(index))
    return audit.sha256_file(path / "index.json")


def test_reordered_subset_is_proven_byte_exact_without_deletion(tmp_path: Path) -> None:
    superset = tmp_path / "train_cache"
    subset = tmp_path / "historical_cache"
    superset_sha = _cache(superset, ["a", "b", "c"], [1, 2, 3], "train")
    subset_sha = _cache(subset, ["c", "a"], [3, 1], "historical_train_cohort_only")
    args = argparse.Namespace(superset=superset, expected_superset_index_sha256=superset_sha, subset=subset, expected_subset_index_sha256=subset_sha)
    result = audit.audit(args)
    assert result["same_case_order"] is False
    assert result["image_byte_exact_cases"] == 2
    assert result["label_byte_exact_cases"] == 2
    assert result["all_tensors_byte_exact"] is True
    assert result["deletion_performed"] is False
