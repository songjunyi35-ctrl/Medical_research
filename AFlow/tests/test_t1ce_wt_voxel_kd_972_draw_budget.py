from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import WeightedRandomSampler

from scripts.build_t1ce_wt_native_volume_artifact import native_sampling_weights
import scripts.train_t1ce_wt_voxel_kd_972_draw_budget as budget


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "archive/research_records/2026-08-24_voxel_kd_972_draw_budget"
PRIOR = ROOT / "archive/research_records/2026-08-21_voxel_kd_data972_native_sampling"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_ids() -> list[str]:
    path = ROOT / "data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl"
    return [json.loads(line)["case_id"] for line in path.read_text().splitlines() if line]


def test_frozen_documents_and_native_artifact() -> None:
    assert sha256(RECORD / "IDEA.md") == budget.IDEA_SHA256
    assert sha256(RECORD / "PREREG.md") == budget.PREREG_SHA256
    assert sha256(PRIOR / "AMENDMENT_003_FREEZE_NATIVE_ARTIFACT.md") == budget.AMENDMENT_SHA256
    assert sha256(PRIOR / "NATIVE_WT_VOLUMES.json") == budget.NATIVE_ARTIFACT_SHA256


def test_both_arms_use_same_972_population_and_manifest() -> None:
    assert budget.ARM_COUNTS == {
        "budget752_control": (972, 444),
        "budget968_candidate": (972, 444),
    }
    assert len(set(budget.ARM_MANIFEST_SHA256.values())) == 1
    assert next(iter(budget.ARM_MANIFEST_SHA256.values())) == sha256(
        ROOT / "data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl"
    )


def test_native_weights_are_exact() -> None:
    artifact = json.loads((PRIOR / "NATIVE_WT_VOLUMES.json").read_text())
    weights = native_sampling_weights(case_ids(), artifact["native_wt_voxels"])
    assert budget.tensor_sha256(weights) == budget.NATIVE_WEIGHT_SHA256
    assert weights.numel() == 972
    assert float(weights.min()) == 0.5
    assert float(weights.max()) == 4.0


def test_arm_budget_is_the_only_numeric_contrast() -> None:
    assert budget.ARM_SAMPLER_DRAWS == {
        "budget752_control": 756,
        "budget968_candidate": 972,
    }
    assert budget.ARM_CONSUMED_DRAWS == {
        "budget752_control": 752,
        "budget968_candidate": 968,
    }
    assert budget.ARM_STEPS_PER_EPOCH == {
        "budget752_control": 94,
        "budget968_candidate": 121,
    }
    assert 968 / 752 == 121 / 94


def test_candidate_sampler_extends_control_sequence_for_first_epoch() -> None:
    weights = torch.linspace(0.5, 4.0, 972, dtype=torch.float64)
    control = list(WeightedRandomSampler(
        weights, 756, replacement=True,
        generator=torch.Generator().manual_seed(20260924),
    ))
    candidate = list(WeightedRandomSampler(
        weights, 972, replacement=True,
        generator=torch.Generator().manual_seed(20260924),
    ))
    assert candidate[:756] == control
    assert len(control[:752]) == 752
    assert len(candidate[:968]) == 968


def test_checkpoint_grid_and_cost_limits_are_frozen() -> None:
    assert tuple(budget.SAVE_GLOBAL_EPOCHS) == (176, 184, 192, 200)
    assert budget.ARM_MAX_FORMAL_SECONDS == {
        "budget752_control": 2700,
        "budget968_candidate": 3600,
    }
    assert budget.MAX_PEAK_BYTES == 24 * 1024**3
