from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

import scripts.evaluate_t1ce_wt_voxel_kd_972_draw_budget_validation as evaluate
import scripts.train_t1ce_wt_voxel_kd_972_draw_budget as train


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_972_draw_budget_20260824"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_formal_reports_match_frozen_arm_budgets() -> None:
    runner_sha = sha256(ROOT / "scripts/train_t1ce_wt_voxel_kd_972_draw_budget.py")
    for arm in train.ARMS:
        value = evaluate.load_training_report(
            WORK / f"formal_{arm}/TRAIN_DONE.json", arm, runner_sha
        )
        draws, consumed, steps, _ = evaluate.ARM_BUDGET[arm]
        sampling = value["sampling_contract"]
        assert sampling["sampler_num_samples"] == draws
        assert sampling["consumed_draws_per_epoch"] == consumed
        assert sampling["optimizer_steps_per_epoch"] == steps


def test_all_checkpoint_contracts_and_hashes_match() -> None:
    for arm in train.ARMS:
        directory = WORK / f"formal_{arm}"
        report = json.loads((directory / "TRAIN_DONE.json").read_text())
        draws, consumed, steps, _ = evaluate.ARM_BUDGET[arm]
        for epoch in evaluate.SAVE_EPOCHS:
            item = report["checkpoints"][str(epoch)]
            path = Path(item["path"])
            assert sha256(path) == item["sha256"]
            value = torch.load(path, map_location="cpu", weights_only=False)
            assert value["method_documents"] == evaluate.FROZEN_TRAIN_METHOD_DOCUMENTS
            assert value["sampling_contract"]["sampler_num_samples"] == draws
            assert value["sampling_contract"]["consumed_draws_per_epoch"] == consumed
            assert value["sampling_contract"]["optimizer_steps_per_epoch"] == steps
            assert value["model_state_sha256"] == train.state_dict_sha256(value["model_state_dict"])
            evaluate.require_access_false(value)


def test_checkpoint_sha_maps_are_complete() -> None:
    for arm in train.ARMS:
        report = json.loads((WORK / f"formal_{arm}/TRAIN_DONE.json").read_text())
        values = evaluate.checkpoint_sha_map(report)
        assert set(values) == {"176", "184", "192", "200"}
        assert len(set(values.values())) == 4


def test_fixed_validation_and_kd_hashes() -> None:
    manifest = ROOT / "data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/validation_t1ce_wt.jsonl"
    cache = ROOT / "workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_ema_20260819/validation108/index.json"
    kd = ROOT / "workspace/BraTS/sammed3d_t1ce_wt/plain_voxel_kd_20260819/KD_VALIDATION.json"
    assert sha256(manifest) == evaluate.VALIDATION_MANIFEST_SHA256
    assert sha256(cache) == evaluate.VALIDATION_CACHE_INDEX_SHA256
    assert sha256(kd) == evaluate.KD_VALIDATION_SHA256
