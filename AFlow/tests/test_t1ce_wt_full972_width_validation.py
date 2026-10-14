from __future__ import annotations

import hashlib
from pathlib import Path

import torch

import scripts.evaluate_t1ce_wt_full972_width_validation as evaluate
import scripts.train_t1ce_wt_full972_width_final as train


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "workspace/BraTS/sammed3d_t1ce_wt/full972_width24_20260826"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_final_reports_and_checkpoints_are_valid() -> None:
    runner_sha = sha256(ROOT / "scripts/train_t1ce_wt_full972_width_final.py")
    for arm in evaluate.ARMS:
        directory = WORK / f"final_formal_width{train.WIDTH[arm]}"
        report = evaluate.load_training_report(directory / "TRAIN_DONE.json", arm, runner_sha)
        path = directory / evaluate.CHECKPOINT_NAMES[arm]
        value = torch.load(path, map_location="cpu", weights_only=False)
        assert sha256(path) == report["checkpoint"]["sha256"]
        assert value["model_state_sha256"] == train.state_dict_sha256(value["model_state_dict"])
        assert value["architecture"]["init_filters"] == train.WIDTH[arm]
        evaluate.require_access_false(value)


def test_fixed_evidence_hashes() -> None:
    assert sha256(ROOT / "data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/validation_t1ce_wt.jsonl") == evaluate.VALIDATION_MANIFEST_SHA256
    assert sha256(ROOT / "workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_ema_20260819/validation108/index.json") == evaluate.VALIDATION_CACHE_INDEX_SHA256
    assert sha256(ROOT / "workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824/FULLTRAJECTORY_VALIDATION.json") == evaluate.ARCHIVED_SEED32_VALIDATION_SHA256
