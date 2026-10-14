from __future__ import annotations
import hashlib
from pathlib import Path
import torch
import scripts.evaluate_t1ce_wt_voxel_kd_fulltrajectory_validation as evaluate
import scripts.train_t1ce_wt_voxel_kd_fulltrajectory_final as train
ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824"
def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def test_final_reports_and_checkpoints_are_valid() -> None:
    runner_sha = sha256(ROOT / "scripts/train_t1ce_wt_voxel_kd_fulltrajectory_final.py")
    initial = set()
    for arm in evaluate.ARMS:
        directory = WORK / f"final_formal_{arm}"
        report = evaluate.load_training_report(directory / "TRAIN_DONE.json", arm, runner_sha)
        initial.add(report["initial_state_sha256"])
        path = directory / evaluate.CHECKPOINT_NAMES[arm]
        value = torch.load(path, map_location="cpu", weights_only=False)
        assert sha256(path) == report["checkpoint"]["sha256"]
        assert value["model_state_sha256"] == train.state_dict_sha256(value["model_state_dict"])
        evaluate.require_access_false(value)
    assert len(initial) == 1
def test_fixed_evidence_hashes() -> None:
    assert sha256(ROOT / "data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/validation_t1ce_wt.jsonl") == evaluate.VALIDATION_MANIFEST_SHA256
    assert sha256(ROOT / "workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_ema_20260819/validation108/index.json") == evaluate.VALIDATION_CACHE_INDEX_SHA256
    assert sha256(ROOT / "workspace/BraTS/sammed3d_t1ce_wt/plain_voxel_kd_20260819/KD_VALIDATION.json") == evaluate.KD_VALIDATION_SHA256
