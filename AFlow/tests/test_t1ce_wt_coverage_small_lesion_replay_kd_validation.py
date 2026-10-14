from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

import scripts.evaluate_t1ce_wt_coverage_small_lesion_replay_kd_validation as evaluate
import scripts.train_t1ce_wt_coverage_small_lesion_replay_kd as train


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "workspace/BraTS/sammed3d_t1ce_wt/coverage_small_lesion_replay_kd_20260824"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_formal_reports_pass_frozen_contracts() -> None:
    runner_sha = sha256(ROOT / "scripts/train_t1ce_wt_coverage_small_lesion_replay_kd.py")
    for arm in train.ARMS:
        report = WORK / f"formal_{arm}/TRAIN_DONE.json"
        value = evaluate.load_training_report(report, arm, runner_sha)
        assert value["all_gates_passed"] is True
        assert value["sampling_contract"]["strategy"] == evaluate.ARM_STRATEGY[arm]


def test_all_checkpoint_files_and_model_state_hashes_match() -> None:
    for arm in train.ARMS:
        directory = WORK / f"formal_{arm}"
        report = json.loads((directory / "TRAIN_DONE.json").read_text())
        for epoch in evaluate.SAVE_EPOCHS:
            item = report["checkpoints"][str(epoch)]
            path = Path(item["path"])
            assert sha256(path) == item["sha256"]
            value = torch.load(path, map_location="cpu", weights_only=False)
            assert value["schema"] == train.CHECKPOINT_SCHEMA
            assert value["arm"] == arm
            assert value["global_epoch"] == epoch
            assert value["method_documents"] == evaluate.FROZEN_TRAIN_METHOD_DOCUMENTS
            assert value["sampling_contract"]["strategy"] == evaluate.ARM_STRATEGY[arm]
            assert value["model_state_sha256"] == train.core.state_dict_sha256(
                value["model_state_dict"]
            )
            evaluate.require_access_false(value)


def test_candidate_formal_has_all_adjacent_coverage_gates() -> None:
    path = WORK / "formal_coverage_replay_candidate/TRAIN_DONE.json"
    value = json.loads(path.read_text())
    assert len(value["sampling_contract"]["epoch_audits"]) == 32
    assert all(value["sampler_gates"].values())
    assert all(
        row["adjacent_two_epoch_coverage_union"] == 972
        for row in value["sampling_contract"]["epoch_audits"][1:]
    )


def test_checkpoint_sha_maps_are_complete_and_distinct() -> None:
    maps = {}
    for arm in train.ARMS:
        report = json.loads((WORK / f"formal_{arm}/TRAIN_DONE.json").read_text())
        maps[arm] = evaluate.checkpoint_sha_map(report)
        assert set(maps[arm]) == {"176", "184", "192", "200"}
        assert len(set(maps[arm].values())) == 4
    assert set(maps[train.ARMS[0]].values()).isdisjoint(
        set(maps[train.ARMS[1]].values())
    )


def test_fixed_external_evidence_hashes_exist() -> None:
    validation_manifest = ROOT / "data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/validation_t1ce_wt.jsonl"
    validation_cache = ROOT / "workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_ema_20260819/validation108/index.json"
    kd_validation = ROOT / "workspace/BraTS/sammed3d_t1ce_wt/plain_voxel_kd_20260819/KD_VALIDATION.json"
    assert sha256(validation_manifest) == evaluate.VALIDATION_MANIFEST_SHA256
    assert sha256(validation_cache) == evaluate.VALIDATION_CACHE_INDEX_SHA256
    assert sha256(kd_validation) == evaluate.KD_VALIDATION_SHA256
