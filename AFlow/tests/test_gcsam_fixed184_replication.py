from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.build_t1ce_wt_gcsam_fixed184_replication_splits as split
import scripts.train_t1ce_wt_gcsam_fixed184_replication as train
import scripts.evaluate_t1ce_wt_gcsam_fixed184_replication_inner as evaluate


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "workspace/BraTS/sammed3d_t1ce_wt/gcsam_fixed184_replication_20260903"


def test_frozen_replication_seeds_are_distinct() -> None:
    assert split.SPLIT_SEEDS == {"a": 20261101, "b": 20261103}
    assert split.TRAIN_SEEDS == {"a": 20261102, "b": 20261104}
    assert len(set(split.SPLIT_SEEDS.values()) | set(split.TRAIN_SEEDS.values())) == 4


def test_source_is_exact_full972_train_role() -> None:
    path = ROOT / "data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl"
    assert split.sha256_file(path) == split.SOURCE_SHA256
    rows = split.read_rows(path)
    assert len(rows) == 972
    assert len({row["subject_id"] for row in rows}) == 444


def test_role_writer_has_no_paths_or_labels(tmp_path: Path) -> None:
    path = tmp_path / "role.jsonl"
    split.write_rows(path, [{"case_id": "BraTS-GLI-00001-100", "subject_id": "BraTS-GLI-00001"}], "gcsam_replication_fit", "a")
    row = json.loads(path.read_text())
    assert set(row) == {"case_id", "subject_id", "role", "replicate"}
    assert row["role"] == "gcsam_replication_fit"
    assert row["replicate"] == "a"


def test_source_reader_rejects_wrong_count(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"case_id": "x", "subject_id": "s", "split": "train"}) + "\n")
    with pytest.raises(ValueError, match="972"):
        split.read_rows(path)


def test_training_contract_is_fixed_epoch184_and_two_replicates() -> None:
    assert train.FORMAL_EPOCHS == 184
    assert train.RHO == 0.05
    assert train.REPLICATES["a"]["seed"] == 20261102
    assert train.REPLICATES["b"]["seed"] == 20261104
    assert train.STEPS_PER_EPOCH == 121
    assert train.CONSUMED_DRAWS == 968


def test_training_path_guard_rejects_evaluation_roles(tmp_path: Path) -> None:
    for name in ("validation", "test270", "external75", "confirmation", "sealed"):
        with pytest.raises(ValueError, match="forbidden"):
            train.assert_train_only_path(tmp_path / name)


def test_training_runner_has_no_inner_manifest_cli() -> None:
    source = (ROOT / "scripts/train_t1ce_wt_gcsam_fixed184_replication.py").read_text()
    assert "--inner" not in source
    assert '"inner_dataset_constructed": False' in source
    assert '"inner_metrics": None' in source


def test_gpu_smoke_pair_is_exact_and_inner_sealed() -> None:
    control_path = WORK / "smoke_a_adamw_control/TRAIN_DONE.json"
    candidate_path = WORK / "smoke_a_gcsam_candidate/TRAIN_DONE.json"
    assert train.base.sha256_file(control_path) == "2a3878a0337062c7f0a23afcc581f98d3a35dc7ccda24bd81c67cbf15fa351e1"
    assert train.base.sha256_file(candidate_path) == "4dc5178e14c123b6b0eb3ffd574de830f2a354b4f51f0ad1c5419897b4872d88"
    control = json.loads(control_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    assert control["all_gates_passed"] is True
    assert candidate["all_gates_passed"] is True
    assert control["initial_state_sha256"] == candidate["initial_state_sha256"]
    assert control["sampling_weight_sha256"] == candidate["sampling_weight_sha256"]
    assert control["first_batch_sha256"] == candidate["first_batch_sha256"]
    assert control["first_batch_loss"] == candidate["first_batch_loss"]
    assert control["history"][0]["sampler_index_sha256"] == candidate["history"][0]["sampler_index_sha256"]
    assert candidate["history"][0]["mean_perturbation_norm"] == pytest.approx(0.05, abs=2e-6)
    assert control["access"]["inner_metrics"] is False
    assert candidate["access"]["inner_metrics"] is False


def test_evaluator_accepts_smoke_pair_as_contract_equivalent() -> None:
    control = evaluate.load_training_report(WORK / "smoke_a_adamw_control/TRAIN_DONE.json", "a", "adamw_control", expected_epoch=1)
    candidate = evaluate.load_training_report(WORK / "smoke_a_gcsam_candidate/TRAIN_DONE.json", "a", "gcsam_candidate", expected_epoch=1)
    pairing = evaluate.verify_pair(control, candidate)
    assert pairing["epochs_compared"] == 1
    assert all(value is True for key, value in pairing.items() if key != "epochs_compared")


def _fake_summary(values: list[float], subjects: list[str], points: list[bool]) -> dict:
    return {"per_case": [{"case_id": f"case-{index}", "subject_id": subject, "locator_dsc": value, "automatic_point_hits_gt": point} for index, (value, subject, point) in enumerate(zip(values, subjects, points))]}


def test_replication_comparison_and_gates_are_directional() -> None:
    control = _fake_summary([0.5, 0.7, 0.6], ["s1", "s1", "s2"], [True, False, True])
    candidate = _fake_summary([0.51, 0.71, 0.61], ["s1", "s1", "s2"], [True, True, True])
    comparison = evaluate.compare(control, candidate, seed=7)
    assert comparison["candidate_minus_control"]["scan_dice"] == pytest.approx(0.01)
    assert comparison["candidate_minus_control"]["subject_dice"] == pytest.approx(0.01)
    assert comparison["candidate_minus_control"]["point_hits"] == 1
    assert evaluate.replication_gates(comparison)["all_passed"] is True
    comparison["scan_dice_95ci"][0] = -0.003001
    assert evaluate.replication_gates(comparison)["all_passed"] is False
