from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scripts.build_t1ce_wt_percentile_clip_cache as cache
import scripts.build_t1ce_wt_percentile_clip_replication_splits as splits
import scripts.evaluate_t1ce_wt_percentile_clip_replication_inner as evaluate
import scripts.train_t1ce_wt_percentile_clip_replication as train


def test_percentile_clip_is_fixed_linear_and_preserves_background() -> None:
    image = np.zeros((3, 3, 3), dtype=np.float32)
    image.flat[1:] = np.arange(1, 27, dtype=np.float32)
    clipped, stats = cache.percentile_clip_nonzero(image)
    expected = np.percentile(np.arange(1, 27, dtype=np.float64), [1, 99], method="linear")
    assert stats["low"] == expected[0]
    assert stats["high"] == expected[1]
    assert clipped.flat[0] == 0
    assert clipped.flat[1] == pytest.approx(expected[0])
    assert clipped.flat[-1] == pytest.approx(expected[1])
    assert np.isfinite(clipped).all()


def test_percentile_clip_rejects_nonfinite_constant_or_empty() -> None:
    with pytest.raises(ValueError):
        cache.percentile_clip_nonzero(np.full((2, 2, 2), np.nan, dtype=np.float32))
    with pytest.raises(ValueError):
        cache.percentile_clip_nonzero(np.ones((2, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        cache.percentile_clip_nonzero(np.zeros((2, 2, 2), dtype=np.float32))


def test_frozen_seeds_epoch_and_percentiles() -> None:
    assert splits.SPLIT_SEEDS == {"a": 20261111, "b": 20261113}
    assert splits.TRAIN_SEEDS == train.REPLICATES == {"a": 20261112, "b": 20261114}
    assert splits.BOOTSTRAP_SEEDS == {"a": 20261115, "b": 20261116}
    assert train.FORMAL_EPOCHS == 168
    assert (cache.LOW_PERCENTILE, cache.HIGH_PERCENTILE) == (1.0, 99.0)


def test_forbidden_role_paths_rejected(tmp_path: Path) -> None:
    for token in ("validation", "test270", "external75", "confirmation", "sealed"):
        with pytest.raises(ValueError):
            train.assert_train_only_path(tmp_path / token)
        with pytest.raises(ValueError):
            cache.assert_train_only_path(tmp_path / token)


def test_candidate_cache_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    index = {
        "count": 972,
        "split": "train",
        "test_access": False,
        "manifest_sha256": train.SOURCE_SHA256,
        "labels_sha256": train.CONTROL_LABELS_SHA256,
        "images_sha256": "different",
        "labels_byte_exact_to_control": True,
        "preprocessing": {
            "spatial": "braincrop_resize",
            "uses_gt_for_localization": False,
            "low_percentile": 1.0,
            "high_percentile": 99.0,
            "percentile_method": "numpy.percentile linear",
        },
    }
    (candidate / "index.json").write_text(json.dumps(index))
    monkeypatch.setattr(train.base, "sha256_file", lambda _: "a" * 64)
    assert train._cache_contract(candidate, "percentile_clip", "a" * 64) == index
    index["preprocessing"]["low_percentile"] = 2.0
    (candidate / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError):
        train._cache_contract(candidate, "percentile_clip", "a" * 64)


def _report(arm: str) -> dict:
    return {
        "replicate": "a",
        "initial_state_sha256": "initial",
        "sampling_weight_sha256": "weights",
        "first_batch_sha256": {"image": arm, "label": "label"},
        "data_roles": {"input_cache": {"labels_sha256": train.CONTROL_LABELS_SHA256, "images_sha256": arm}},
        "history": [
            {
                "epoch": 1,
                "sampler_index_sha256": "sampler",
                "augmentation_rng_before_sha256": "before",
                "augmentation_rng_after_sha256": "after",
            }
        ],
    }


def test_pairing_allows_only_intentional_image_difference() -> None:
    result = evaluate.verify_pair(_report("control"), _report("candidate"))
    assert result["cache_labels_exact"] is True
    assert result["cache_images_intentionally_different"] is True


def test_pairing_rejects_label_or_rng_mismatch() -> None:
    control = _report("control")
    candidate = _report("candidate")
    candidate["data_roles"]["input_cache"]["labels_sha256"] = "wrong"
    with pytest.raises(ValueError):
        evaluate.verify_pair(control, candidate)
    candidate = _report("candidate")
    candidate["history"][0]["augmentation_rng_after_sha256"] = "wrong"
    with pytest.raises(ValueError):
        evaluate.verify_pair(control, candidate)


def test_scientific_gates_direction_and_ci_floor() -> None:
    passing = {
        "candidate_minus_control": {"scan_dice": 0.001, "subject_dice": 0.001, "point_hits": -1},
        "scan_dice_95ci": [-0.003, 0.004],
        "subject_dice_95ci": [-0.002, 0.004],
    }
    assert evaluate.replication_gates(passing)["all_passed"] is True
    passing["candidate_minus_control"]["scan_dice"] = 0.0
    assert evaluate.replication_gates(passing)["all_passed"] is False


def test_compare_is_case_and_subject_paired() -> None:
    control = {"per_case": [
        {"case_id": "A-100", "subject_id": "A", "locator_dsc": 0.4, "automatic_point_hits_gt": True},
        {"case_id": "A-101", "subject_id": "A", "locator_dsc": 0.6, "automatic_point_hits_gt": False},
        {"case_id": "B-100", "subject_id": "B", "locator_dsc": 0.5, "automatic_point_hits_gt": True},
    ]}
    candidate = {"per_case": [
        {"case_id": "A-100", "subject_id": "A", "locator_dsc": 0.5, "automatic_point_hits_gt": True},
        {"case_id": "A-101", "subject_id": "A", "locator_dsc": 0.7, "automatic_point_hits_gt": True},
        {"case_id": "B-100", "subject_id": "B", "locator_dsc": 0.5, "automatic_point_hits_gt": True},
    ]}
    result = evaluate.compare(control, candidate, 7)
    assert result["candidate_minus_control"]["scan_dice"] == pytest.approx(2 / 30)
    assert result["candidate_minus_control"]["subject_dice"] == pytest.approx(0.05)
    assert result["candidate_minus_control"]["point_hits"] == 1


def test_training_source_has_no_inner_loader_or_epoch_selection() -> None:
    source = Path(train.__file__).read_text()
    assert "inner_dataset_constructed\": False" in source
    assert "FORMAL_EPOCHS = 168" in source
    assert "best_checkpoint" not in source
    assert "inner_tune" not in source


def test_evaluator_writes_started_before_inner_dataset() -> None:
    source = Path(evaluate.__file__).read_text()
    main_source = source[source.index("def main() -> None:") :]
    assert main_source.index("write_json_atomic(started_output") < main_source.index("for replicate in (\"a\", \"b\"):")
    assert '"fixed_validation108": False' in main_source
    assert '"test270": False' in main_source
    assert '"ucsd_external75": False' in main_source
