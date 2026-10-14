from copy import deepcopy
from pathlib import Path

import pytest
import torch

import scripts.build_t1ce_wt_flip_equivariance_consistency_splits as splits
import scripts.evaluate_t1ce_wt_flip_equivariance_consistency_inner as evaluate
import scripts.train_t1ce_wt_flip_equivariance_consistency as train


ROOT = Path(__file__).resolve().parents[1]


def _summary(rows: list[tuple[str, str, float, bool]]) -> dict:
    return {
        "per_case": [
            {
                "case_id": case,
                "subject_id": subject,
                "locator_dsc": dice,
                "automatic_point": [0, 0, 0],
                "automatic_point_hits_gt": hit,
            }
            for case, subject, dice, hit in rows
        ]
    }


def _history() -> list[dict]:
    return [
        {
            "sampler_index_sha256": f"sampler{i}",
            "augmentation_rng_before_sha256": f"before{i}",
            "augmentation_rng_after_sha256": f"after{i}",
            "reflection_schedule_sha256": f"reflection{i}",
            "student_forwards_per_step": 2,
            "train_consistency_loss": 0.1,
        }
        for i in range(168)
    ]


def _report(arm: str) -> dict:
    return {
        "initial_state_sha256": "init",
        "sampling_weight_sha256": "weights",
        "first_batch_sha256": {"image": "image", "label": "label"},
        "first_step": {"axes": [-1], "existing_loss": 1.25, "consistency_loss": 0.2},
        "history": _history(),
        "all_gates_passed": True,
        "arm": arm,
    }


def test_squared_soft_dice_is_zero_for_exact_equivariance_and_has_gradients() -> None:
    original = torch.randn(2, 1, 4, 5, 6, requires_grad=True)
    axes = (-3, -1)
    exact_reflected = torch.flip(original, dims=axes)
    exact = train.sigmoid_soft_dice_consistency(original, exact_reflected, axes)
    assert float(exact) == 0.0
    perturbed = (exact_reflected.detach() + 0.5).requires_grad_()
    loss = train.sigmoid_soft_dice_consistency(original, perturbed, axes)
    assert float(loss) > 0
    loss.backward()
    assert original.grad is not None and torch.isfinite(original.grad).all()
    assert perturbed.grad is not None and torch.isfinite(perturbed.grad).all()


def test_reflection_cycle_is_complete_fixed_and_repeats() -> None:
    assert train.NONIDENTITY_AXES == ((-1,), (-2,), (-2, -1), (-3,), (-3, -1), (-3, -2), (-3, -2, -1))
    assert [train.reflection_axes(1, step) for step in range(14)] == list(train.NONIDENTITY_AXES) * 2
    assert train.reflection_axes(2, 0) == train.NONIDENTITY_AXES[121 % 7]


def test_single_factor_constants_and_document_binding() -> None:
    assert train.ARMS == ("control_w0", "candidate_w01")
    assert train.CONSISTENCY_WEIGHT == {"control_w0": 0.0, "candidate_w01": 0.1}
    assert train.FORMAL_EPOCHS == 168
    assert train.STEPS_PER_EPOCH == 121
    assert train.CONSUMED_DRAWS == 968
    assert train.AMENDMENT_SHA256 == "80604f241a6a220bee667019721cd76c5785f88a56f8ae17cc6fe99f9ea84ad4"


def test_fresh_subject_selection_excludes_old_and_other_replication() -> None:
    counts = {f"S{index:03d}": 1 for index in range(260)}
    old = {f"S{index:03d}" for index in range(20)}
    c = splits.choose_fresh_subjects(counts, excluded=old, seed=splits.SPLIT_SEEDS["c"])
    d = splits.choose_fresh_subjects(counts, excluded=old | c, seed=splits.SPLIT_SEEDS["d"])
    assert len(c) == len(d) == 108
    assert not (c & d)
    assert not ((c | d) & old)


def test_pairing_gates_cover_every_epoch_and_first_forward() -> None:
    pair = {"control_w0": _report("control_w0"), "candidate_w01": _report("candidate_w01")}
    gates = evaluate.pairing_gates(pair)
    assert gates["all_passed"] is True
    broken = deepcopy(pair)
    broken["candidate_w01"]["history"][83]["reflection_schedule_sha256"] = "wrong"
    gates = evaluate.pairing_gates(broken)
    assert gates["all_reflection_schedules_exact"] is False
    assert gates["all_passed"] is False


def test_comparison_and_gate_directions() -> None:
    control = _summary([("A1", "A", 0.5, True), ("A2", "A", 0.6, True), ("B1", "B", 0.7, False)])
    candidate = _summary([("A1", "A", 0.6, True), ("A2", "A", 0.7, False), ("B1", "B", 0.8, True)])
    result = evaluate.compare(control, candidate, 7)
    assert result["candidate_minus_control"]["scan_dice"] == pytest.approx(0.1)
    assert result["candidate_minus_control"]["subject_dice"] == pytest.approx(0.1)
    assert result["candidate_minus_control"]["point_hits"] == 0
    assert all(evaluate.primary_gates(result).values())
    assert all(evaluate.identity_gates(result).values())
    tied = deepcopy(result)
    tied["candidate_minus_control"]["scan_dice"] = 0.0
    assert evaluate.primary_gates(tied)["scan_strictly_positive"] is False
    assert evaluate.identity_gates(tied)["identity_scan_strictly_positive"] is False


def test_mechanism_gate_uses_only_last_five_epochs() -> None:
    pair = {"control_w0": _report("control_w0"), "candidate_w01": _report("candidate_w01")}
    for row in pair["candidate_w01"]["history"][-5:]:
        row["train_consistency_loss"] = 0.089
    result = evaluate.mechanism(pair)
    assert result["epochs"] == [164, 165, 166, 167, 168]
    assert result["candidate_over_control"] == pytest.approx(0.89)
    assert result["at_most_0p90"] is True


def test_sources_forbid_revealed_roles_and_tuning_controls() -> None:
    trainer = (ROOT / "scripts/train_t1ce_wt_flip_equivariance_consistency.py").read_text()
    evaluator = (ROOT / "scripts/evaluate_t1ce_wt_flip_equivariance_consistency_inner.py").read_text()
    freezer = (ROOT / "scripts/freeze_t1ce_wt_flip_equivariance_consistency_inner_preaccess.py").read_text()
    assert 'total = existing + consistency_weight * consistency' in trainer
    assert 'reflected_logits = model(reflected_image).segmentation_logits' in trainer
    assert '"student_forwards_per_step": 2' in trainer
    assert '--consistency-weight' not in trainer
    assert '--inner' not in trainer
    assert 'fixed_epoch168.pth' in trainer
    for token in ("validation", "test270", "external75", "confirmation", "sealed"):
        assert token in train.FORBIDDEN_TOKENS
    assert 'frozen_after_four_trainings_before_inner_access' in freezer
    assert 'existing_validation_test_or_external_access_authorized": False' in evaluator
    assert 'result_must_not_trigger_weight_loss_view_epoch_threshold_or_seed_tuning' in evaluator
