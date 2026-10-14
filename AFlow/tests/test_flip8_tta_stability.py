from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.build_t1ce_wt_flip8_tta_replication_splits as splits
import scripts.evaluate_t1ce_wt_flip8_tta_replication_inner as evaluate
import scripts.train_t1ce_wt_flip8_tta_replication as train


EXPECTED_AXES = (
    (), (-1,), (-2,), (-2, -1), (-3,), (-3, -1), (-3, -2), (-3, -2, -1)
)


def test_reflection_group_exact_order_and_unique() -> None:
    assert evaluate.REFLECTION_AXES == EXPECTED_AXES
    assert len(set(evaluate.REFLECTION_AXES)) == 8
    assert {tuple(sorted(abs(axis) for axis in axes)) for axes in evaluate.REFLECTION_AXES} == {
        (), (1,), (2,), (1, 2), (3,), (1, 3), (2, 3), (1, 2, 3)
    }


@pytest.mark.parametrize("axes", EXPECTED_AXES)
def test_reflection_is_its_own_inverse(axes: tuple[int, ...]) -> None:
    tensor = torch.arange(2 * 3 * 4).reshape(1, 1, 2, 3, 4)
    transformed = torch.flip(tensor, dims=axes) if axes else tensor
    restored = torch.flip(transformed, dims=axes) if axes else transformed
    assert torch.equal(restored, tensor)


def test_flip8_reuses_identity_and_averages_float32(monkeypatch: pytest.MonkeyPatch) -> None:
    class IdentityModel:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, image: torch.Tensor) -> SimpleNamespace:
            self.calls += 1
            return SimpleNamespace(segmentation_logits=image.to(torch.bfloat16))

    monkeypatch.setattr(evaluate.torch, "autocast", lambda **_: nullcontext())
    model = IdentityModel()
    image = torch.randn(1, 1, 2, 3, 4)
    identity, averaged = evaluate.flip8_logits(model, image)
    expected = image.to(torch.bfloat16).float()
    assert model.calls == 8
    assert identity.dtype == averaged.dtype == torch.float32
    assert torch.equal(identity, expected)
    assert torch.equal(averaged, expected)


def test_case_row_uses_logit_zero_and_argmax_point() -> None:
    logits = torch.full((1, 1, 2, 2, 2), -1.0)
    label = torch.zeros_like(logits)
    logits[0, 0, 1, 0, 1] = 2.0
    label[0, 0, 1, 0, 1] = 1.0
    row = evaluate._case_row(logits, label, "case", "subject")
    assert row["locator_dsc"] == 1.0
    assert row["automatic_point"] == [1, 0, 1]
    assert row["automatic_point_hits_gt"] is True


def _summary(values: list[tuple[str, str, float, bool]]) -> dict:
    return {"per_case": [
        {"case_id": case, "subject_id": subject, "locator_dsc": dice, "automatic_point_hits_gt": hit}
        for case, subject, dice, hit in values
    ]}


def test_compare_is_case_and_subject_paired() -> None:
    identity = _summary([("A1", "A", 0.4, True), ("A2", "A", 0.6, False), ("B1", "B", 0.5, True)])
    flip8 = _summary([("A1", "A", 0.5, True), ("A2", "A", 0.7, True), ("B1", "B", 0.5, True)])
    result = evaluate.compare(identity, flip8, 7)
    assert result["flip8_minus_identity"]["scan_dice"] == pytest.approx(2 / 30)
    assert result["flip8_minus_identity"]["subject_dice"] == pytest.approx(0.05)
    assert result["flip8_minus_identity"]["point_hits"] == 1


def test_scientific_gate_directions() -> None:
    passing = {
        "flip8_minus_identity": {"scan_dice": 0.001, "subject_dice": 0.001, "point_hits": -1},
        "scan_dice_95ci": [-0.003, 0.004],
        "subject_dice_95ci": [-0.002, 0.004],
    }
    assert evaluate.replication_gates(passing)["all_passed"] is True
    passing["flip8_minus_identity"]["scan_dice"] = 0.0
    assert evaluate.replication_gates(passing)["all_passed"] is False


def test_frozen_seeds_epoch_and_margin() -> None:
    assert splits.SPLIT_SEEDS == {"a": 20261201, "b": 20261203}
    assert splits.TRAIN_SEEDS == train.REPLICATES == {"a": 20261202, "b": 20261204}
    assert splits.BOOTSTRAP_SEEDS == {"a": 20261205, "b": 20261206}
    assert train.FORMAL_EPOCHS == 168
    assert evaluate.LEADER_MARGIN == 0.000243638


def test_forbidden_role_paths_rejected(tmp_path: Path) -> None:
    for token in ("validation", "test270", "external75", "confirmation", "sealed"):
        with pytest.raises(ValueError):
            train.assert_train_only_path(tmp_path / token)


def test_training_source_has_no_inner_loader_or_selection() -> None:
    source = Path(train.__file__).read_text()
    assert '"inner_dataset_constructed": False' in source
    assert "FORMAL_EPOCHS = 168" in source
    assert "best_checkpoint" not in source
    assert "inner_tune" not in source


def test_evaluator_started_receipt_precedes_inner_construction() -> None:
    source = Path(evaluate.__file__).read_text()
    main_source = source[source.index("def main() -> None:") :]
    assert main_source.index("write_json_atomic(started_output") < main_source.index("for replicate in (\"a\", \"b\"):")
    assert '"test270": False' in main_source
    assert '"ucsd_external75": False' in main_source
