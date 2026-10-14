from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.analyze_t1ce_wt_plain_long_schedule import analyze
from scripts.train_t1ce_wt_plain_long_schedule import (
    assert_inner_only_path,
    learning_rate_for_epoch,
)


def test_frozen_lr_schedule_has_expected_endpoints_and_is_monotone() -> None:
    values = [
        learning_rate_for_epoch(
            epoch,
            base_lr=2e-4,
            min_lr=1e-5,
            constant_epochs=100,
            total_epochs=200,
        )
        for epoch in range(1, 201)
    ]
    assert values[:101] == [2e-4] * 101
    assert values[-1] == pytest.approx(1e-5)
    assert all(left >= right for left, right in zip(values[100:], values[101:]))


@pytest.mark.parametrize("name", ["selector.jsonl", "confirmation.jsonl", "sealed/x", "test270/x"])
def test_inner_path_guard_rejects_forbidden_roles(tmp_path, name: str) -> None:
    with pytest.raises(ValueError):
        assert_inner_only_path(tmp_path / name)


def _report(window: str, shift: float, point_hit: float = 0.9) -> dict:
    cases = [
        {
            "case_id": f"case-{index}",
            "subject_id": f"subject-{index}",
            "locator_dsc": 0.7 + shift,
        }
        for index in range(12)
    ]
    return {
        "window": window,
        "role": "fresh_tune",
        "epoch": 80 if window == "pre100" else 160,
        "checkpoint": {"path": f"{window}.pth", "sha256": window},
        "summary": {
            "locator_dsc_mean": 0.7 + shift,
            "locator_subject_dsc_mean": 0.7 + shift,
            "automatic_point_hit_rate": point_hit,
            "per_case": cases,
        },
        "reload_max_abs_difference": 0.0,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }


def test_all_gates_pass_for_clear_paired_improvement() -> None:
    result = analyze(_report("pre100", 0.0), _report("post100", 0.01))
    assert result["all_inner_gates_passed"] is True
    assert result["outer_selector_authorized"] is True


def test_point_drop_blocks_selector_even_when_dice_improves() -> None:
    result = analyze(
        _report("pre100", 0.0, point_hit=0.9),
        _report("post100", 0.01, point_hit=0.8),
    )
    assert result["gates"]["point_hit_drop_at_most_0.02"] is False
    assert result["outer_selector_authorized"] is False


def test_unsealed_report_is_rejected() -> None:
    post = deepcopy(_report("post100", 0.01))
    post["test_access"] = True
    with pytest.raises(ValueError):
        analyze(_report("pre100", 0.0), post)
