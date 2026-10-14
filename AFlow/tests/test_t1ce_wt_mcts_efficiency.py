from __future__ import annotations

import pytest

from sammed3d.t1ce_wt_mcts_efficiency import (
    mandatory_anchor_pool,
    margin_sweep,
    pareto_front,
    select_at_accuracy_margin,
)


def record(workflow_id: str, accuracy: float, forwards: float, reward: float = 0.0) -> dict:
    return {
        "workflow_id": workflow_id,
        "accuracy": accuracy,
        "mean_forwards": forwards,
        "reward": reward,
    }


def test_mandatory_anchor_reaches_selector_when_not_in_reward_top_k() -> None:
    anchor = record("locked", 0.70, 24, reward=0.68)
    ranked = [
        record("cheap-a", 0.699, 1, reward=0.698),
        record("cheap-b", 0.698, 3, reward=0.697),
        anchor,
        record("other", 0.69, 8, reward=0.689),
    ]
    pool = mandatory_anchor_pool(ranked, anchor_evaluation=anchor, top_k=3)
    assert [value["workflow_id"] for value in pool] == ["locked", "cheap-a", "cheap-b"]


def test_pareto_front_removes_more_expensive_less_accurate_workflow() -> None:
    front = pareto_front(
        [
            record("one", 0.698, 1),
            record("eight", 0.701, 8),
            record("dominated", 0.697, 12),
            record("locked", 0.702, 24),
        ]
    )
    assert [value["workflow_id"] for value in front] == ["one", "eight", "locked"]


def test_margin_selection_prefers_compute_only_after_accuracy_floor() -> None:
    values = [
        record("locked", 0.7000, 24),
        record("one", 0.6979, 1),
        record("eight", 0.6991, 8),
    ]
    strict = select_at_accuracy_margin(values, anchor_workflow_id="locked", margin=0.0)
    medium = select_at_accuracy_margin(values, anchor_workflow_id="locked", margin=0.001)
    loose = select_at_accuracy_margin(values, anchor_workflow_id="locked", margin=0.003)
    assert strict.selected["workflow_id"] == "locked"
    assert medium.selected["workflow_id"] == "eight"
    assert loose.selected["workflow_id"] == "one"


def test_margin_sweep_requires_declared_increasing_unique_margins() -> None:
    values = [record("locked", 0.7, 24)]
    with pytest.raises(ValueError, match="increasing"):
        margin_sweep(values, anchor_workflow_id="locked", margins=(0.002, 0.001))
