"""Efficiency-oriented selection utilities for the frozen MCTS workflow space.

This module is intentionally separate from :mod:`sammed3d.t1ce_wt_mcts`.
The exact atom cache and the completed 2026-08-13 MCTS run bind the SHA-256 of
that historical module, so changing it would destroy rather than improve the
old evidence chain.

The helpers below make two policies explicit:

* the locked 24-forward reference must reach the inner selector even when its
  cost-aware search reward is not in the top-k; and
* deployment selection is made from a Dice--compute Pareto set at several
  declared accuracy margins instead of hiding the trade-off in one scalar.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


DEFAULT_ACCURACY_MARGINS = (0.0, 0.001, 0.002, 0.005)


def _workflow_id(record: Mapping[str, Any]) -> str:
    value = record.get("workflow_id")
    if not isinstance(value, str) or not value:
        raise ValueError("workflow evaluation is missing a workflow_id")
    return value


def mandatory_anchor_pool(
    ranked_search_evaluations: Sequence[Mapping[str, Any]],
    *,
    anchor_evaluation: Mapping[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    """Return exactly ``top_k`` unique candidates including the locked anchor.

    The historical runner inserted the anchor in the search evaluation cache,
    but then sent only the top-k scalar rewards to the selector.  Consequently
    the anchor could still be dropped.  Here it consumes one selector slot and
    is always first; the remaining slots are filled by search rank.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    anchor_id = _workflow_id(anchor_evaluation)
    result = [dict(anchor_evaluation)]
    seen = {anchor_id}
    for evaluation in ranked_search_evaluations:
        workflow_id = _workflow_id(evaluation)
        if workflow_id in seen:
            continue
        result.append(dict(evaluation))
        seen.add(workflow_id)
        if len(result) == top_k:
            break
    if len(result) != top_k:
        raise ValueError("fewer unique workflows than requested selector pool")
    return result


def pareto_dominates(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    accuracy_key: str = "accuracy",
    compute_key: str = "mean_forwards",
) -> bool:
    """Whether ``left`` weakly improves accuracy and compute, one strictly."""
    left_accuracy = float(left[accuracy_key])
    right_accuracy = float(right[accuracy_key])
    left_compute = float(left[compute_key])
    right_compute = float(right[compute_key])
    return (
        left_accuracy >= right_accuracy
        and left_compute <= right_compute
        and (left_accuracy > right_accuracy or left_compute < right_compute)
    )


def pareto_front(
    evaluations: Iterable[Mapping[str, Any]],
    *,
    accuracy_key: str = "accuracy",
    compute_key: str = "mean_forwards",
) -> list[dict[str, Any]]:
    """Return the deterministic Dice--compute nondominated frontier."""
    records = [dict(value) for value in evaluations]
    ids = [_workflow_id(value) for value in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Pareto input contains duplicate workflow ids")
    front = [
        value
        for index, value in enumerate(records)
        if not any(
            pareto_dominates(
                other,
                value,
                accuracy_key=accuracy_key,
                compute_key=compute_key,
            )
            for other_index, other in enumerate(records)
            if other_index != index
        )
    ]
    return sorted(
        front,
        key=lambda value: (
            float(value[compute_key]),
            -float(value[accuracy_key]),
            _workflow_id(value),
        ),
    )


@dataclass(frozen=True)
class MarginSelection:
    margin: float
    reference_accuracy: float
    accuracy_floor: float
    feasible_workflows: int
    selected: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "margin": self.margin,
            "reference_accuracy": self.reference_accuracy,
            "accuracy_floor": self.accuracy_floor,
            "feasible_workflows": self.feasible_workflows,
            "selected": self.selected,
        }


def select_at_accuracy_margin(
    evaluations: Sequence[Mapping[str, Any]],
    *,
    anchor_workflow_id: str,
    margin: float,
    accuracy_key: str = "accuracy",
    compute_key: str = "mean_forwards",
) -> MarginSelection:
    """Choose the cheapest workflow within ``margin`` of the locked anchor.

    This is a development selection rule, not a statistical non-inferiority
    test.  An outer/external paired interval must be reported separately.
    """
    if margin < 0:
        raise ValueError("accuracy margin must be nonnegative")
    records = [dict(value) for value in evaluations]
    anchors = [value for value in records if _workflow_id(value) == anchor_workflow_id]
    if len(anchors) != 1:
        raise ValueError("selector pool must contain the locked anchor exactly once")
    reference_accuracy = float(anchors[0][accuracy_key])
    floor = reference_accuracy - float(margin)
    feasible = [value for value in records if float(value[accuracy_key]) >= floor]
    if not feasible:
        raise RuntimeError("locked anchor unexpectedly failed its own accuracy floor")
    selected = min(
        feasible,
        key=lambda value: (
            float(value[compute_key]),
            -float(value[accuracy_key]),
            _workflow_id(value),
        ),
    )
    return MarginSelection(
        margin=float(margin),
        reference_accuracy=reference_accuracy,
        accuracy_floor=floor,
        feasible_workflows=len(feasible),
        selected=selected,
    )


def margin_sweep(
    evaluations: Sequence[Mapping[str, Any]],
    *,
    anchor_workflow_id: str,
    margins: Sequence[float] = DEFAULT_ACCURACY_MARGINS,
) -> list[dict[str, Any]]:
    if not margins:
        raise ValueError("at least one accuracy margin is required")
    if tuple(sorted(set(float(value) for value in margins))) != tuple(
        float(value) for value in margins
    ):
        raise ValueError("accuracy margins must be unique and increasing")
    return [
        select_at_accuracy_margin(
            evaluations,
            anchor_workflow_id=anchor_workflow_id,
            margin=float(margin),
        ).as_dict()
        for margin in margins
    ]
