from __future__ import annotations

import numpy as np

from sammed3d.t1ce_wt_adaptive_router import (
    STAGE1_FEATURE_NAMES,
    fit_subject_balanced_ridge,
    route_stages,
    score_route,
    select_thresholds_at_margin,
    stage1_features,
    threshold_grid,
)


def test_stage1_features_are_label_free_finite_and_boundary_sensitive() -> None:
    logits = np.full((4, 4, 4), -3.0, dtype=np.float32)
    logits[0, 1:3, 1:3] = 0.0
    mask = logits > -0.75
    features = stage1_features(logits, mask, threshold=-0.75)
    assert features.shape == (len(STAGE1_FEATURE_NAMES),)
    assert np.all(np.isfinite(features))
    assert features[-1] > 0.0


def test_subject_balanced_ridge_is_deterministic_and_predicts_signal() -> None:
    features = np.asarray([[0.0], [0.0], [1.0], [2.0]], dtype=np.float64)
    targets = np.asarray([0.0, 0.0, 1.0, 2.0], dtype=np.float64)
    case_ids = ["subject-a-100", "subject-a-101", "subject-b-100", "subject-c-100"]
    first = fit_subject_balanced_ridge(
        features,
        targets,
        case_ids,
        feature_names=("signal",),
        alpha=1.0,
    )
    second = fit_subject_balanced_ridge(
        features,
        targets,
        case_ids,
        feature_names=("signal",),
        alpha=1.0,
    )
    assert first == second
    prediction = first.predict(np.asarray([[0.0], [2.0]]))
    assert prediction[1] > prediction[0]


def test_threshold_grid_contains_always_and_never_escalate_sentinels() -> None:
    predictions = np.asarray([-0.1, 0.0, 0.2])
    grid = threshold_grid(predictions)
    assert grid[0] < predictions.min()
    assert grid[-1] > predictions.max()


def test_route_stages_are_nested_and_never_pay_for_unused_stage24() -> None:
    first = np.asarray([-0.2, 0.1, 0.2])
    second = np.asarray([1.0, -0.1, 0.3])
    stages = route_stages(
        first,
        second,
        threshold_1_to_8=0.0,
        threshold_8_to_24=0.0,
    )
    assert stages.tolist() == [1, 8, 24]


def test_selector_keeps_locked_route_feasible_at_strict_margin() -> None:
    ids = ["subject-a", "subject-b", "subject-c"]
    stage1 = np.asarray([0.7, 0.6, 0.7])
    stage8 = np.asarray([0.8, 0.7, 0.8])
    stage24 = np.asarray([0.9, 0.8, 0.9])
    prediction1 = np.asarray([0.1, 0.2, 0.3])
    prediction8 = np.asarray([0.1, 0.2, 0.3])
    selected = select_thresholds_at_margin(
        ids,
        stage1,
        stage8,
        stage24,
        prediction1,
        prediction8,
        margin=0.0,
    )
    assert selected["score"]["subject_macro_dsc"] >= np.mean(stage24)
    assert set(selected["score"]["forward_counts"]) == {24}
    rescored = score_route(ids, stage1, stage8, stage24, np.full(3, 24))
    assert rescored["subject_macro_dsc"] == np.mean(stage24)
