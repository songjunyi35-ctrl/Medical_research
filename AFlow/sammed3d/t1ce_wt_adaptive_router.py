"""Prediction-only utilities for the preregistered 1→8→24 router.

This module is separate from the immutable 2026-08-13 MCTS core.  It contains
no data paths and no test-set entrypoint.  Labels are accepted only by the
explicit ridge fitting and policy scoring helpers; feature extraction and
route execution remain label-free.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from sammed3d.t1ce_wt_mcts import aggregate_scores, binary_dice, subject_id


STAGE1_FEATURE_NAMES = (
    "log_mask_voxels",
    "roi_fraction",
    "near_threshold_fraction",
    "mean_abs_margin",
    "p10_abs_margin",
    "face_contact_fraction",
)
STAGE8_EXTRA_FEATURE_NAMES = (
    "view_disagreement",
    "stage1_stage8_mask_disagreement",
    "log_volume_ratio_stage8_stage1",
    "mean_probability_std_roi",
    "p90_probability_std_roi",
    "stage8_near_threshold_fraction",
)
STAGE8_FEATURE_NAMES = STAGE1_FEATURE_NAMES + STAGE8_EXTRA_FEATURE_NAMES
ACCURACY_MARGINS = (0.0, 0.001, 0.002, 0.005)
RIDGE_ALPHA = 1.0


def _finite_array(values: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def stage1_features(
    logits: np.ndarray,
    mask: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    """Extract the frozen label-free features available after one forward."""
    logits = np.asarray(logits, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if logits.shape != mask.shape or logits.ndim != 3:
        raise ValueError("stage1 logits/mask must be equally shaped 3D arrays")
    roi = logits > float(threshold) - 2.0
    roi_count = int(np.count_nonzero(roi))
    voxel_count = logits.size
    if roi_count:
        margins = np.abs(logits[roi] - float(threshold)).astype(np.float64, copy=False)
        near = float(np.mean(margins <= 0.5))
        mean_margin = float(np.mean(margins))
        p10_margin = float(np.quantile(margins, 0.10))
    else:
        near = 0.0
        mean_margin = 2.0
        p10_margin = 2.0
    mask_count = int(np.count_nonzero(mask))
    boundary_hits = sum(
        int(np.count_nonzero(face))
        for face in (
            mask[0, :, :],
            mask[-1, :, :],
            mask[:, 0, :],
            mask[:, -1, :],
            mask[:, :, 0],
            mask[:, :, -1],
        )
    )
    result = np.asarray(
        (
            np.log1p(mask_count),
            roi_count / voxel_count,
            near,
            mean_margin,
            p10_margin,
            boundary_hits / max(mask_count, 1),
        ),
        dtype=np.float64,
    )
    return _finite_array(result, name="stage1 features")


def stage8_features(
    first_features: np.ndarray,
    stage1_mask: np.ndarray,
    stage8_logits: np.ndarray,
    stage8_mask: np.ndarray,
    probability_std: np.ndarray,
    *,
    threshold: float,
    view_disagreement: float,
) -> np.ndarray:
    """Add frozen multi-view features after the eight-forward stage."""
    first = _finite_array(first_features, name="stage1 features")
    if first.shape != (len(STAGE1_FEATURE_NAMES),):
        raise ValueError("unexpected stage1 feature shape")
    stage1_mask = np.asarray(stage1_mask, dtype=bool)
    stage8_logits = np.asarray(stage8_logits, dtype=np.float32)
    stage8_mask = np.asarray(stage8_mask, dtype=bool)
    probability_std = np.asarray(probability_std, dtype=np.float32)
    if not (
        stage1_mask.shape
        == stage8_logits.shape
        == stage8_mask.shape
        == probability_std.shape
    ) or stage8_logits.ndim != 3:
        raise ValueError("stage8 arrays must have the same 3D shape")
    roi = stage8_logits > float(threshold) - 2.0
    if np.any(roi):
        roi_std = probability_std[roi].astype(np.float64, copy=False)
        margins = np.abs(stage8_logits[roi] - float(threshold))
        mean_std = float(np.mean(roi_std))
        p90_std = float(np.quantile(roi_std, 0.90))
        near = float(np.mean(margins <= 0.5))
    else:
        mean_std = 0.0
        p90_std = 0.0
        near = 0.0
    count1 = int(np.count_nonzero(stage1_mask))
    count8 = int(np.count_nonzero(stage8_mask))
    extra = np.asarray(
        (
            float(view_disagreement),
            1.0 - binary_dice(stage1_mask, stage8_mask),
            np.log1p(count8) - np.log1p(count1),
            mean_std,
            p90_std,
            near,
        ),
        dtype=np.float64,
    )
    return _finite_array(np.concatenate((first, extra)), name="stage8 features")


@dataclass(frozen=True)
class RidgeRouter:
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    alpha: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = _finite_array(features, name="router features")
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("router feature matrix has an unexpected shape")
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        return self.intercept + ((values - mean) / scale) @ coefficients

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "alpha": self.alpha,
        }


def fit_subject_balanced_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    case_ids: Sequence[str],
    *,
    feature_names: Sequence[str],
    alpha: float = RIDGE_ALPHA,
) -> RidgeRouter:
    """Fit deterministic ridge regression with equal total subject weight."""
    x = _finite_array(features, name="ridge features")
    y = _finite_array(targets, name="ridge targets")
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(x) != len(case_ids):
        raise ValueError("ridge inputs have inconsistent shapes")
    if x.shape[1] != len(feature_names) or len(x) < 2:
        raise ValueError("ridge feature names/count are invalid")
    if alpha < 0:
        raise ValueError("ridge alpha must be nonnegative")
    subjects = [subject_id(value) for value in case_ids]
    counts = Counter(subjects)
    weights = np.asarray([1.0 / counts[value] for value in subjects], dtype=np.float64)
    weights *= len(weights) / weights.sum()
    mean = np.average(x, axis=0, weights=weights)
    centered = x - mean
    variance = np.average(centered * centered, axis=0, weights=weights)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale < 1e-12] = 1.0
    normalized = centered / scale
    target_mean = float(np.average(y, weights=weights))
    centered_target = y - target_mean
    root_weight = np.sqrt(weights)[:, None]
    weighted_x = normalized * root_weight
    weighted_y = centered_target * root_weight[:, 0]
    regularized = weighted_x.T @ weighted_x + float(alpha) * np.eye(x.shape[1])
    coefficients = np.linalg.solve(regularized, weighted_x.T @ weighted_y)
    return RidgeRouter(
        feature_names=tuple(feature_names),
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        coefficients=tuple(float(value) for value in coefficients),
        intercept=target_mean,
        alpha=float(alpha),
    )


def threshold_grid(predictions: np.ndarray) -> tuple[float, ...]:
    """Frozen decile grid plus finite sentinels for always/never escalate."""
    values = _finite_array(predictions, name="selector predictions").reshape(-1)
    if not len(values):
        raise ValueError("cannot build a threshold grid from no predictions")
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, 11))
    span = float(values.max() - values.min())
    padding = max(1e-6, span * 0.01)
    grid = [float(values.min() - padding), *(float(value) for value in quantiles), float(values.max() + padding)]
    return tuple(sorted(set(grid)))


def route_stages(
    predicted_gain_1_to_8: np.ndarray,
    predicted_gain_8_to_24: np.ndarray,
    *,
    threshold_1_to_8: float,
    threshold_8_to_24: float,
) -> np.ndarray:
    """Return one of {1,8,24}; higher predicted gain triggers escalation."""
    first = _finite_array(predicted_gain_1_to_8, name="stage1 predicted gain")
    second = _finite_array(predicted_gain_8_to_24, name="stage8 predicted gain")
    if first.shape != second.shape:
        raise ValueError("router prediction shapes differ")
    result = np.ones(first.shape, dtype=np.int16)
    escalate8 = first > float(threshold_1_to_8)
    result[escalate8] = 8
    result[escalate8 & (second > float(threshold_8_to_24))] = 24
    return result


def score_route(
    case_ids: Sequence[str],
    stage1_dice: np.ndarray,
    stage8_dice: np.ndarray,
    stage24_dice: np.ndarray,
    stages: np.ndarray,
) -> dict[str, Any]:
    """Score a fixed route.  This helper is label-aware by design."""
    first = _finite_array(stage1_dice, name="stage1 dice")
    eighth = _finite_array(stage8_dice, name="stage8 dice")
    locked = _finite_array(stage24_dice, name="stage24 dice")
    route = np.asarray(stages, dtype=np.int16)
    if not (first.shape == eighth.shape == locked.shape == route.shape == (len(case_ids),)):
        raise ValueError("route score arrays have inconsistent shapes")
    if not set(np.unique(route)).issubset({1, 8, 24}):
        raise ValueError("route contains an unsupported stage")
    values = np.where(route == 1, first, np.where(route == 8, eighth, locked))
    aggregate = aggregate_scores(case_ids, [float(value) for value in values])
    return {
        **aggregate,
        "case_ids": list(case_ids),
        "case_dice": [float(value) for value in values],
        "forward_counts": [int(value) for value in route],
        "mean_forwards": float(np.mean(route)),
        "max_forwards": int(np.max(route)),
        "stop_stage_counts": {
            str(stage): int(np.count_nonzero(route == stage)) for stage in (1, 8, 24)
        },
    }


def select_thresholds_at_margin(
    case_ids: Sequence[str],
    stage1_dice: np.ndarray,
    stage8_dice: np.ndarray,
    stage24_dice: np.ndarray,
    predicted_gain_1_to_8: np.ndarray,
    predicted_gain_8_to_24: np.ndarray,
    *,
    margin: float,
) -> dict[str, Any]:
    """Exhaust the preregistered selector grid and return the cheapest feasible route."""
    if margin < 0:
        raise ValueError("accuracy margin must be nonnegative")
    reference = score_route(
        case_ids,
        stage1_dice,
        stage8_dice,
        stage24_dice,
        np.full(len(case_ids), 24),
    )
    floor = float(reference["subject_macro_dsc"]) - float(margin)
    candidates: list[dict[str, Any]] = []
    for threshold1 in threshold_grid(predicted_gain_1_to_8):
        for threshold8 in threshold_grid(predicted_gain_8_to_24):
            stages = route_stages(
                predicted_gain_1_to_8,
                predicted_gain_8_to_24,
                threshold_1_to_8=threshold1,
                threshold_8_to_24=threshold8,
            )
            score = score_route(
                case_ids,
                stage1_dice,
                stage8_dice,
                stage24_dice,
                stages,
            )
            if float(score["subject_macro_dsc"]) + 1e-15 >= floor:
                candidates.append(
                    {
                        "threshold_1_to_8": float(threshold1),
                        "threshold_8_to_24": float(threshold8),
                        "score": score,
                    }
                )
    if not candidates:
        raise RuntimeError("locked route unexpectedly absent from feasible selector policies")
    selected = min(
        candidates,
        key=lambda item: (
            float(item["score"]["mean_forwards"]),
            -float(item["score"]["subject_macro_dsc"]),
            int(item["score"]["max_forwards"]),
            float(item["threshold_1_to_8"]),
            float(item["threshold_8_to_24"]),
        ),
    )
    return {
        "margin": float(margin),
        "reference_subject_macro_dsc": float(reference["subject_macro_dsc"]),
        "accuracy_floor": floor,
        "feasible_policies": len(candidates),
        **selected,
    }
