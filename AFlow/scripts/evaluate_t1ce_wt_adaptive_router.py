#!/usr/bin/env python3
"""Run the preregistered prediction-only 1→8→24 adaptive router study.

The only accepted data are the exact 108-case validation atom cache and its
matching validation labels.  No test mode or generic split argument exists.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_adaptive_router import (  # noqa: E402
    ACCURACY_MARGINS,
    RIDGE_ALPHA,
    STAGE1_FEATURE_NAMES,
    STAGE8_FEATURE_NAMES,
    fit_subject_balanced_ridge,
    route_stages,
    score_route,
    select_thresholds_at_margin,
    stage1_features,
    stage8_features,
)
from sammed3d.t1ce_wt_mcts import (  # noqa: E402
    VALIDATION_CACHE_INDEX_SHA256,
    RobustAtomCache,
    WorkflowOutput,
    WorkflowSpec,
    aggregate_scores,
    binary_dice,
    nested_roles,
    selected_atom_indices,
    sha256_file,
    stable_id,
    subject_id,
    workflow_mask,
)
from scripts.train_sammed3d_t1ce_wt_residual_refiner import (  # noqa: E402
    RefinerDataset,
    write_json_atomic,
)


DEFAULT_ATOMS = Path("workspace/BraTS/cache/t1ce_wt_robust_mcts_atoms_f32")
DEFAULT_VALIDATION = Path(
    "workspace/BraTS/cache/t1ce_wt_refiner_graymean_20260813/validation"
)
DEFAULT_OUTPUT_DIR = Path(
    "workspace/BraTS/sammed3d_t1ce_wt/adaptive_router_20260814"
)
DEFAULT_PREREG = Path("ADAPTIVE_ROUTER_PREREG_20260814.md")
DEFAULT_CANDIDATE = Path("MCTS_DEPLOYMENT_CANDIDATE_20260813.json")
DEFAULT_HISTORICAL = Path(
    "workspace/BraTS/sammed3d_t1ce_wt/mcts_workflow_20260813_v2/final_report.json"
)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260814


STAGE1_SPEC = WorkflowSpec(
    acquisition="fixed",
    model_mask=4,
    view_policy="identity",
    fusion="logit_mean",
    threshold_q=-15,
    uncertainty_beta_q=0,
    min_component_voxels=500,
)
STAGE8_SPEC = WorkflowSpec(
    acquisition="fixed",
    model_mask=4,
    view_policy="all8",
    fusion="logit_mean",
    threshold_q=-5,
    uncertainty_beta_q=-5,
    min_component_voxels=250,
)
STAGE24_SPEC = WorkflowSpec.locked_reference()
STAGE_PROTOCOL = {
    "stage1": STAGE1_SPEC.canonical_dict(),
    "stage8": STAGE8_SPEC.canonical_dict(),
    "stage24": STAGE24_SPEC.canonical_dict(),
}
STAGE_PROTOCOL_ID = stable_id(STAGE_PROTOCOL)


def _fail_if_test_path(path: Path) -> None:
    lowered = str(path.resolve()).lower()
    if "sealed_test" in lowered or "test270" in lowered or "/test/" in lowered:
        raise ValueError(f"test-like path is forbidden: {path}")


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    result = np.empty_like(value, dtype=np.float32)
    positive = value >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _fuse_logit_mean(atoms: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    total = np.zeros(atoms.shape[1:], dtype=np.float32)
    for index in indices:
        total += np.asarray(atoms[index], dtype=np.float32)
    return total / len(indices)


def _stage8_probability_statistics(
    atoms: np.ndarray,
    indices: tuple[int, ...],
) -> tuple[np.ndarray, float]:
    probabilities = np.stack([_sigmoid(atoms[index]) for index in indices], axis=0)
    mean = np.mean(probabilities, axis=0, dtype=np.float32)
    mean_square = np.mean(probabilities * probabilities, axis=0, dtype=np.float32)
    standard_deviation = np.sqrt(
        np.maximum(mean_square - mean * mean, 0.0)
    ).astype(np.float32, copy=False)
    flat = probabilities.reshape(len(indices), -1)
    gram = flat @ flat.T
    diagonal = np.diag(gram)
    left, right = np.triu_indices(len(indices), k=1)
    agreements = (2.0 * gram[left, right] + 1e-6) / (
        diagonal[left] + diagonal[right] + 1e-6
    )
    disagreement = float(1.0 - np.mean(agreements, dtype=np.float64))
    return standard_deviation, disagreement


def _extract_case(
    atoms: np.ndarray,
    label: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    stage1_indices = selected_atom_indices(4, "identity")
    stage8_indices = selected_atom_indices(4, "all8")
    stage24_indices = selected_atom_indices(7, "all8")
    stage1_logits = np.asarray(atoms[stage1_indices[0]], dtype=np.float32)
    stage8_logits = _fuse_logit_mean(atoms, stage8_indices)
    probability_std, disagreement = _stage8_probability_statistics(atoms, stage8_indices)
    stage24_logits = _fuse_logit_mean(atoms, stage24_indices)

    output1 = WorkflowOutput(stage1_logits, None, 1, 1, 0.0)
    output8 = WorkflowOutput(stage8_logits, probability_std, 8, 8, disagreement)
    output24 = WorkflowOutput(stage24_logits, None, 24, 24, 0.0)
    mask1 = workflow_mask(output1, STAGE1_SPEC)
    mask8 = workflow_mask(output8, STAGE8_SPEC)
    mask24 = workflow_mask(output24, STAGE24_SPEC)
    features1 = stage1_features(
        stage1_logits,
        mask1,
        threshold=STAGE1_SPEC.threshold,
    )
    features8 = stage8_features(
        features1,
        mask1,
        stage8_logits,
        mask8,
        probability_std,
        threshold=STAGE8_SPEC.threshold,
        view_disagreement=disagreement,
    )
    target = np.asarray(label) > 0
    dice = (
        binary_dice(mask1, target),
        binary_dice(mask8, target),
        binary_dice(mask24, target),
    )
    return features1, features8, dice


def _save_feature_cache(
    path: Path,
    *,
    case_ids: list[str],
    features1: np.ndarray,
    features8: np.ndarray,
    stage_dice: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema=np.asarray("aflow.sammed3d.t1ce_wt_adaptive_router.features.v1"),
            stage_protocol_id=np.asarray(STAGE_PROTOCOL_ID),
            case_ids=np.asarray(case_ids),
            features1=np.asarray(features1, dtype=np.float64),
            features8=np.asarray(features8, dtype=np.float64),
            stage_dice=np.asarray(stage_dice, dtype=np.float64),
        )
    os.replace(temporary, path)


def _load_feature_cache(
    path: Path,
    *,
    expected_case_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as value:
        if str(value["schema"].item()) != "aflow.sammed3d.t1ce_wt_adaptive_router.features.v1":
            raise RuntimeError("unexpected adaptive feature cache schema")
        if str(value["stage_protocol_id"].item()) != STAGE_PROTOCOL_ID:
            raise RuntimeError("adaptive feature cache stage protocol mismatch")
        if list(value["case_ids"].astype(str)) != expected_case_ids:
            raise RuntimeError("adaptive feature cache case ordering mismatch")
        features1 = np.asarray(value["features1"], dtype=np.float64)
        features8 = np.asarray(value["features8"], dtype=np.float64)
        stage_dice = np.asarray(value["stage_dice"], dtype=np.float64)
    expected = len(expected_case_ids)
    if features1.shape != (expected, len(STAGE1_FEATURE_NAMES)):
        raise RuntimeError("adaptive stage1 feature cache shape mismatch")
    if features8.shape != (expected, len(STAGE8_FEATURE_NAMES)):
        raise RuntimeError("adaptive stage8 feature cache shape mismatch")
    if stage_dice.shape != (expected, 3):
        raise RuntimeError("adaptive stage Dice cache shape mismatch")
    return features1, features8, stage_dice


def extract_or_load_features(
    atoms: RobustAtomCache,
    validation: RefinerDataset,
    feature_cache: Path,
    *,
    reuse: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if reuse and feature_cache.exists():
        print(f"reusing {feature_cache}", flush=True)
        return _load_feature_cache(feature_cache, expected_case_ids=atoms.case_ids)
    validation._open()
    assert validation._labels is not None
    features1: list[np.ndarray] = []
    features8: list[np.ndarray] = []
    stage_dice: list[tuple[float, float, float]] = []
    started = time.time()
    for slot, case_id in enumerate(atoms.case_ids):
        first, eighth, dice = _extract_case(
            atoms.atoms[slot],
            validation._labels[slot],
        )
        features1.append(first)
        features8.append(eighth)
        stage_dice.append(dice)
        if (slot + 1) % 4 == 0 or slot + 1 == len(atoms.case_ids):
            print(
                f"feature extraction {slot + 1}/{len(atoms.case_ids)} "
                f"elapsed={time.time() - started:.1f}s case={case_id}",
                flush=True,
            )
    matrix1 = np.stack(features1)
    matrix8 = np.stack(features8)
    dice_matrix = np.asarray(stage_dice, dtype=np.float64)
    _save_feature_cache(
        feature_cache,
        case_ids=atoms.case_ids,
        features1=matrix1,
        features8=matrix8,
        stage_dice=dice_matrix,
    )
    return matrix1, matrix8, dice_matrix


def _subset(values: np.ndarray, indices: list[int]) -> np.ndarray:
    return np.asarray(values[np.asarray(indices, dtype=np.int64)])


def _bootstrap_delta(
    candidate: dict[str, float],
    reference: dict[str, float],
    *,
    seed: int,
) -> dict[str, Any]:
    subjects = sorted(candidate)
    if subjects != sorted(reference):
        raise RuntimeError("paired bootstrap subject sets differ")
    delta = np.asarray([candidate[key] - reference[key] for key in subjects])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(BOOTSTRAP_REPLICATES, len(delta)))
    means = delta[draws].mean(axis=1)
    return {
        "subjects": len(subjects),
        "mean": float(np.mean(delta)),
        "ci95": [float(value) for value in np.quantile(means, (0.025, 0.975))],
        "improved_subjects": int(np.count_nonzero(delta > 0)),
        "worsened_subjects": int(np.count_nonzero(delta < 0)),
        "unchanged_subjects": int(np.count_nonzero(delta == 0)),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
    }


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if np.std(left) < 1e-15 or np.std(right) < 1e-15:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _compact_score(score: dict[str, Any]) -> dict[str, Any]:
    return {
        key: score[key]
        for key in (
            "scan_macro_dsc",
            "subject_macro_dsc",
            "subjects",
            "mean_forwards",
            "max_forwards",
            "stop_stage_counts",
        )
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_cache = args.output_dir / "stage_features_and_dice.npz"
    output_path = args.output_dir / "FINAL_ADAPTIVE_ROUTER_RESULT.json"
    for path in (
        args.atom_cache,
        args.validation_cache,
        args.output_dir,
        args.prereg,
        args.deployment_candidate,
        args.historical_report,
    ):
        _fail_if_test_path(path)
    if tuple(args.accuracy_margins) != ACCURACY_MARGINS:
        raise ValueError("accuracy margins must match the preregistration")
    if args.seed != 20260813 or args.ridge_alpha != RIDGE_ALPHA:
        raise ValueError("seed and ridge alpha must match the preregistration")

    atoms = RobustAtomCache(args.atom_cache)
    validation = RefinerDataset(args.validation_cache)
    if validation.case_ids != atoms.case_ids or len(atoms.case_ids) != 108:
        raise RuntimeError("validation atom/label case ordering mismatch")
    validation_index = args.validation_cache / "index.json"
    if sha256_file(validation_index) != VALIDATION_CACHE_INDEX_SHA256:
        raise RuntimeError("validation cache binding mismatch")
    deployment = json.loads(args.deployment_candidate.read_text())
    if deployment.get("spec") != STAGE8_SPEC.canonical_dict():
        raise RuntimeError("stage8 does not match the frozen deployment candidate")
    historical = json.loads(args.historical_report.read_text())
    if historical.get("test_access") is not False or len(historical.get("folds", [])) != 3:
        raise RuntimeError("locked historical report binding is invalid")

    features1, features8, stage_dice = extract_or_load_features(
        atoms,
        validation,
        feature_cache,
        reuse=args.reuse_feature_cache,
    )
    dice1, dice8, dice24 = (stage_dice[:, index] for index in range(3))

    fixed: dict[str, dict[str, Any]] = {}
    for name, values, forwards in (
        ("stage1", dice1, 1),
        ("stage8", dice8, 8),
        ("stage24_locked", dice24, 24),
    ):
        score = score_route(
            atoms.case_ids,
            dice1,
            dice8,
            dice24,
            np.full(len(atoms.case_ids), forwards),
        )
        fixed[name] = score
    stacked = np.stack((dice1, dice8, dice24), axis=1)
    stage_values = np.asarray((1, 8, 24), dtype=np.int16)
    oracle_stages = stage_values[np.argmax(stacked, axis=1)]
    oracle = score_route(
        atoms.case_ids,
        dice1,
        dice8,
        dice24,
        oracle_stages,
    )

    folds: list[dict[str, Any]] = []
    oof: dict[float, dict[int, tuple[float, int]]] = {
        margin: {} for margin in args.accuracy_margins
    }
    oof_prediction1: dict[int, float] = {}
    oof_prediction8: dict[int, float] = {}
    target1 = dice8 - dice1
    target8 = dice24 - dice8
    for outer_fold in range(3):
        roles = nested_roles(atoms.case_ids, outer_fold, seed=args.seed)
        search_indices = roles["mcts_search"]
        selector_indices = roles["inner_selector"]
        outer_indices = roles["outer_holdout"]
        model1 = fit_subject_balanced_ridge(
            _subset(features1, search_indices),
            _subset(target1, search_indices),
            [atoms.case_ids[index] for index in search_indices],
            feature_names=STAGE1_FEATURE_NAMES,
            alpha=args.ridge_alpha,
        )
        model8 = fit_subject_balanced_ridge(
            _subset(features8, search_indices),
            _subset(target8, search_indices),
            [atoms.case_ids[index] for index in search_indices],
            feature_names=STAGE8_FEATURE_NAMES,
            alpha=args.ridge_alpha,
        )
        prediction1 = model1.predict(features1)
        prediction8 = model8.predict(features8)
        for index in outer_indices:
            if index in oof_prediction1:
                raise RuntimeError("outer folds overlap")
            oof_prediction1[index] = float(prediction1[index])
            oof_prediction8[index] = float(prediction8[index])
        selector_ids = [atoms.case_ids[index] for index in selector_indices]
        outer_ids = [atoms.case_ids[index] for index in outer_indices]
        margin_records: list[dict[str, Any]] = []
        for margin in args.accuracy_margins:
            selection = select_thresholds_at_margin(
                selector_ids,
                _subset(dice1, selector_indices),
                _subset(dice8, selector_indices),
                _subset(dice24, selector_indices),
                _subset(prediction1, selector_indices),
                _subset(prediction8, selector_indices),
                margin=margin,
            )
            outer_stages = route_stages(
                _subset(prediction1, outer_indices),
                _subset(prediction8, outer_indices),
                threshold_1_to_8=selection["threshold_1_to_8"],
                threshold_8_to_24=selection["threshold_8_to_24"],
            )
            outer_score = score_route(
                outer_ids,
                _subset(dice1, outer_indices),
                _subset(dice8, outer_indices),
                _subset(dice24, outer_indices),
                outer_stages,
            )
            for local_index, global_index in enumerate(outer_indices):
                if global_index in oof[margin]:
                    raise RuntimeError("outer route folds overlap")
                oof[margin][global_index] = (
                    float(outer_score["case_dice"][local_index]),
                    int(outer_score["forward_counts"][local_index]),
                )
            margin_records.append(
                {
                    "margin": margin,
                    "threshold_1_to_8": selection["threshold_1_to_8"],
                    "threshold_8_to_24": selection["threshold_8_to_24"],
                    "reference_subject_macro_dsc": selection[
                        "reference_subject_macro_dsc"
                    ],
                    "accuracy_floor": selection["accuracy_floor"],
                    "feasible_policies": selection["feasible_policies"],
                    "selector_score": _compact_score(selection["score"]),
                    "outer_score": outer_score,
                }
            )
        folds.append(
            {
                "outer_fold": outer_fold,
                "roles": roles,
                "role_subjects": {
                    role: sorted({subject_id(atoms.case_ids[index]) for index in indices})
                    for role, indices in roles.items()
                },
                "stage1_router": model1.as_dict(),
                "stage8_router": model8.as_dict(),
                "outer_gain_prediction": {
                    "stage1_to_stage8_pearson": _pearson(
                        _subset(prediction1, outer_indices),
                        _subset(target1, outer_indices),
                    ),
                    "stage8_to_stage24_pearson": _pearson(
                        _subset(prediction8, outer_indices),
                        _subset(target8, outer_indices),
                    ),
                },
                "margin_selections": margin_records,
            }
        )
        print(f"completed nested outer fold {outer_fold}", flush=True)

    if sorted(oof_prediction1) != list(range(108)):
        raise RuntimeError("OOF router predictions do not cover 108 scans")
    locked_aggregate = aggregate_scores(
        atoms.case_ids, [float(value) for value in dice24]
    )
    fixed8_aggregate = aggregate_scores(
        atoms.case_ids, [float(value) for value in dice8]
    )
    margins: dict[str, Any] = {}
    for margin_index, margin in enumerate(args.accuracy_margins):
        if sorted(oof[margin]) != list(range(108)):
            raise RuntimeError("OOF adaptive route does not cover 108 scans")
        values = np.asarray([oof[margin][index][0] for index in range(108)])
        forwards = np.asarray([oof[margin][index][1] for index in range(108)])
        aggregate = aggregate_scores(
            atoms.case_ids, [float(value) for value in values]
        )
        paired_locked = _bootstrap_delta(
            aggregate["subject_values"],
            locked_aggregate["subject_values"],
            seed=BOOTSTRAP_SEED + margin_index,
        )
        paired_fixed8 = _bootstrap_delta(
            aggregate["subject_values"],
            fixed8_aggregate["subject_values"],
            seed=BOOTSTRAP_SEED + 100 + margin_index,
        )
        margins[f"{margin:.3f}"] = {
            "accuracy_margin": margin,
            "candidate": {
                **aggregate,
                "case_ids": atoms.case_ids,
                "case_dice": [float(value) for value in values],
                "forward_counts": [int(value) for value in forwards],
                "mean_forwards": float(np.mean(forwards)),
                "max_forwards": int(np.max(forwards)),
                "stop_stage_counts": {
                    str(stage): int(np.count_nonzero(forwards == stage))
                    for stage in (1, 8, 24)
                },
            },
            "paired_vs_locked": paired_locked,
            "paired_vs_fixed_stage8": paired_fixed8,
            "forward_reduction_vs_locked": 1.0 - float(np.mean(forwards)) / 24.0,
            "forward_reduction_vs_fixed_stage8": 1.0 - float(np.mean(forwards)) / 8.0,
        }

    prediction1_ordered = np.asarray([oof_prediction1[index] for index in range(108)])
    prediction8_ordered = np.asarray([oof_prediction8[index] for index in range(108)])
    report = {
        "schema": "aflow.sammed3d.t1ce_wt_adaptive_router.result",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "split": "validation",
        "evidence_status": (
            "post-preregistered development cross-fitting on previously used validation; "
            "stage workflows were inherited from prior validation development"
        ),
        "geometry": "GT-centered N4 128^3 oracle geometry; not automatic localization",
        "scope": "refiner member-view forwards; not end-to-end SAM-Med3D latency",
        "stage_protocol": STAGE_PROTOCOL,
        "stage_protocol_id": STAGE_PROTOCOL_ID,
        "fixed_stage_results": fixed,
        "oracle_routing_ceiling": oracle,
        "folds": folds,
        "oof_margins": margins,
        "oof_gain_prediction": {
            "stage1_to_stage8_pearson": _pearson(prediction1_ordered, target1),
            "stage8_to_stage24_pearson": _pearson(prediction8_ordered, target8),
        },
        "protocol": {
            "seed": args.seed,
            "ridge_alpha": args.ridge_alpha,
            "accuracy_margins": list(args.accuracy_margins),
            "outer_folds": 3,
            "scans": 108,
            "subjects": len({subject_id(value) for value in atoms.case_ids}),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "bindings": {
            "prereg": str(args.prereg),
            "prereg_sha256": sha256_file(args.prereg),
            "atom_index_sha256": sha256_file(atoms.index_path),
            "validation_index_sha256": sha256_file(validation_index),
            "feature_cache": str(feature_cache),
            "feature_cache_sha256": sha256_file(feature_cache),
            "deployment_candidate_sha256": sha256_file(args.deployment_candidate),
            "historical_mcts_report_sha256": sha256_file(args.historical_report),
            "mcts_core_sha256": sha256_file(ROOT / "sammed3d/t1ce_wt_mcts.py"),
            "router_core_sha256": sha256_file(
                ROOT / "sammed3d/t1ce_wt_adaptive_router.py"
            ),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "test_sha256": sha256_file(
                ROOT / "tests/test_t1ce_wt_adaptive_router.py"
            ),
        },
        "elapsed_seconds": time.time() - started,
    }
    write_json_atomic(output_path, report)
    concise = {
        "fixed": {
            key: {
                "scan_dice": value["scan_macro_dsc"],
                "subject_dice": value["subject_macro_dsc"],
                "forwards": value["mean_forwards"],
            }
            for key, value in fixed.items()
        },
        "oracle": {
            "scan_dice": oracle["scan_macro_dsc"],
            "subject_dice": oracle["subject_macro_dsc"],
            "forwards": oracle["mean_forwards"],
        },
        "adaptive": {
            key: {
                "scan_dice": value["candidate"]["scan_macro_dsc"],
                "subject_dice": value["candidate"]["subject_macro_dsc"],
                "delta_locked": value["paired_vs_locked"]["mean"],
                "ci95": value["paired_vs_locked"]["ci95"],
                "forwards": value["candidate"]["mean_forwards"],
                "stops": value["candidate"]["stop_stage_counts"],
            }
            for key, value in margins.items()
        },
        "gain_prediction": report["oof_gain_prediction"],
        "result": str(output_path),
        "result_sha256": sha256_file(output_path),
    }
    print(json.dumps(concise, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atom-cache", type=Path, default=DEFAULT_ATOMS)
    parser.add_argument("--validation-cache", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--deployment-candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--historical-report", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--accuracy-margins", type=float, nargs="+", default=ACCURACY_MARGINS)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--ridge-alpha", type=float, default=RIDGE_ALPHA)
    parser.add_argument("--reuse-feature-cache", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
