#!/usr/bin/env python3
"""Frozen paired and budget-matched random controls for adaptive routing."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_mcts import aggregate_scores, sha256_file  # noqa: E402
from scripts.train_sammed3d_t1ce_wt_residual_refiner import write_json_atomic  # noqa: E402


DEFAULT_RESULT = Path(
    "workspace/BraTS/sammed3d_t1ce_wt/adaptive_router_20260814/"
    "FINAL_ADAPTIVE_ROUTER_RESULT.json"
)
DEFAULT_FEATURES = Path(
    "workspace/BraTS/sammed3d_t1ce_wt/adaptive_router_20260814/"
    "stage_features_and_dice.npz"
)
DEFAULT_PREREG = Path("ADAPTIVE_ROUTER_CONTROL_PREREG_20260814.md")
DEFAULT_OUTPUT = Path(
    "workspace/BraTS/sammed3d_t1ce_wt/adaptive_router_20260814/"
    "FINAL_ADAPTIVE_ROUTER_CONTROL.json"
)
EXPECTED_RESULT_SHA256 = "dbba96e593c4bcf7742edb7f1e8afc614b058c6625ec8548e418849aef74eddf"
REPLICATES = 10_000
SEED = 20260814


def bootstrap_delta(
    candidate: dict[str, float],
    reference: dict[str, float],
    *,
    seed: int,
) -> dict[str, Any]:
    subjects = sorted(candidate)
    if subjects != sorted(reference):
        raise RuntimeError("paired subject sets differ")
    delta = np.asarray([candidate[key] - reference[key] for key in subjects])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(REPLICATES, len(delta)))
    means = delta[draws].mean(axis=1)
    return {
        "subjects": len(subjects),
        "mean": float(np.mean(delta)),
        "ci95": [float(value) for value in np.quantile(means, (0.025, 0.975))],
        "improved_subjects": int(np.count_nonzero(delta > 0)),
        "worsened_subjects": int(np.count_nonzero(delta < 0)),
        "unchanged_subjects": int(np.count_nonzero(delta == 0)),
        "replicates": REPLICATES,
        "seed": seed,
    }


def random_count_matched(
    report: dict[str, Any],
    stage_dice: np.ndarray,
    case_ids: list[str],
    *,
    margin_key: str,
    seed: int,
) -> dict[str, Any]:
    fold_payloads: list[tuple[np.ndarray, np.ndarray]] = []
    learned_values = np.empty(len(case_ids), dtype=np.float64)
    for fold in report["folds"]:
        selection = next(
            value
            for value in fold["margin_selections"]
            if f"{float(value['margin']):.3f}" == margin_key
        )
        indices = np.asarray(fold["roles"]["outer_holdout"], dtype=np.int64)
        stages = np.asarray(selection["outer_score"]["forward_counts"], dtype=np.int16)
        values = np.asarray(selection["outer_score"]["case_dice"], dtype=np.float64)
        if len(indices) != len(stages) or len(indices) != len(values):
            raise RuntimeError("fold route lengths differ")
        if list(selection["outer_score"]["case_ids"]) != [case_ids[index] for index in indices]:
            raise RuntimeError("fold route case ordering differs")
        learned_values[indices] = values
        fold_payloads.append((indices, stages))

    rng = np.random.default_rng(seed)
    random_subject_dice = np.empty(REPLICATES, dtype=np.float64)
    stage_column = {1: 0, 8: 1, 24: 2}
    for replicate in range(REPLICATES):
        routed = np.empty(len(case_ids), dtype=np.float64)
        for indices, stages in fold_payloads:
            shuffled = rng.permutation(stages)
            for index, stage in zip(indices, shuffled, strict=True):
                routed[index] = stage_dice[index, stage_column[int(stage)]]
        aggregate = aggregate_scores(
            case_ids, [float(value) for value in routed]
        )
        random_subject_dice[replicate] = aggregate["subject_macro_dsc"]
    learned = aggregate_scores(
        case_ids, [float(value) for value in learned_values]
    )["subject_macro_dsc"]
    return {
        "margin": float(margin_key),
        "learned_subject_macro_dsc": float(learned),
        "random_subject_macro_dsc_mean": float(np.mean(random_subject_dice)),
        "random_subject_macro_dsc_std": float(np.std(random_subject_dice, ddof=1)),
        "random_subject_macro_dsc_ci95": [
            float(value) for value in np.quantile(random_subject_dice, (0.025, 0.975))
        ],
        "learned_minus_random_mean": float(learned - np.mean(random_subject_dice)),
        "fraction_random_at_least_learned": float(
            np.mean(random_subject_dice >= learned)
        ),
        "replicates": REPLICATES,
        "seed": seed,
        "matching": "exact outer-fold 1/8/24 stop counts",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.result, args.features, args.prereg, args.output):
        lowered = str(path.resolve()).lower()
        if "sealed_test" in lowered or "test270" in lowered:
            raise ValueError("test paths are forbidden")
    if sha256_file(args.result) != EXPECTED_RESULT_SHA256:
        raise RuntimeError("adaptive result differs from the frozen control input")
    report = json.loads(args.result.read_text())
    if report.get("test_access") is not False or report.get("split") != "validation":
        raise RuntimeError("adaptive result has an invalid evidence boundary")
    with np.load(args.features, allow_pickle=False) as value:
        case_ids = list(value["case_ids"].astype(str))
        stage_dice = np.asarray(value["stage_dice"], dtype=np.float64)
    if case_ids != report["fixed_stage_results"]["stage24_locked"]["case_ids"]:
        raise RuntimeError("feature/result case ordering differs")

    fixed = report["fixed_stage_results"]
    adaptive_strict = report["oof_margins"]["0.000"]["candidate"]
    paired = {
        "stage1_minus_locked": bootstrap_delta(
            fixed["stage1"]["subject_values"],
            fixed["stage24_locked"]["subject_values"],
            seed=SEED,
        ),
        "stage8_minus_locked": bootstrap_delta(
            fixed["stage8"]["subject_values"],
            fixed["stage24_locked"]["subject_values"],
            seed=SEED + 1,
        ),
        "stage8_minus_stage1": bootstrap_delta(
            fixed["stage8"]["subject_values"],
            fixed["stage1"]["subject_values"],
            seed=SEED + 2,
        ),
        "adaptive_strict_minus_stage1": bootstrap_delta(
            adaptive_strict["subject_values"],
            fixed["stage1"]["subject_values"],
            seed=SEED + 3,
        ),
        "adaptive_strict_minus_stage8": bootstrap_delta(
            adaptive_strict["subject_values"],
            fixed["stage8"]["subject_values"],
            seed=SEED + 4,
        ),
    }
    random_controls = {
        margin: random_count_matched(
            report,
            stage_dice,
            case_ids,
            margin_key=margin,
            seed=SEED + index,
        )
        for index, margin in enumerate(("0.000", "0.001", "0.002", "0.005"))
    }
    strict_random = random_controls["0.000"]
    adaptive_supported = (
        strict_random["learned_minus_random_mean"] >= 0.0005
        and strict_random["fraction_random_at_least_learned"] <= 0.05
    )
    output = {
        "schema": "aflow.sammed3d.t1ce_wt_adaptive_router.controls",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "split": "validation",
        "evidence_status": "post-preregistered control analysis on previously used validation",
        "paired_subject_bootstrap": paired,
        "count_matched_random_routing": random_controls,
        "adaptive_routing_intelligence_supported": adaptive_supported,
        "interpretation": (
            "adaptive routing supported beyond count-matched random allocation"
            if adaptive_supported
            else "adaptive routing not supported beyond count-matched random allocation; "
            "attribute primary efficiency gain to fixed member/view redundancy"
        ),
        "bindings": {
            "adaptive_result": str(args.result),
            "adaptive_result_sha256": sha256_file(args.result),
            "features": str(args.features),
            "features_sha256": sha256_file(args.features),
            "prereg": str(args.prereg),
            "prereg_sha256": sha256_file(args.prereg),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_json_atomic(args.output, output)
    concise = {
        "paired": paired,
        "random_controls": random_controls,
        "adaptive_supported": adaptive_supported,
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
    }
    print(json.dumps(concise, indent=2), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
