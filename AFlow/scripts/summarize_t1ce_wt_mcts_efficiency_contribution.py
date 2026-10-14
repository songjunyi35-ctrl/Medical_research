#!/usr/bin/env python3
"""Aggregate three-fold equal-budget controller reports into OOF evidence."""
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
from sammed3d.t1ce_wt_mcts_efficiency import DEFAULT_ACCURACY_MARGINS  # noqa: E402
from scripts.train_sammed3d_t1ce_wt_residual_refiner import write_json_atomic  # noqa: E402


DEFAULT_REPORTS = Path(
    "workspace/BraTS/sammed3d_t1ce_wt/"
    "mcts_efficiency_contribution_20260813/controller_comparison"
)
DEFAULT_HISTORICAL = Path(
    "workspace/BraTS/sammed3d_t1ce_wt/mcts_workflow_20260813_v2/final_report.json"
)


def bootstrap_delta(
    candidate: dict[str, float],
    reference: dict[str, float],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    subjects = sorted(set(candidate) & set(reference))
    if set(subjects) != set(candidate) or set(subjects) != set(reference):
        raise RuntimeError("paired subject sets differ")
    delta = np.asarray(
        [candidate[key] - reference[key] for key in subjects],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(replicates, len(delta)))
    means = delta[draws].mean(axis=1)
    return {
        "subjects": len(subjects),
        "mean": float(delta.mean()),
        "ci95": [float(value) for value in np.quantile(means, (0.025, 0.975))],
        "improved_subjects": int(np.count_nonzero(delta > 0)),
        "worsened_subjects": int(np.count_nonzero(delta < 0)),
        "unchanged_subjects": int(np.count_nonzero(delta == 0)),
        "replicates": replicates,
        "seed": seed,
    }


def selected_for_margin(report: dict[str, Any], margin: float) -> dict[str, Any]:
    matches = [
        value
        for value in report["accuracy_margin_sweep"]
        if abs(float(value["margin"]) - margin) < 1e-12
    ]
    if len(matches) != 1:
        raise RuntimeError("fold report does not contain exactly one requested margin")
    return matches[0]["selected"]


def run(args: argparse.Namespace) -> dict[str, Any]:
    historical = json.loads(args.historical_report.read_text())
    if historical.get("test_access") is not False or len(historical.get("folds", [])) != 3:
        raise ValueError("historical locked-reference report is invalid")
    locked_cases: dict[str, float] = {}
    for fold in historical["folds"]:
        for case_id, dice in zip(
            fold["outer_reference"]["case_ids"],
            fold["outer_reference"]["case_dice"],
            strict=True,
        ):
            if case_id in locked_cases:
                raise RuntimeError("historical outer folds overlap")
            locked_cases[case_id] = float(dice)
    if len(locked_cases) != 108:
        raise RuntimeError("historical locked OOF does not cover 108 cases")
    ordered_ids = sorted(locked_cases)
    locked_aggregate = aggregate_scores(
        ordered_ids,
        [locked_cases[key] for key in ordered_ids],
    )

    reports: dict[str, list[dict[str, Any]]] = {}
    bindings: dict[str, str] = {}
    for controller in ("mcts", "random", "greedy"):
        reports[controller] = []
        for fold in range(3):
            path = args.reports_dir / f"{controller}_fold{fold}.json"
            value = json.loads(path.read_text())
            if (
                value.get("test_access") is not False
                or value.get("controller") != controller
                or int(value.get("outer_fold", -1)) != fold
                or value.get("mandatory_anchor_present") is not True
                or int(value.get("unique_workflows", -1)) != 128
                or int(value.get("selector_top_k", -1)) != 8
            ):
                raise RuntimeError(f"invalid controller fold report: {path}")
            reports[controller].append(value)
            bindings[str(path)] = sha256_file(path)

    controller_results: dict[str, dict[str, Any]] = {}
    subject_values_by_controller_margin: dict[tuple[str, float], dict[str, float]] = {}
    for controller, folds in reports.items():
        margins: dict[str, Any] = {}
        for margin in DEFAULT_ACCURACY_MARGINS:
            case_values: dict[str, float] = {}
            forward_counts: list[int] = []
            winners: list[dict[str, Any]] = []
            for fold in folds:
                selected = selected_for_margin(fold, margin)
                outer = selected["outer_evaluation"]
                for case_id, dice, forwards in zip(
                    outer["case_ids"],
                    outer["case_dice"],
                    outer["forward_counts"],
                    strict=True,
                ):
                    if case_id in case_values:
                        raise RuntimeError("controller outer folds overlap")
                    case_values[case_id] = float(dice)
                    forward_counts.append(int(forwards))
                winners.append(
                    {
                        "outer_fold": int(fold["outer_fold"]),
                        "workflow_id": selected["workflow_id"],
                        "spec": selected["spec"],
                        "selector_forwards": selected["selector_evaluation"]["mean_forwards"],
                        "outer_subject_macro_dsc": outer["subject_macro_dsc"],
                    }
                )
            if set(case_values) != set(locked_cases):
                raise RuntimeError("controller OOF cases differ from locked reference")
            aggregate = aggregate_scores(
                ordered_ids,
                [case_values[key] for key in ordered_ids],
            )
            subject_values_by_controller_margin[(controller, margin)] = aggregate["subject_values"]
            paired = bootstrap_delta(
                aggregate["subject_values"],
                locked_aggregate["subject_values"],
                seed=args.seed + int(round(margin * 1_000_000)),
                replicates=args.bootstrap_replicates,
            )
            margins[f"{margin:.3f}"] = {
                "accuracy_margin": margin,
                "candidate": aggregate,
                "locked_reference": locked_aggregate,
                "paired_subject_delta": paired,
                "mean_member_view_forwards": float(np.mean(forward_counts)),
                "maximum_member_view_forwards": max(forward_counts),
                "forward_reduction_fraction": 1.0 - float(np.mean(forward_counts)) / 24.0,
                "fold_winners": winners,
            }
        controller_results[controller] = {"margins": margins}

    pairwise: dict[str, Any] = {}
    for margin in DEFAULT_ACCURACY_MARGINS:
        for left, right in (("mcts", "random"), ("mcts", "greedy"), ("greedy", "random")):
            pairwise[f"{left}_minus_{right}_margin_{margin:.3f}"] = bootstrap_delta(
                subject_values_by_controller_margin[(left, margin)],
                subject_values_by_controller_margin[(right, margin)],
                seed=args.seed + 10_000 + int(round(margin * 1_000_000)),
                replicates=args.bootstrap_replicates,
            )

    report = {
        "schema": "aflow.sammed3d.t1ce_wt_mcts_efficiency_contribution.summary",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "split": "validation",
        "evidence_status": "development comparison on previously used validation; not independent confirmation",
        "geometry": "GT-centered N4 128^3 oracle geometry; not automatic localization",
        "controllers": controller_results,
        "pairwise_controller_deltas": pairwise,
        "protocol": {
            "outer_folds": 3,
            "subjects": 50,
            "scans": 108,
            "unique_workflows_per_fold": 128,
            "selector_candidates_per_fold": 8,
            "mandatory_locked_anchor": True,
            "accuracy_margins": list(DEFAULT_ACCURACY_MARGINS),
            "bootstrap_replicates": args.bootstrap_replicates,
            "seed": args.seed,
        },
        "bindings": {
            "historical_report": str(args.historical_report),
            "historical_report_sha256": sha256_file(args.historical_report),
            "fold_reports": bindings,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_json_atomic(args.output, report)
    concise = {
        controller: {
            margin: {
                "subject_dice": values["candidate"]["subject_macro_dsc"],
                "delta": values["paired_subject_delta"]["mean"],
                "ci95": values["paired_subject_delta"]["ci95"],
                "forwards": values["mean_member_view_forwards"],
            }
            for margin, values in result["margins"].items()
        }
        for controller, result in controller_results.items()
    }
    print(json.dumps(concise, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--historical-report", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260813)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
