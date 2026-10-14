#!/usr/bin/env python3
"""Apply the preregistered inner gates to fixed-epoch 128^3 vs 160^3 runs."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from typing import Any

import numpy as np

from scripts.train_t1ce_wt_capg3d import sha256_file, write_json_atomic


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_resolution_inner_decision"


def subject_values(per_case: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in per_case:
        grouped[str(item["subject_id"])].append(float(item["locator_dsc"]))
    return {subject: statistics.fmean(values) for subject, values in grouped.items()}


def _validate_report(report: dict[str, Any], *, resolution: int, arm: str) -> None:
    if report.get("status") != "complete":
        raise ValueError(f"{arm} training report is incomplete")
    if report.get("resolution") != resolution or report.get("arm") != arm:
        raise ValueError(f"expected {resolution} {arm} report")
    if report.get("fixed_epoch") != 168 or report.get("fixed_logit_threshold") != 0.0:
        raise ValueError("comparison requires fixed epoch168 and raw logit > 0")
    if report.get("inner_tune_first_access") != "after_final_optimizer_step":
        raise ValueError("inner tune was not sealed until the final optimizer step")
    if any(
        report.get(key) is not False
        for key in ("outer_selector_access", "confirmation_access", "test_access")
    ):
        raise ValueError("report declares forbidden data access")
    if float(report.get("reload_max_abs_difference", 1.0)) != 0.0:
        raise ValueError("checkpoint reload is not exact")


def analyze(
    control: dict[str, Any],
    candidate: dict[str, Any],
    *,
    bootstrap_seed: int = 20260830,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    _validate_report(control, resolution=128, arm="128_control")
    _validate_report(candidate, resolution=160, arm="160_candidate")
    for key in (
        "fit",
        "inner_tune",
        "fit_cases",
        "fit_subjects",
        "inner_tune_cases",
        "inner_tune_subjects",
        "case_overlap",
        "subject_overlap",
    ):
        if control["data_roles"][key] != candidate["data_roles"][key]:
            raise ValueError(f"data role mismatch: {key}")
    if control["parameter_count"] != candidate["parameter_count"]:
        raise ValueError("parameter count differs between arms")
    if control["sampling_contract"] != candidate["sampling_contract"]:
        raise ValueError("sampler weights/order contract differs between arms")
    if control["schedule"] != candidate["schedule"]:
        raise ValueError("learning-rate schedule differs between arms")
    matched_hyperparameters = (
        "seed",
        "epochs",
        "schedule_total_epochs",
        "constant_epochs",
        "batch_size",
        "validation_batch_size",
        "num_workers",
        "init_filters",
        "dropout",
        "base_lr",
        "min_lr",
        "weight_decay",
        "max_shift",
        "ce_weight",
        "expected_parameter_count",
        "expected_cohort_sha256",
        "expected_source_positive_voxels_sha256",
    )
    for key in matched_hyperparameters:
        if control["hyperparameters"][key] != candidate["hyperparameters"][key]:
            raise ValueError(f"matched hyperparameter differs: {key}")

    control_cases = {
        str(item["case_id"]): item for item in control["validation"]["per_case"]
    }
    candidate_cases = {
        str(item["case_id"]): item for item in candidate["validation"]["per_case"]
    }
    if control_cases.keys() != candidate_cases.keys():
        raise ValueError("paired inner-tune case sets differ")
    control_subjects = subject_values(list(control_cases.values()))
    candidate_subjects = subject_values(list(candidate_cases.values()))
    if control_subjects.keys() != candidate_subjects.keys():
        raise ValueError("paired inner-tune subject sets differ")

    case_ids = sorted(control_cases)
    subject_ids = sorted(control_subjects)
    scan_differences = np.asarray(
        [
            float(candidate_cases[case_id]["locator_dsc"])
            - float(control_cases[case_id]["locator_dsc"])
            for case_id in case_ids
        ],
        dtype=np.float64,
    )
    subject_differences = np.asarray(
        [
            candidate_subjects[subject_id] - control_subjects[subject_id]
            for subject_id in subject_ids
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = np.mean(
        subject_differences[
            rng.integers(
                0,
                len(subject_differences),
                size=(bootstrap_samples, len(subject_differences)),
            )
        ],
        axis=1,
    )
    subject_ci = [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])]
    scan_gain = float(scan_differences.mean())
    subject_gain = float(subject_differences.mean())
    point_gain = float(
        candidate["validation"]["automatic_point_hit_rate"]
        - control["validation"]["automatic_point_hit_rate"]
    )
    gates = {
        "scan_gain_at_least_0.005": scan_gain >= 0.005,
        "subject_gain_at_least_0.005": subject_gain >= 0.005,
        "paired_subject_95ci_lower_nonnegative": subject_ci[0] >= 0.0,
        "point_hit_drop_at_most_0.02": point_gain >= -0.02,
        "exact_checkpoint_reloads": True,
        "matched_single_factor_contract": True,
    }
    transitions = {
        "control_hit_candidate_hit": 0,
        "control_hit_candidate_miss": 0,
        "control_miss_candidate_hit": 0,
        "control_miss_candidate_miss": 0,
    }
    for case_id in case_ids:
        left = bool(control_cases[case_id]["automatic_point_hits_gt"])
        right = bool(candidate_cases[case_id]["automatic_point_hits_gt"])
        key = (
            f"control_{'hit' if left else 'miss'}_"
            f"candidate_{'hit' if right else 'miss'}"
        )
        transitions[key] += 1
    passed = all(gates.values())
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": "160^3 candidate minus 128^3 control",
        "fixed_epoch": 168,
        "fixed_logit_threshold": 0.0,
        "control": {
            "resolution": 128,
            "scan": float(control["validation"]["locator_dsc_mean"]),
            "subject": float(control["validation"]["locator_subject_dsc_mean"]),
            "point_hit": float(control["validation"]["automatic_point_hit_rate"]),
            "checkpoint": control["checkpoint"],
            "reload_max_abs_difference": control["reload_max_abs_difference"],
        },
        "candidate": {
            "resolution": 160,
            "scan": float(candidate["validation"]["locator_dsc_mean"]),
            "subject": float(candidate["validation"]["locator_subject_dsc_mean"]),
            "point_hit": float(candidate["validation"]["automatic_point_hit_rate"]),
            "checkpoint": candidate["checkpoint"],
            "reload_max_abs_difference": candidate["reload_max_abs_difference"],
        },
        "paired_differences": {
            "scan": scan_gain,
            "subject": subject_gain,
            "point_hit": point_gain,
            "subject_95ci": subject_ci,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "scans_improved": int(np.sum(scan_differences > 0)),
            "scans_declined": int(np.sum(scan_differences < 0)),
            "scans_tied": int(np.sum(scan_differences == 0)),
            "subjects_improved": int(np.sum(subject_differences > 0)),
            "subjects_declined": int(np.sum(subject_differences < 0)),
            "subjects_tied": int(np.sum(subject_differences == 0)),
            "point_transitions": transitions,
        },
        "gates": gates,
        "all_inner_gates_passed": passed,
        "decision": (
            "160_positive_on_preregistered_inner_gates"
            if passed
            else "160_negative_stop_without_scale_or_threshold_tuning"
        ),
        "further_fixed_checkpoint_evaluation_performed": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260830)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    for path in (args.control, args.candidate, args.output):
        lowered = str(path.resolve()).lower()
        if any(token in lowered for token in ("selector", "confirmation", "sealed", "test270")):
            raise ValueError(f"forbidden data role in path: {path}")
    control = json.loads(args.control.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = analyze(
        control,
        candidate,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    result["inputs"] = {
        "control": {"path": str(args.control.resolve()), "sha256": sha256_file(args.control)},
        "candidate": {
            "path": str(args.candidate.resolve()),
            "sha256": sha256_file(args.candidate),
        },
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite decision: {args.output}")
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
