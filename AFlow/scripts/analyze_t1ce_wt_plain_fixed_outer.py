#!/usr/bin/env python3
"""Apply the preregistered outer-selector gates to baseline and fixed-epoch reports."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from typing import Any

import numpy as np

from scripts.evaluate_t1ce_wt_plain_fixed_outer import (
    OUTER_SELECTOR_SHA256,
    SCHEMA as CANDIDATE_SCHEMA,
)
from scripts.train_t1ce_wt_capg3d import sha256_file, write_json_atomic


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_fixed_epoch_outer_decision"
BASELINE_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_locator_selector_reload"
BASELINE_REPORT_SHA256 = "9d1afe54a313a3cab370b56207cee9c0cf97d5bebbe4864a7d573a28c1983ee1"


def _subject_values(per_case: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in per_case:
        grouped[str(item["subject_id"])].append(float(item["locator_dsc"]))
    return {key: statistics.fmean(values) for key, values in grouped.items()}


def analyze(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    bootstrap_seed: int = 20260825,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    if baseline.get("schema") != BASELINE_SCHEMA:
        raise ValueError("unexpected baseline schema")
    if baseline.get("exact_report_equal") is not True or float(baseline.get("scan_dsc_difference", 1)) != 0:
        raise ValueError("baseline was not an exact checkpoint reload")
    if baseline.get("selector_manifest_sha256") != OUTER_SELECTOR_SHA256:
        raise ValueError("baseline selector manifest mismatch")
    if any(baseline.get(key) is not False for key in ("confirmation_access", "test_access")):
        raise ValueError("baseline accessed a forbidden role")
    if candidate.get("schema") != CANDIDATE_SCHEMA or candidate.get("role") != "outer_selector":
        raise ValueError("unexpected candidate report")
    if candidate.get("case_manifest", {}).get("sha256") != OUTER_SELECTOR_SHA256:
        raise ValueError("candidate selector manifest mismatch")
    if candidate.get("outer_selector_access") is not True:
        raise ValueError("candidate report did not identify outer-selector access")
    if any(candidate.get(key) is not False for key in ("confirmation_access", "test_access")):
        raise ValueError("candidate accessed a forbidden role")

    base_summary = baseline["reload_validation"]
    cand_summary = candidate["summary"]
    base_cases = {str(item["case_id"]): item for item in base_summary["per_case"]}
    cand_cases = {str(item["case_id"]): item for item in cand_summary["per_case"]}
    if base_cases.keys() != cand_cases.keys():
        raise ValueError("paired outer-selector case sets differ")
    base_subject = _subject_values(list(base_cases.values()))
    cand_subject = _subject_values(list(cand_cases.values()))
    if base_subject.keys() != cand_subject.keys():
        raise ValueError("paired outer-selector subject sets differ")

    case_ids = sorted(base_cases)
    subject_ids = sorted(base_subject)
    case_differences = np.asarray([
        float(cand_cases[key]["locator_dsc"]) - float(base_cases[key]["locator_dsc"])
        for key in case_ids
    ])
    subject_differences = np.asarray([
        cand_subject[key] - base_subject[key] for key in subject_ids
    ])
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = np.mean(
        subject_differences[
            rng.integers(0, len(subject_differences), size=(bootstrap_samples, len(subject_differences)))
        ],
        axis=1,
    )
    subject_ci = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    scan_difference = float(case_differences.mean())
    subject_difference = float(subject_differences.mean())
    point_difference = float(
        cand_summary["automatic_point_hit_rate"] - base_summary["automatic_point_hit_rate"]
    )
    gates = {
        "scan_gain_at_least_0.003": scan_difference >= 0.003,
        "subject_gain_at_least_0.003": subject_difference >= 0.003,
        "point_hit_drop_at_most_0.02": point_difference >= -0.02,
        "same_outer_role_and_forbidden_roles_sealed": True,
    }
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "outer_selector",
        "fixed_logit_threshold": 0.0,
        "baseline": {
            "epoch": int(baseline["selected_epoch"]),
            "scan": float(base_summary["locator_dsc_mean"]),
            "subject": float(base_summary["locator_subject_dsc_mean"]),
            "point_hit": float(base_summary["automatic_point_hit_rate"]),
            "checkpoint_sha256": baseline["checkpoint_sha256"],
        },
        "candidate": {
            "epoch": int(candidate["fixed_epoch"]),
            "scan": float(cand_summary["locator_dsc_mean"]),
            "subject": float(cand_summary["locator_subject_dsc_mean"]),
            "point_hit": float(cand_summary["automatic_point_hit_rate"]),
            "checkpoint": candidate["checkpoint"],
        },
        "paired_differences": {
            "scan": scan_difference,
            "subject": subject_difference,
            "point_hit": point_difference,
            "subject_95ci_descriptive": subject_ci,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        },
        "gates": gates,
        "all_outer_gates_passed": all(gates.values()),
        "candidate_status": (
            "development_candidate_passed_outer_selector"
            if all(gates.values())
            else "development_candidate_failed_outer_selector"
        ),
        "outer_selector_access": True,
        "confirmation_access": False,
        "test_access": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--expected-baseline-sha256", default=BASELINE_REPORT_SHA256)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.baseline) != args.expected_baseline_sha256:
        raise ValueError("baseline report SHA-256 mismatch")
    if args.expected_baseline_sha256 != BASELINE_REPORT_SHA256:
        raise ValueError("baseline report is not the frozen reference")
    if sha256_file(args.candidate) != args.expected_candidate_sha256:
        raise ValueError("candidate report SHA-256 mismatch")
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    result = analyze(baseline, candidate)
    result["input_reports"] = {
        "baseline": {"path": str(args.baseline.resolve()), "sha256": args.expected_baseline_sha256},
        "candidate": {"path": str(args.candidate.resolve()), "sha256": args.expected_candidate_sha256},
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite outer decision: {args.output}")
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
