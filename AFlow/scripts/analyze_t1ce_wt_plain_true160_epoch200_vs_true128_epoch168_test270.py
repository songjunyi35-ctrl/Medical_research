#!/usr/bin/env python3
"""Descriptively pair true-160 epoch200 against historical true-128 epoch168."""
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


SCHEMA = "aflow.sammed3d.t1ce_wt_true160_epoch200_vs_true128_epoch168_test270"
BASELINE_SHA256 = "91433c4b6b88b233d827e41b0d038f459c25ef41dec224c84e154e90f370bb8c"
CANDIDATE_SHA256 = "973e4cc452e4ea7b9de3a190d5ad293724f2cd94e0a82f861f521a2e15477f35"
BOOTSTRAP_SEED = 20260832
BOOTSTRAP_SAMPLES = 10_000


def _subject_values(per_case: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in per_case:
        grouped[str(item["subject_id"])].append(float(item["locator_dsc"]))
    return {subject: statistics.fmean(values) for subject, values in grouped.items()}


def _ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    return [float(value) for value in np.quantile(values[draws].mean(axis=1), [0.025, 0.975])]


def analyze(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if baseline.get("schema") != "aflow.sammed3d.t1ce_wt_plain_fixed_epoch168_test270":
        raise ValueError("unexpected historical true-128 epoch168 report")
    if candidate.get("schema") != "aflow.sammed3d.t1ce_wt_plain_true160_epoch200_test270":
        raise ValueError("unexpected true-160 epoch200 report")
    if baseline.get("test_access") is not True or candidate.get("test_access") is not True:
        raise ValueError("both inputs must be authorized test reports")
    if candidate.get("confirmation_access") is not False:
        raise ValueError("candidate accessed confirmation")
    if baseline["data"]["manifest_sha256"] != candidate["data"]["manifest_sha256"]:
        raise ValueError("test manifest differs between reports")
    old_cases = {
        str(item["case_id"]): item for item in baseline["summary"]["per_case"]
    }
    new_cases = {
        str(item["case_id"]): item for item in candidate["summary"]["per_case"]
    }
    if old_cases.keys() != new_cases.keys():
        raise ValueError("paired case sets differ")
    old_subjects = _subject_values(list(old_cases.values()))
    new_subjects = _subject_values(list(new_cases.values()))
    if old_subjects.keys() != new_subjects.keys():
        raise ValueError("paired subject sets differ")
    case_ids = sorted(old_cases)
    subject_ids = sorted(old_subjects)
    case_delta = np.asarray([
        float(new_cases[key]["locator_dsc"]) - float(old_cases[key]["locator_dsc"])
        for key in case_ids
    ])
    subject_delta = np.asarray([
        new_subjects[key] - old_subjects[key] for key in subject_ids
    ])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    transitions = {"hit_to_hit": 0, "hit_to_miss": 0, "miss_to_hit": 0, "miss_to_miss": 0}
    for key in case_ids:
        old_hit = bool(old_cases[key]["automatic_point_hits_gt"])
        new_hit = bool(new_cases[key]["automatic_point_hits_gt"])
        transitions[("hit" if old_hit else "miss") + "_to_" + ("hit" if new_hit else "miss")] += 1
    scan_delta = float(case_delta.mean())
    subject_delta_mean = float(subject_delta.mean())
    point_delta = float(
        candidate["summary"]["automatic_point_hit_rate"]
        - baseline["summary"]["automatic_point_hit_rate"]
    )
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "descriptive paired comparison on a reused internal test set",
        "not_pristine_blind_test": True,
        "baseline": {
            "name": "true-128^3 full756 epoch168",
            "scan": float(baseline["summary"]["locator_dsc_mean"]),
            "subject": float(baseline["summary"]["locator_subject_dsc_mean"]),
            "point_hit": float(baseline["summary"]["automatic_point_hit_rate"]),
            "checkpoint_sha256": baseline["checkpoint"]["sha256"]
        },
        "candidate": {
            "name": "true-160^3 full756 epoch200",
            "scan": float(candidate["summary"]["locator_dsc_mean"]),
            "subject": float(candidate["summary"]["locator_subject_dsc_mean"]),
            "point_hit": float(candidate["summary"]["automatic_point_hit_rate"]),
            "checkpoint_sha256": candidate["checkpoint"]["sha256"]
        },
        "candidate_minus_baseline": {
            "scan": scan_delta,
            "subject": subject_delta_mean,
            "point_hit": point_delta,
            "scan_95ci": _ci(case_delta, rng),
            "subject_95ci": _ci(subject_delta, rng),
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED
        },
        "distribution": {
            "cases_improved": int(np.sum(case_delta > 0)),
            "cases_tied": int(np.sum(case_delta == 0)),
            "cases_worse": int(np.sum(case_delta < 0)),
            "subjects_improved": int(np.sum(subject_delta > 0)),
            "subjects_tied": int(np.sum(subject_delta == 0)),
            "subjects_worse": int(np.sum(subject_delta < 0)),
            "median_case_delta": float(np.median(case_delta)),
            "median_subject_delta": float(np.median(subject_delta)),
            "point_transitions": transitions
        },
        "candidate_has_higher_scan_and_subject_point_estimates": (
            scan_delta > 0.0 and subject_delta_mean > 0.0
        ),
        "interpretation": (
            "The true-160 epoch200 candidate did not improve the reused internal test point "
            "estimate over true-128 epoch168. Cross-resolution Dice deltas also include "
            "nearest-exact label rasterization differences between the 128 and 160 grids."
        ),
        "result_will_not_trigger_tuning": True,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": True
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.baseline) != BASELINE_SHA256:
        raise ValueError("historical true-128 report SHA-256 mismatch")
    if sha256_file(args.candidate) != CANDIDATE_SHA256:
        raise ValueError("true-160 report SHA-256 mismatch")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = analyze(
        json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.candidate.read_text(encoding="utf-8"))
    )
    result["inputs"] = {
        "baseline": {"path": str(args.baseline.resolve()), "sha256": BASELINE_SHA256},
        "candidate": {"path": str(args.candidate.resolve()), "sha256": CANDIDATE_SHA256}
    }
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
