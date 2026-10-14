#!/usr/bin/env python3
"""Pair the epoch-168 test270 result against the saved epoch-99 test report."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from typing import Any

import numpy as np

from scripts.evaluate_t1ce_wt_plain_fixed_test270 import (
    CHECKPOINT_SHA256,
    SCHEMA as CANDIDATE_SCHEMA,
    TEST_MANIFEST_SHA256,
)
from scripts.train_t1ce_wt_capg3d import sha256_file, write_json_atomic


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_epoch168_vs_epoch99_test270"
BASELINE_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_locator_test270"
BASELINE_REPORT_SHA256 = "43b88933337e0325caaf97b6add8fbaf64a60b2574d75acaaedc9dceaea7e78b"
BASELINE_CHECKPOINT_SHA256 = "0bb7ca52d7ac30068f752a4fc05a859f4dce42dce6ace40ba1f6b8b332889c2f"
BOOTSTRAP_SEED = 20260827
BOOTSTRAP_SAMPLES = 10_000


def _subject_values(per_case: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in per_case:
        grouped[str(item["subject_id"])].append(float(item["locator_dsc"]))
    return {key: statistics.fmean(values) for key, values in grouped.items()}


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, samples: int) -> list[float]:
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    return [float(x) for x in np.quantile(values[draws].mean(axis=1), [0.025, 0.975])]


def analyze(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    if baseline.get("schema") != BASELINE_SCHEMA or baseline.get("test_access") is not True:
        raise ValueError("unexpected epoch99 test report")
    if baseline.get("checkpoint", {}).get("sha256") != BASELINE_CHECKPOINT_SHA256:
        raise ValueError("unexpected epoch99 checkpoint")
    if baseline.get("data", {}).get("manifest_sha256") != TEST_MANIFEST_SHA256:
        raise ValueError("epoch99 test manifest mismatch")
    if candidate.get("schema") != CANDIDATE_SCHEMA or candidate.get("test_access") is not True:
        raise ValueError("unexpected epoch168 test report")
    if candidate.get("checkpoint", {}).get("sha256") != CHECKPOINT_SHA256:
        raise ValueError("unexpected epoch168 checkpoint")
    if candidate.get("data", {}).get("manifest_sha256") != TEST_MANIFEST_SHA256:
        raise ValueError("epoch168 test manifest mismatch")
    if candidate.get("confirmation_access") is not False:
        raise ValueError("candidate accessed confirmation")

    old_summary = baseline["summary"]
    new_summary = candidate["summary"]
    old_cases = {str(item["case_id"]): item for item in old_summary["per_case"]}
    new_cases = {str(item["case_id"]): item for item in new_summary["per_case"]}
    if old_cases.keys() != new_cases.keys():
        raise ValueError("paired test case sets differ")
    old_subject = _subject_values(list(old_cases.values()))
    new_subject = _subject_values(list(new_cases.values()))
    if old_subject.keys() != new_subject.keys():
        raise ValueError("paired test subject sets differ")

    case_ids = sorted(old_cases)
    subject_ids = sorted(old_subject)
    case_delta = np.asarray([
        float(new_cases[key]["locator_dsc"]) - float(old_cases[key]["locator_dsc"])
        for key in case_ids
    ])
    subject_delta = np.asarray([
        new_subject[key] - old_subject[key] for key in subject_ids
    ])
    rng = np.random.default_rng(bootstrap_seed)
    scan_ci = _bootstrap_ci(case_delta, rng, bootstrap_samples)
    subject_ci = _bootstrap_ci(subject_delta, rng, bootstrap_samples)
    point_delta = float(
        new_summary["automatic_point_hit_rate"] - old_summary["automatic_point_hit_rate"]
    )
    transitions = {"hit_to_hit": 0, "hit_to_miss": 0, "miss_to_hit": 0, "miss_to_miss": 0}
    for key in case_ids:
        old_hit = bool(old_cases[key]["automatic_point_hits_gt"])
        new_hit = bool(new_cases[key]["automatic_point_hits_gt"])
        name = ("hit" if old_hit else "miss") + "_to_" + ("hit" if new_hit else "miss")
        transitions[name] += 1

    scan_delta = float(case_delta.mean())
    subject_delta_mean = float(subject_delta.mean())
    history_high = scan_delta > 0.0 and subject_delta_mean > 0.0
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "paired comparison on a reused internal test set",
        "not_pristine_blind_test": True,
        "baseline": {
            "epoch": 99,
            "scan": float(old_summary["locator_dsc_mean"]),
            "subject": float(old_summary["locator_subject_dsc_mean"]),
            "point_hit": float(old_summary["automatic_point_hit_rate"]),
            "checkpoint_sha256": BASELINE_CHECKPOINT_SHA256,
        },
        "candidate": {
            "epoch": 168,
            "scan": float(new_summary["locator_dsc_mean"]),
            "subject": float(new_summary["locator_subject_dsc_mean"]),
            "point_hit": float(new_summary["automatic_point_hit_rate"]),
            "checkpoint_sha256": CHECKPOINT_SHA256,
        },
        "paired_differences": {
            "scan": scan_delta,
            "subject": subject_delta_mean,
            "point_hit": point_delta,
            "scan_95ci": scan_ci,
            "subject_95ci": subject_ci,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        },
        "distribution": {
            "cases_improved": int((case_delta > 0).sum()),
            "cases_tied": int((case_delta == 0).sum()),
            "cases_worse": int((case_delta < 0).sum()),
            "subjects_improved": int((subject_delta > 0).sum()),
            "subjects_tied": int((subject_delta == 0).sum()),
            "subjects_worse": int((subject_delta < 0).sum()),
            "median_case_delta": float(np.median(case_delta)),
            "median_subject_delta": float(np.median(subject_delta)),
            "point_transitions": transitions,
        },
        "new_internal_test_point_estimate_high": history_high,
        "paired_scan_ci_lower_positive": float(scan_ci[0]) > 0.0,
        "paired_subject_ci_lower_positive": float(subject_ci[0]) > 0.0,
        "result_will_not_trigger_tuning": True,
        "confirmation_access": False,
        "test_access": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.baseline) != BASELINE_REPORT_SHA256:
        raise ValueError("epoch99 baseline report SHA-256 mismatch")
    if sha256_file(args.candidate) != args.expected_candidate_sha256:
        raise ValueError("epoch168 candidate report SHA-256 mismatch")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite test270 comparison")
    result = analyze(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
    result["input_reports"] = {
        "baseline": {"path": str(args.baseline.resolve()), "sha256": BASELINE_REPORT_SHA256},
        "candidate": {"path": str(args.candidate.resolve()), "sha256": args.expected_candidate_sha256},
    }
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
