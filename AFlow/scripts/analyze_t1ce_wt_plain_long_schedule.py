#!/usr/bin/env python3
"""Apply frozen paired gates to pre/post-100 strict reload reports."""
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


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_long_schedule_inner_decision"


def _subject_values(per_case: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in per_case:
        grouped[str(item["subject_id"])].append(float(item["locator_dsc"]))
    return {key: statistics.fmean(values) for key, values in grouped.items()}


def analyze(
    pre: dict[str, Any],
    post: dict[str, Any],
    *,
    bootstrap_seed: int = 20260824,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    if pre.get("window") != "pre100" or post.get("window") != "post100":
        raise ValueError("expected pre100/post100 reports")
    for report in (pre, post):
        if report.get("role") != "fresh_tune":
            raise ValueError("analysis accepts only fresh_tune reports")
        if float(report.get("reload_max_abs_difference", 1.0)) != 0.0:
            raise ValueError("reload report is not exact")
        if any(report.get(key) is not False for key in (
            "outer_selector_access", "confirmation_access", "test_access"
        )):
            raise ValueError("access flags are not sealed")
    pre_cases = {str(x["case_id"]): x for x in pre["summary"]["per_case"]}
    post_cases = {str(x["case_id"]): x for x in post["summary"]["per_case"]}
    if pre_cases.keys() != post_cases.keys():
        raise ValueError("paired case sets differ")
    pre_subject = _subject_values(list(pre_cases.values()))
    post_subject = _subject_values(list(post_cases.values()))
    if pre_subject.keys() != post_subject.keys():
        raise ValueError("paired subject sets differ")

    case_ids = sorted(pre_cases)
    subject_ids = sorted(pre_subject)
    case_differences = np.asarray([
        float(post_cases[key]["locator_dsc"]) - float(pre_cases[key]["locator_dsc"])
        for key in case_ids
    ])
    subject_differences = np.asarray([
        post_subject[key] - pre_subject[key] for key in subject_ids
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
        post["summary"]["automatic_point_hit_rate"]
        - pre["summary"]["automatic_point_hit_rate"]
    )
    gates = {
        "scan_gain_at_least_0.003": scan_difference >= 0.003,
        "subject_gain_at_least_0.003": subject_difference >= 0.003,
        "subject_95ci_lower_nonnegative": float(subject_ci[0]) >= 0.0,
        "point_hit_drop_at_most_0.02": point_difference >= -0.02,
        "exact_reloads_and_sealed_access": True,
    }
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixed_logit_threshold": 0.0,
        "pre100": {
            "epoch": int(pre["epoch"]),
            "scan": float(pre["summary"]["locator_dsc_mean"]),
            "subject": float(pre["summary"]["locator_subject_dsc_mean"]),
            "point_hit": float(pre["summary"]["automatic_point_hit_rate"]),
            "checkpoint": pre["checkpoint"],
        },
        "post100": {
            "epoch": int(post["epoch"]),
            "scan": float(post["summary"]["locator_dsc_mean"]),
            "subject": float(post["summary"]["locator_subject_dsc_mean"]),
            "point_hit": float(post["summary"]["automatic_point_hit_rate"]),
            "checkpoint": post["checkpoint"],
        },
        "paired_differences": {
            "scan": scan_difference,
            "subject": subject_difference,
            "point_hit": point_difference,
            "subject_95ci": subject_ci,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        },
        "gates": gates,
        "all_inner_gates_passed": all(gates.values()),
        "outer_selector_authorized": all(gates.values()),
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre100", type=Path, required=True)
    parser.add_argument("--post100", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pre = json.loads(args.pre100.read_text())
    post = json.loads(args.post100.read_text())
    result = analyze(pre, post)
    result["input_reports"] = {
        "pre100": {"path": str(args.pre100.resolve()), "sha256": sha256_file(args.pre100)},
        "post100": {"path": str(args.post100.resolve()), "sha256": sha256_file(args.post100)},
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite decision: {args.output}")
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
