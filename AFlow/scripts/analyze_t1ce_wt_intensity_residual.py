#!/usr/bin/env python3
"""Apply preregistered paired gates to three inner-tune reload reports."""
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


SCHEMA = "aflow.sammed3d.t1ce_wt_intensity_residual_inner_decision"
BOOTSTRAP_SEED = 20260820
BOOTSTRAP_SAMPLES = 10_000


def load_report(path: Path, expected_arm: str) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if report.get("arm") != expected_arm or report.get("role") != "inner_tune":
        raise RuntimeError(f"unexpected arm/role in {path}")
    if report.get("embedded_reload_max_abs_difference") != 0.0:
        raise RuntimeError(f"reload was not exact for {path}")
    return report


def paired_values(left: dict[str, Any], right: dict[str, Any]) -> tuple[list[float], list[float]]:
    left_cases = {case["case_id"]: case for case in left["summary"]["per_case"]}
    right_cases = {case["case_id"]: case for case in right["summary"]["per_case"]}
    if set(left_cases) != set(right_cases):
        raise RuntimeError("paired reports contain different cases")
    scan = [float(left_cases[key]["dsc"]) - float(right_cases[key]["dsc"]) for key in sorted(left_cases)]
    by_subject: dict[str, list[float]] = defaultdict(list)
    for key in left_cases:
        if left_cases[key]["subject_id"] != right_cases[key]["subject_id"]:
            raise RuntimeError("paired subject identity mismatch")
        by_subject[str(left_cases[key]["subject_id"])].append(
            float(left_cases[key]["dsc"]) - float(right_cases[key]["dsc"])
        )
    subject = [statistics.fmean(values) for _, values in sorted(by_subject.items())]
    return scan, subject


def interval(values: list[float], rng: np.random.Generator) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_SAMPLES, len(array)))
    means = array[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--intensity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite inner decision: {args.output}")
    reports = {
        "plain": load_report(args.plain, "plain"),
        "residual_control": load_report(args.control, "residual_control"),
        "intensity_residual": load_report(args.intensity, "intensity_residual"),
    }
    role_hashes = {report["case_role"]["sha256"] for report in reports.values()}
    if len(role_hashes) != 1:
        raise RuntimeError("inner-tune reports use different role manifests")
    summaries = {key: value["summary"] for key, value in reports.items()}
    scan_ic, subject_ic = paired_values(reports["intensity_residual"], reports["residual_control"])
    scan_ip, subject_ip = paired_values(reports["intensity_residual"], reports["plain"])
    delta_ic = {
        "scan": statistics.fmean(scan_ic), "subject": statistics.fmean(subject_ic),
        "point_hit": summaries["intensity_residual"]["automatic_point_hit_rate"]
        - summaries["residual_control"]["automatic_point_hit_rate"],
    }
    delta_ip = {
        "scan": statistics.fmean(scan_ip), "subject": statistics.fmean(subject_ip),
        "point_hit": summaries["intensity_residual"]["automatic_point_hit_rate"]
        - summaries["plain"]["automatic_point_hit_rate"],
    }
    gates = {
        "intensity_minus_control_scan_at_least_0.003": delta_ic["scan"] >= 0.003,
        "intensity_minus_control_subject_at_least_0.003": delta_ic["subject"] >= 0.003,
        "intensity_not_worse_plain_scan_beyond_0.001": delta_ip["scan"] >= -0.001,
        "intensity_not_worse_plain_subject_beyond_0.001": delta_ip["subject"] >= -0.001,
        "intensity_control_point_hit_drop_at_most_0.02": delta_ic["point_hit"] >= -0.02,
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "confirmation_access": False,
        "outer_selector_access": False,
        "fixed_logit_threshold": 0.0,
        "arms": {
            key: {
                "scan_dsc_mean": summary["scan_dsc_mean"],
                "subject_dsc_mean": summary["subject_dsc_mean"],
                "automatic_point_hit_rate": summary["automatic_point_hit_rate"],
                "checkpoint": reports[key]["checkpoint"],
                "reload_report_sha256": sha256_file(path),
            }
            for (key, summary), path in zip(summaries.items(), (args.plain, args.control, args.intensity))
        },
        "paired_differences": {
            "intensity_minus_control": delta_ic,
            "intensity_minus_plain": delta_ip,
            "intensity_minus_control_scan_95ci": interval(scan_ic, rng),
            "intensity_minus_control_subject_95ci": interval(subject_ic, rng),
        },
        "gates": gates,
        "all_inner_gates_passed": all(gates.values()),
        "selected_arm": "intensity_residual" if all(gates.values()) else None,
        "outer_selector_authorized": all(gates.values()),
        "inner_tune_manifest_sha256": role_hashes.pop(),
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
