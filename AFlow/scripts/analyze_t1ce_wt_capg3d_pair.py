#!/usr/bin/env python3
"""Paired subject-bootstrap analysis for the preregistered CAPG-3D contrast."""
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


SCHEMA = "aflow.sammed3d.t1ce_wt_capg3d_paired_analysis"


def subject_means(report: dict[str, Any], method: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for case in report["cases"]:
        grouped[str(case["subject_id"])].append(float(case["metrics"][method]["dsc"]))
    return {key: statistics.fmean(values) for key, values in grouped.items()}


def paired_result(
    plain: dict[str, Any], capg: dict[str, Any], method: str, *, seed: int, replicates: int
) -> dict[str, Any]:
    plain_cases = {str(case["case_id"]): case for case in plain["cases"]}
    capg_cases = {str(case["case_id"]): case for case in capg["cases"]}
    if list(plain_cases) != list(capg_cases):
        raise RuntimeError("plain/CAPG case membership or order differs")
    scan_differences = np.asarray(
        [
            float(capg_cases[key]["metrics"][method]["dsc"])
            - float(plain_cases[key]["metrics"][method]["dsc"])
            for key in plain_cases
        ],
        dtype=np.float64,
    )
    plain_subject = subject_means(plain, method)
    capg_subject = subject_means(capg, method)
    if list(plain_subject) != list(capg_subject):
        raise RuntimeError("plain/CAPG subject membership differs")
    keys = list(plain_subject)
    subject_differences = np.asarray(
        [capg_subject[key] - plain_subject[key] for key in keys], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    selections = rng.integers(0, len(keys), size=(replicates, len(keys)))
    bootstrap = subject_differences[selections].mean(axis=1)
    return {
        "method": method,
        "cases": len(scan_differences),
        "subjects": len(subject_differences),
        "plain_scan_dsc": float(plain["summary"][method]["scan_dsc_mean"]),
        "capg_scan_dsc": float(capg["summary"][method]["scan_dsc_mean"]),
        "capg_minus_plain_scan_dsc": float(scan_differences.mean()),
        "plain_subject_dsc": float(plain["summary"][method]["subject_dsc_mean"]),
        "capg_subject_dsc": float(capg["summary"][method]["subject_dsc_mean"]),
        "capg_minus_plain_subject_dsc": float(subject_differences.mean()),
        "subject_bootstrap_95ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "improved_cases": int(np.sum(scan_differences > 0)),
        "worsened_cases": int(np.sum(scan_differences < 0)),
        "tied_cases": int(np.sum(scan_differences == 0)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    plain = json.loads(args.plain.read_text())
    capg = json.loads(args.capg.read_text())
    if plain.get("test_access") is not False or capg.get("test_access") is not False:
        raise RuntimeError("paired analysis refuses test-accessible reports")
    if plain.get("arm") != "plain" or capg.get("arm") != "capg":
        raise RuntimeError("arm identities are invalid")
    if plain["sam"]["checkpoint_sha256"] != capg["sam"]["checkpoint_sha256"]:
        raise RuntimeError("SAM checkpoint differs between paired arms")
    if plain["validation_cache"]["index_sha256"] != capg["validation_cache"]["index_sha256"]:
        raise RuntimeError("validation cache differs between paired arms")
    results = {
        method: paired_result(
            plain, capg, method, seed=args.seed + offset, replicates=args.replicates
        )
        for offset, method in enumerate(
            ("locator", "sam_point_only", "sam_dense_only", "sam_point_dense")
        )
    }
    primary = results["sam_point_dense"]
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "primary_method": "sam_point_dense",
        "preregistered_success_threshold_scan_dsc": args.success_threshold,
        "passes_preregistered_success_threshold": (
            primary["capg_minus_plain_scan_dsc"] >= args.success_threshold
        ),
        "interpretation": (
            "positive architecture candidate"
            if primary["capg_minus_plain_scan_dsc"] >= args.success_threshold
            else "did not pass the preregistered architecture threshold"
        ),
        "results": results,
        "point_hit_rate": {
            "plain": plain["summary"]["automatic_point_hit_rate"],
            "capg": capg["summary"]["automatic_point_hit_rate"],
            "capg_minus_plain": (
                capg["summary"]["automatic_point_hit_rate"]
                - plain["summary"]["automatic_point_hit_rate"]
            ),
        },
        "bootstrap": {"unit": "subject", "replicates": args.replicates, "seed": args.seed},
        "inputs": {
            "plain": {"path": str(args.plain.resolve()), "sha256": sha256_file(args.plain)},
            "capg": {"path": str(args.capg.resolve()), "sha256": sha256_file(args.capg)},
        },
    }
    write_json_atomic(args.output.resolve(), report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain", type=Path, required=True)
    parser.add_argument("--capg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--success-threshold", type=float, default=0.005)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
