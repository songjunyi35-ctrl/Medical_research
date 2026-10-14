#!/usr/bin/env python3
"""Fail-closed paired analysis for matched T1ce/WT validation reports.

The two inputs must be complete reports produced by the same reload evaluator
on the same validation manifest, cases, subjects, protocol, and RNG seed.  The
analysis is deliberately derived from per-case DSC values rather than trusting
the aggregate differences recorded in either input report.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

import numpy as np


REPORT_SCHEMA = "aflow.sammed3d.t1ce_wt_checkpoint_evaluation"
OUTPUT_SCHEMA = "aflow.sammed3d.t1ce_wt_paired_validation_analysis"
OUTPUT_SCHEMA_VERSION = 1
CLICK_KEYS = ("1", "2", "3")
SUMMARY_ABS_TOLERANCE = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _require_mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{location} must be an object")
    return value


def _require_exact_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{location} must be an integer")
    return value


def _require_dsc(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        _fail(f"{location} must be finite and within [0,1]")
    return result


def _require_sha256(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        _fail(f"{location} must be a 64-character SHA-256 hex digest")
    return value.lower()


def _require_close(actual: Any, expected: float, location: str) -> None:
    value = _require_dsc(actual, location)
    if not math.isclose(
        value,
        expected,
        rel_tol=0.0,
        abs_tol=SUMMARY_ABS_TOLERANCE,
    ):
        _fail(f"{location} is inconsistent with per-case metrics")


def _audit_report(
    report: Any,
    *,
    name: str,
    expected_cases: int,
) -> dict[str, Any]:
    report = _require_mapping(report, name)
    if report.get("schema") != REPORT_SCHEMA:
        _fail(f"{name}.schema is not {REPORT_SCHEMA!r}")
    schema_version = _require_exact_int(
        report.get("schema_version"), f"{name}.schema_version"
    )
    if schema_version <= 0:
        _fail(f"{name}.schema_version must be positive")

    protocol = _require_mapping(report.get("protocol"), f"{name}.protocol")
    if protocol.get("num_clicks") != len(CLICK_KEYS):
        _fail(f"{name}.protocol.num_clicks must be {len(CLICK_KEYS)}")

    data = _require_mapping(report.get("data"), f"{name}.data")
    if data.get("expected_split") != "validation":
        _fail(f"{name} must be a validation report")
    manifest_sha256 = _require_sha256(
        data.get("manifest_sha256"), f"{name}.data.manifest_sha256"
    )
    selected_cases = _require_exact_int(
        data.get("selected_cases"), f"{name}.data.selected_cases"
    )
    full_manifest_cases = _require_exact_int(
        data.get("full_manifest_cases"), f"{name}.data.full_manifest_cases"
    )
    if selected_cases != expected_cases:
        _fail(
            f"{name} has {selected_cases} selected cases; expected {expected_cases}"
        )
    if full_manifest_cases != expected_cases:
        _fail(
            f"{name} is not a complete {expected_cases}-case validation report"
        )

    seed = _require_exact_int(report.get("seed"), f"{name}.seed")
    raw_cases = report.get("cases")
    if not isinstance(raw_cases, list):
        _fail(f"{name}.cases must be an array")
    if len(raw_cases) != expected_cases:
        _fail(f"{name}.cases has {len(raw_cases)} rows; expected {expected_cases}")

    cases: dict[str, dict[str, Any]] = {}
    for index, raw_case in enumerate(raw_cases):
        location = f"{name}.cases[{index}]"
        case = _require_mapping(raw_case, location)
        case_id = case.get("case_id")
        subject_id = case.get("subject_id")
        if not isinstance(case_id, str) or not case_id:
            _fail(f"{location}.case_id must be a non-empty string")
        if not isinstance(subject_id, str) or not subject_id:
            _fail(f"{location}.subject_id must be a non-empty string")
        if case_id in cases:
            _fail(f"{name} contains duplicate case_id {case_id!r}")
        metrics = _require_mapping(case.get("metrics"), f"{location}.metrics")
        dsc = {}
        for click in CLICK_KEYS:
            click_metrics = _require_mapping(
                metrics.get(click), f"{location}.metrics[{click!r}]"
            )
            dsc[click] = _require_dsc(
                click_metrics.get("dsc"),
                f"{location}.metrics[{click!r}].dsc",
            )
        cases[case_id] = {"subject_id": subject_id, "dsc": dsc}

    summary = _require_mapping(report.get("summary"), f"{name}.summary")
    for click in CLICK_KEYS:
        click_summary = _require_mapping(
            summary.get(click), f"{name}.summary[{click!r}]"
        )
        scan_macro = _require_mapping(
            click_summary.get("scan_macro"),
            f"{name}.summary[{click!r}].scan_macro",
        )
        if (
            _require_exact_int(
                scan_macro.get("cases"),
                f"{name}.summary[{click!r}].scan_macro.cases",
            )
            != expected_cases
        ):
            _fail(f"{name}.summary[{click!r}] has the wrong case count")
        per_case = [case["dsc"][click] for case in cases.values()]
        _require_close(
            scan_macro.get("dsc_mean"),
            statistics.fmean(per_case),
            f"{name}.summary[{click!r}].scan_macro.dsc_mean",
        )

        by_subject: dict[str, list[float]] = defaultdict(list)
        for case in cases.values():
            by_subject[case["subject_id"]].append(case["dsc"][click])
        subject_means = [
            statistics.fmean(values) for values in by_subject.values()
        ]
        subject_balanced = _require_mapping(
            click_summary.get("subject_balanced"),
            f"{name}.summary[{click!r}].subject_balanced",
        )
        if (
            _require_exact_int(
                subject_balanced.get("subjects"),
                f"{name}.summary[{click!r}].subject_balanced.subjects",
            )
            != len(subject_means)
        ):
            _fail(f"{name}.summary[{click!r}] has the wrong subject count")
        _require_close(
            subject_balanced.get("dsc_mean"),
            statistics.fmean(subject_means),
            f"{name}.summary[{click!r}].subject_balanced.dsc_mean",
        )

    return {
        "schema": report["schema"],
        "schema_version": schema_version,
        "protocol": protocol,
        "manifest_sha256": manifest_sha256,
        "seed": seed,
        "cases": cases,
        "model": report.get("model"),
    }


def _percentile_interval(
    samples: np.ndarray,
    confidence_level: float,
) -> tuple[float, float]:
    alpha = 1.0 - confidence_level
    quantiles = (alpha / 2.0, 1.0 - alpha / 2.0)
    try:
        low, high = np.quantile(samples, quantiles, method="linear")
    except TypeError:  # NumPy < 1.22
        low, high = np.quantile(samples, quantiles, interpolation="linear")
    return float(low), float(high)


def _model_binding(report: dict[str, Any]) -> dict[str, Any]:
    model = report.get("model")
    if not isinstance(model, dict):
        return {}
    binding = {}
    for key in ("checkpoint", "checkpoint_sha256", "model_type", "adapter_config"):
        if key in model:
            binding[key] = model[key]
    return binding


def analyze_reports(
    reference_report: Any,
    candidate_report: Any,
    *,
    expected_cases: int = 108,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 20260724,
    confidence_level: float = 0.95,
    tie_tolerance: float = 0.0,
    reference_source: dict[str, Any] | None = None,
    candidate_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit, align, and compare two matched validation evaluator reports."""
    if expected_cases <= 0:
        raise ValueError("expected_cases must be positive")
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be within (0,1)")
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise ValueError("tie_tolerance must be finite and non-negative")

    reference = _audit_report(
        reference_report, name="reference", expected_cases=expected_cases
    )
    candidate = _audit_report(
        candidate_report, name="candidate", expected_cases=expected_cases
    )
    if candidate["schema"] != reference["schema"]:
        _fail("report schemas differ")
    if candidate["schema_version"] != reference["schema_version"]:
        _fail("report schema versions differ")
    if candidate["manifest_sha256"] != reference["manifest_sha256"]:
        _fail("validation manifest SHA-256 values differ")
    if candidate["protocol"] != reference["protocol"]:
        _fail("evaluation protocols differ")
    if candidate["seed"] != reference["seed"]:
        _fail("evaluation RNG seeds differ")

    reference_case_ids = set(reference["cases"])
    candidate_case_ids = set(candidate["cases"])
    if candidate_case_ids != reference_case_ids:
        missing = sorted(reference_case_ids - candidate_case_ids)
        extra = sorted(candidate_case_ids - reference_case_ids)
        _fail(f"case ID sets differ: missing={missing}, extra={extra}")
    for case_id in sorted(reference_case_ids):
        if (
            candidate["cases"][case_id]["subject_id"]
            != reference["cases"][case_id]["subject_id"]
        ):
            _fail(f"subject mapping differs for case {case_id!r}")

    case_ids = sorted(reference_case_ids)
    subject_ids = sorted(
        {reference["cases"][case_id]["subject_id"] for case_id in case_ids}
    )
    subject_cases: dict[str, list[str]] = defaultdict(list)
    for case_id in case_ids:
        subject_cases[reference["cases"][case_id]["subject_id"]].append(case_id)

    subject_delta_matrix = np.empty(
        (len(subject_ids), len(CLICK_KEYS)), dtype=np.float64
    )
    point_estimates: dict[str, dict[str, Any]] = {}
    for click_index, click in enumerate(CLICK_KEYS):
        reference_values = [
            reference["cases"][case_id]["dsc"][click] for case_id in case_ids
        ]
        candidate_values = [
            candidate["cases"][case_id]["dsc"][click] for case_id in case_ids
        ]
        case_deltas = [
            candidate_value - reference_value
            for reference_value, candidate_value in zip(
                reference_values, candidate_values
            )
        ]
        subject_reference = []
        subject_candidate = []
        subject_deltas = []
        for subject_index, subject_id in enumerate(subject_ids):
            ids = subject_cases[subject_id]
            reference_mean = statistics.fmean(
                reference["cases"][case_id]["dsc"][click] for case_id in ids
            )
            candidate_mean = statistics.fmean(
                candidate["cases"][case_id]["dsc"][click] for case_id in ids
            )
            delta = candidate_mean - reference_mean
            subject_reference.append(reference_mean)
            subject_candidate.append(candidate_mean)
            subject_deltas.append(delta)
            subject_delta_matrix[subject_index, click_index] = delta

        point_estimates[click] = {
            "scan_macro": {
                "reference_dsc_mean": statistics.fmean(reference_values),
                "candidate_dsc_mean": statistics.fmean(candidate_values),
                "difference": statistics.fmean(case_deltas),
            },
            "case_paired": {
                "improved": sum(delta > tie_tolerance for delta in case_deltas),
                "worse": sum(delta < -tie_tolerance for delta in case_deltas),
                "tie": sum(abs(delta) <= tie_tolerance for delta in case_deltas),
                "tie_tolerance": tie_tolerance,
            },
            "subject_balanced": {
                "subjects": len(subject_ids),
                "reference_dsc_mean": statistics.fmean(subject_reference),
                "candidate_dsc_mean": statistics.fmean(subject_candidate),
                "difference": statistics.fmean(subject_deltas),
            },
            "subject_paired": {
                "improved": sum(
                    delta > tie_tolerance for delta in subject_deltas
                ),
                "worse": sum(delta < -tie_tolerance for delta in subject_deltas),
                "tie": sum(
                    abs(delta) <= tie_tolerance for delta in subject_deltas
                ),
                "tie_tolerance": tie_tolerance,
            },
        }

    rng = np.random.default_rng(bootstrap_seed)
    sampled_subject_indices = rng.integers(
        0,
        len(subject_ids),
        size=(bootstrap_iterations, len(subject_ids)),
    )
    bootstrap_samples = subject_delta_matrix[sampled_subject_indices, :].mean(
        axis=1
    )
    for click_index, click in enumerate(CLICK_KEYS):
        low, high = _percentile_interval(
            bootstrap_samples[:, click_index], confidence_level
        )
        point_estimates[click]["subject_level_paired_bootstrap"] = {
            "unit": "subject",
            "statistic": "mean within-subject candidate-minus-reference DSC",
            "iterations": bootstrap_iterations,
            "seed": bootstrap_seed,
            "confidence_level": confidence_level,
            "interval_method": "percentile_linear",
            "ci": [low, high],
        }

    return {
        "schema": OUTPUT_SCHEMA,
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": {
            "direction": "candidate_minus_reference",
            "reference": {
                **(reference_source or {}),
                "model": _model_binding(reference),
            },
            "candidate": {
                **(candidate_source or {}),
                "model": _model_binding(candidate),
            },
        },
        "validation_contract": {
            "report_schema": reference["schema"],
            "report_schema_version": reference["schema_version"],
            "expected_split": "validation",
            "manifest_sha256": reference["manifest_sha256"],
            "cases": len(case_ids),
            "subjects": len(subject_ids),
            "evaluation_seed": reference["seed"],
            "protocol": reference["protocol"],
            "case_and_subject_mapping_exact_match": True,
            "complete_manifest_required": True,
        },
        "analysis": point_estimates,
    }


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSON report {path}: {error}") from error
    return _require_mapping(payload, str(path))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two matched full clean-validation evaluator reports using "
            "case pairing and a subject-level paired bootstrap."
        )
    )
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=108)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--tie-tolerance", type=float, default=0.0)
    args = parser.parse_args()

    reference_path = args.reference_report.resolve()
    candidate_path = args.candidate_report.resolve()
    output_path = args.output.resolve()
    if reference_path == candidate_path:
        _fail("reference and candidate report paths must differ")
    if output_path in (reference_path, candidate_path):
        _fail("output path must not overwrite an input report")

    result = analyze_reports(
        load_json(reference_path),
        load_json(candidate_path),
        expected_cases=args.expected_cases,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
        tie_tolerance=args.tie_tolerance,
        reference_source={
            "report": str(reference_path),
            "report_sha256": sha256_file(reference_path),
        },
        candidate_source={
            "report": str(candidate_path),
            "report_sha256": sha256_file(candidate_path),
        },
    )
    write_json_atomic(output_path, result)
    print(json.dumps(result["analysis"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
