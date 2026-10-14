from __future__ import annotations

from copy import deepcopy
import statistics

import pytest

from scripts.analyze_sammed3d_t1ce_wt_paired_validation import analyze_reports


MANIFEST_SHA256 = "a" * 64
PROTOCOL = {
    "input": "T1ce single channel",
    "target": "WT = raw labels {1,2,3}; RC label 4 excluded",
    "spatial_preprocessing": "whole_resize",
    "clicks": "matched synthetic three-click protocol",
    "num_clicks": 3,
}


def _report(rows: list[tuple[str, str, tuple[float, float, float]]]) -> dict:
    cases = [
        {
            "case_id": case_id,
            "subject_id": subject_id,
            "metrics": {
                str(click): {"dsc": values[click - 1]}
                for click in range(1, 4)
            },
        }
        for case_id, subject_id, values in rows
    ]
    summary = {}
    for click in ("1", "2", "3"):
        values = [case["metrics"][click]["dsc"] for case in cases]
        by_subject: dict[str, list[float]] = {}
        for case in cases:
            by_subject.setdefault(case["subject_id"], []).append(
                case["metrics"][click]["dsc"]
            )
        subject_means = [
            statistics.fmean(subject_values)
            for subject_values in by_subject.values()
        ]
        summary[click] = {
            "scan_macro": {
                "cases": len(cases),
                "dsc_mean": statistics.fmean(values),
            },
            "subject_balanced": {
                "subjects": len(subject_means),
                "dsc_mean": statistics.fmean(subject_means),
            },
        }
    return {
        "schema": "aflow.sammed3d.t1ce_wt_checkpoint_evaluation",
        "schema_version": 2,
        "protocol": deepcopy(PROTOCOL),
        "data": {
            "manifest": "/development/clean-validation.jsonl",
            "manifest_sha256": MANIFEST_SHA256,
            "expected_split": "validation",
            "selected_cases": len(cases),
            "full_manifest_cases": len(cases),
        },
        "model": {"checkpoint_sha256": "b" * 64},
        "seed": 20260723,
        "summary": summary,
        "cases": cases,
    }


REFERENCE_ROWS = [
    ("a-100", "a", (0.1, 0.2, 0.3)),
    ("a-101", "a", (0.3, 0.4, 0.5)),
    ("b-100", "b", (0.8, 0.5, 0.6)),
    ("c-100", "c", (0.5, 0.7, 0.7)),
]
CANDIDATE_ROWS = [
    ("c-100", "c", (0.5, 0.8, 0.7)),
    ("a-101", "a", (0.4, 0.5, 0.5)),
    ("b-100", "b", (0.7, 0.6, 0.6)),
    ("a-100", "a", (0.2, 0.3, 0.3)),
]


def test_paired_analysis_reports_case_subject_and_bootstrap_results() -> None:
    result = analyze_reports(
        _report(REFERENCE_ROWS),
        _report(CANDIDATE_ROWS),
        expected_cases=4,
        bootstrap_iterations=2_000,
        bootstrap_seed=17,
    )

    assert result["validation_contract"]["cases"] == 4
    assert result["validation_contract"]["subjects"] == 3
    c1 = result["analysis"]["1"]
    assert c1["scan_macro"]["difference"] == pytest.approx(0.025)
    assert c1["case_paired"] == {
        "improved": 2,
        "worse": 1,
        "tie": 1,
        "tie_tolerance": 0.0,
    }
    assert c1["subject_balanced"]["difference"] == pytest.approx(0.0)
    assert c1["subject_paired"]["improved"] == 1
    assert c1["subject_paired"]["worse"] == 1
    assert c1["subject_paired"]["tie"] == 1
    low, high = c1["subject_level_paired_bootstrap"]["ci"]
    assert low < 0.0 < high

    c2 = result["analysis"]["2"]
    assert c2["scan_macro"]["difference"] == pytest.approx(0.1)
    assert c2["case_paired"]["improved"] == 4
    assert c2["subject_balanced"]["difference"] == pytest.approx(0.1)
    assert c2["subject_level_paired_bootstrap"]["ci"] == pytest.approx(
        [0.1, 0.1]
    )

    c3 = result["analysis"]["3"]
    assert c3["scan_macro"]["difference"] == pytest.approx(0.0)
    assert c3["case_paired"]["tie"] == 4
    assert c3["subject_level_paired_bootstrap"]["ci"] == pytest.approx(
        [0.0, 0.0]
    )


def test_bootstrap_is_deterministic_for_fixed_seed() -> None:
    kwargs = {
        "expected_cases": 4,
        "bootstrap_iterations": 321,
        "bootstrap_seed": 99,
    }
    first = analyze_reports(
        _report(REFERENCE_ROWS), _report(CANDIDATE_ROWS), **kwargs
    )
    second = analyze_reports(
        _report(REFERENCE_ROWS), _report(CANDIDATE_ROWS), **kwargs
    )
    first_intervals = {
        click: first["analysis"][click]["subject_level_paired_bootstrap"]["ci"]
        for click in ("1", "2", "3")
    }
    second_intervals = {
        click: second["analysis"][click]["subject_level_paired_bootstrap"]["ci"]
        for click in ("1", "2", "3")
    }
    assert first_intervals == second_intervals


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["data"].__setitem__(
                "manifest_sha256", "c" * 64
            ),
            "manifest SHA-256 values differ",
        ),
        (
            lambda report: report.__setitem__("seed", 1),
            "evaluation RNG seeds differ",
        ),
        (
            lambda report: report["protocol"].__setitem__(
                "spatial_preprocessing", "different"
            ),
            "evaluation protocols differ",
        ),
        (
            lambda report: report["data"].__setitem__(
                "expected_split", "test"
            ),
            "must be a validation report",
        ),
        (
            lambda report: report["data"].__setitem__(
                "full_manifest_cases", 5
            ),
            "is not a complete 4-case validation report",
        ),
    ],
)
def test_contract_mismatches_fail_closed(mutation, message: str) -> None:
    candidate = _report(CANDIDATE_ROWS)
    mutation(candidate)
    with pytest.raises(RuntimeError, match=message):
        analyze_reports(
            _report(REFERENCE_ROWS),
            candidate,
            expected_cases=4,
            bootstrap_iterations=10,
        )


def test_case_subject_mapping_mismatch_fails_closed() -> None:
    candidate = _report(CANDIDATE_ROWS)
    candidate["cases"][0]["subject_id"] = "wrong-subject"
    candidate["summary"] = _report(
        [
            (
                case["case_id"],
                case["subject_id"],
                tuple(
                    case["metrics"][str(click)]["dsc"]
                    for click in range(1, 4)
                ),
            )
            for case in candidate["cases"]
        ]
    )["summary"]
    with pytest.raises(RuntimeError, match="subject mapping differs"):
        analyze_reports(
            _report(REFERENCE_ROWS),
            candidate,
            expected_cases=4,
            bootstrap_iterations=10,
        )


def test_inconsistent_summary_and_nonfinite_metrics_fail_closed() -> None:
    candidate = _report(CANDIDATE_ROWS)
    candidate["summary"]["1"]["scan_macro"]["dsc_mean"] = 0.0
    with pytest.raises(RuntimeError, match="inconsistent with per-case metrics"):
        analyze_reports(
            _report(REFERENCE_ROWS),
            candidate,
            expected_cases=4,
            bootstrap_iterations=10,
        )

    candidate = _report(CANDIDATE_ROWS)
    candidate["cases"][0]["metrics"]["1"]["dsc"] = float("nan")
    with pytest.raises(RuntimeError, match="finite and within"):
        analyze_reports(
            _report(REFERENCE_ROWS),
            candidate,
            expected_cases=4,
            bootstrap_iterations=10,
        )
