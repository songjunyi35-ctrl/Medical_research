from copy import deepcopy

import pytest

from scripts.analyze_t1ce_wt_plain_fixed_outer import analyze
from scripts.evaluate_t1ce_wt_plain_fixed_outer import OUTER_SELECTOR_SHA256, SCHEMA


def _reports(candidate_delta: float = 0.01, point: float = 0.91):
    cases = [
        {"case_id": "S1-100", "subject_id": "S1", "locator_dsc": 0.60},
        {"case_id": "S1-101", "subject_id": "S1", "locator_dsc": 0.70},
        {"case_id": "S2-100", "subject_id": "S2", "locator_dsc": 0.80},
    ]
    candidate_cases = deepcopy(cases)
    for item in candidate_cases:
        item["locator_dsc"] += candidate_delta
    baseline = {
        "schema": "aflow.sammed3d.t1ce_wt_plain_locator_selector_reload",
        "exact_report_equal": True,
        "scan_dsc_difference": 0.0,
        "selector_manifest_sha256": OUTER_SELECTOR_SHA256,
        "confirmation_access": False,
        "test_access": False,
        "selected_epoch": 99,
        "checkpoint_sha256": "old",
        "reload_validation": {
            "locator_dsc_mean": 0.70,
            "locator_subject_dsc_mean": 0.725,
            "automatic_point_hit_rate": 0.92,
            "per_case": cases,
        },
    }
    candidate = {
        "schema": SCHEMA,
        "role": "outer_selector",
        "fixed_epoch": 168,
        "checkpoint": {"sha256": "new"},
        "case_manifest": {"sha256": OUTER_SELECTOR_SHA256},
        "summary": {
            "locator_dsc_mean": 0.70 + candidate_delta,
            "locator_subject_dsc_mean": 0.725 + candidate_delta,
            "automatic_point_hit_rate": point,
            "per_case": candidate_cases,
        },
        "outer_selector_access": True,
        "confirmation_access": False,
        "test_access": False,
    }
    return baseline, candidate


def test_outer_gates_pass_for_material_paired_gain():
    baseline, candidate = _reports()
    result = analyze(baseline, candidate, bootstrap_samples=100, bootstrap_seed=1)
    assert result["all_outer_gates_passed"] is True
    assert result["paired_differences"]["scan"] == pytest.approx(0.01)
    assert result["paired_differences"]["subject"] == pytest.approx(0.01)


def test_outer_gate_rejects_subthreshold_gain():
    baseline, candidate = _reports(candidate_delta=0.002)
    result = analyze(baseline, candidate, bootstrap_samples=100, bootstrap_seed=1)
    assert result["all_outer_gates_passed"] is False
    assert result["gates"]["scan_gain_at_least_0.003"] is False


def test_outer_analysis_rejects_test_access():
    baseline, candidate = _reports()
    candidate["test_access"] = True
    with pytest.raises(ValueError, match="forbidden"):
        analyze(baseline, candidate, bootstrap_samples=10)
