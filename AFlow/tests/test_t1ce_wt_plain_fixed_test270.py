from copy import deepcopy

import pytest

from scripts.analyze_t1ce_wt_plain_fixed_test270 import analyze
from scripts.evaluate_t1ce_wt_plain_fixed_test270 import (
    CHECKPOINT_SHA256,
    SCHEMA as CANDIDATE_SCHEMA,
    TEST_MANIFEST_SHA256,
)


def _reports(delta: float = 0.01):
    cases = [
        {"case_id": "S1-100", "subject_id": "S1", "locator_dsc": 0.5,
         "automatic_point_hits_gt": True},
        {"case_id": "S1-101", "subject_id": "S1", "locator_dsc": 0.7,
         "automatic_point_hits_gt": True},
        {"case_id": "S2-100", "subject_id": "S2", "locator_dsc": 0.8,
         "automatic_point_hits_gt": False},
    ]
    new_cases = deepcopy(cases)
    for item in new_cases:
        item["locator_dsc"] += delta
    baseline = {
        "schema": "aflow.sammed3d.t1ce_wt_plain_locator_test270",
        "test_access": True,
        "checkpoint": {"sha256": "0bb7ca52d7ac30068f752a4fc05a859f4dce42dce6ace40ba1f6b8b332889c2f"},
        "data": {"manifest_sha256": TEST_MANIFEST_SHA256},
        "summary": {"locator_dsc_mean": 2 / 3, "locator_subject_dsc_mean": 0.7,
                    "automatic_point_hit_rate": 2 / 3, "per_case": cases},
    }
    candidate = {
        "schema": CANDIDATE_SCHEMA,
        "test_access": True,
        "confirmation_access": False,
        "checkpoint": {"sha256": CHECKPOINT_SHA256},
        "data": {"manifest_sha256": TEST_MANIFEST_SHA256},
        "summary": {"locator_dsc_mean": 2 / 3 + delta,
                    "locator_subject_dsc_mean": 0.7 + delta,
                    "automatic_point_hit_rate": 2 / 3, "per_case": new_cases},
    }
    return baseline, candidate


def test_paired_test_comparison_detects_uniform_gain():
    baseline, candidate = _reports()
    result = analyze(baseline, candidate, bootstrap_samples=100, bootstrap_seed=1)
    assert result["paired_differences"]["scan"] == pytest.approx(0.01)
    assert result["paired_differences"]["subject"] == pytest.approx(0.01)
    assert result["new_internal_test_point_estimate_high"] is True
    assert result["paired_subject_ci_lower_positive"] is True


def test_paired_test_comparison_reports_regression():
    baseline, candidate = _reports(delta=-0.01)
    result = analyze(baseline, candidate, bootstrap_samples=100, bootstrap_seed=1)
    assert result["new_internal_test_point_estimate_high"] is False
    assert result["distribution"]["cases_worse"] == 3


def test_paired_test_comparison_rejects_confirmation_access():
    baseline, candidate = _reports()
    candidate["confirmation_access"] = True
    with pytest.raises(ValueError, match="confirmation"):
        analyze(baseline, candidate, bootstrap_samples=10)
