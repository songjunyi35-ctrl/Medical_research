from __future__ import annotations

import pytest

from scripts.analyze_t1ce_wt_intensity_residual import paired_values
from scripts.evaluate_t1ce_wt_intensity_residual import maximum_case_difference


def report(values: list[float]) -> dict:
    return {
        "summary": {
            "per_case": [
                {
                    "case_id": f"case-{index}",
                    "subject_id": f"subject-{index}",
                    "dsc": value,
                    "base_dsc": value - 0.1,
                    "residual_abs_mean": 0.2,
                    "automatic_point": [1, 2, 3],
                    "automatic_point_hits_gt": True,
                }
                for index, value in enumerate(values)
            ]
        }
    }


def test_paired_scan_and_subject_values() -> None:
    left = report([0.8, 0.6])
    right = report([0.7, 0.55])
    scan, subject = paired_values(left, right)
    assert scan == pytest.approx([0.1, 0.05])
    assert subject == pytest.approx([0.1, 0.05])


def test_reload_difference_is_exact_and_detects_change() -> None:
    left = report([0.8, 0.6])["summary"]
    right = report([0.8, 0.6])["summary"]
    assert maximum_case_difference(left, right) == 0.0
    right["per_case"][0]["dsc"] += 0.01
    assert maximum_case_difference(left, right) == pytest.approx(0.01)

