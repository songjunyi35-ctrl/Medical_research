from copy import deepcopy
from datetime import timedelta
import pytest
from scripts.prepare_n4_family_full972_admission import DEADLINE, performance_passes, time_gate


def passing_report():
    row = {
        "delta": {"scan_dice": 0.01, "subject_dice": 0.01, "point_hits": 0,
                  "scan_dice_95ci": [0.002, 0.02], "subject_dice_95ci": [0.001, 0.02]},
        "gates": {"scan_positive": True, "subject_positive": True, "point_within_one": True,
                  "scan_ci_safe": True, "subject_ci_safe": True}, "all_gates_passed": True,
    }
    return {"comparisons": {f: {s: deepcopy(row) for s in ("identity", "flip8")} for f in ("e", "f")},
            "aggregate": {"all_gates_passed": True}, "stage_b_authorized": True}


def test_report_boolean_cannot_override_negative_metrics():
    report = passing_report()
    assert performance_passes(report)
    report["comparisons"]["f"]["flip8"]["delta"]["scan_dice"] = -0.001
    with pytest.raises(ValueError):
        performance_passes(report)


def test_missing_cross_fold_margin_cannot_promote():
    report = passing_report()
    for fold in ("e", "f"):
        report["comparisons"][fold]["identity"]["delta"]["scan_dice"] = 0.0005
    with pytest.raises(ValueError):
        performance_passes(report)


def test_full_stage_requires_eight_hours_remaining():
    assert time_gate(DEADLINE - timedelta(hours=8))
    assert not time_gate(DEADLINE - timedelta(hours=8) + timedelta(seconds=1))
    with pytest.raises(ValueError):
        time_gate(DEADLINE.replace(tzinfo=None))
