from __future__ import annotations

from scripts.evaluate_t1ce_wt_lawa6_stage_b_validation import promotion_gates


def _summary(scan: float, subject: float, hits: int) -> dict:
    return {
        "locator_dsc_mean": scan,
        "locator_subject_dsc_mean": subject,
        "automatic_point_hit_rate": hits / 108,
    }


def _comparison(scan: float, subject: float, scan_low: float, subject_low: float) -> dict:
    return {"ema_minus_raw": {
        "scan": scan,
        "subject": subject,
        "scan_95ci": [scan_low, 0.01],
        "subject_95ci": [subject_low, 0.01],
    }}


def test_stage_b_all_eleven_gates_pass() -> None:
    gates = promotion_gates(
        _summary(0.74, 0.73, 103),
        _summary(0.742, 0.732, 104),
        _summary(0.741, 0.731, 103),
        _comparison(0.002, 0.002, -0.001, -0.001),
        _comparison(0.001, 0.001, -0.002, -0.002),
        True,
    )
    assert len(gates) == 11
    assert all(gates.values())


def test_stage_b_point_and_strict_archived_gates_fail() -> None:
    gates = promotion_gates(
        _summary(0.74, 0.73, 103),
        _summary(0.742, 0.732, 102),
        _summary(0.742, 0.732, 103),
        _comparison(0.002, 0.002, -0.001, -0.001),
        _comparison(0.0, 0.0, -0.001, -0.001),
        True,
    )
    assert not gates["lawa_beats_archived_scan"]
    assert not gates["lawa_beats_archived_subject"]
    assert not gates["lawa_point_at_least_raw"]
    assert not gates["lawa_point_at_least_103"]


def test_stage_b_ci_and_engineering_gates_fail() -> None:
    gates = promotion_gates(
        _summary(0.74, 0.73, 103),
        _summary(0.742, 0.732, 104),
        _summary(0.741, 0.731, 103),
        _comparison(0.002, 0.002, -0.003001, -0.003001),
        _comparison(0.001, 0.001, -0.003001, -0.003001),
        False,
    )
    assert not gates["lawa_vs_raw_scan_ci_safe"]
    assert not gates["lawa_vs_raw_subject_ci_safe"]
    assert not gates["lawa_vs_archived_scan_ci_safe"]
    assert not gates["lawa_vs_archived_subject_ci_safe"]
    assert not gates["engineering_gates"]
