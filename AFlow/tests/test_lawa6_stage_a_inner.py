from __future__ import annotations

from scripts.evaluate_t1ce_wt_lawa6_stage_a_inner import promotion_gates
from scripts.train_t1ce_wt_lawa6_stage_a import ARMS


def _summaries(raw_point: float, lawa_point: float) -> dict:
    result = {}
    for arm in ARMS:
        result[f"{arm}_raw"] = {"automatic_point_hit_rate": raw_point}
        result[f"{arm}_lawa6"] = {"automatic_point_hit_rate": lawa_point}
    return result


def _comparisons(scan: float, subject: float, scan_low: float, subject_low: float) -> dict:
    return {
        arm: {
            "ema_minus_raw": {
                "scan": scan,
                "subject": subject,
                "scan_95ci": [scan_low, 0.01],
                "subject_95ci": [subject_low, 0.01],
            }
        }
        for arm in ARMS
    }


def test_stage_a_all_conjunctive_gates_pass() -> None:
    gates = promotion_gates(_summaries(0.95, 0.96), _comparisons(0.001, 0.002, -0.002, -0.001))
    assert len(gates) == 10
    assert all(gates.values())


def test_stage_a_zero_dice_delta_fails_strict_gate() -> None:
    gates = promotion_gates(_summaries(0.95, 0.95), _comparisons(0.0, 0.001, -0.001, -0.001))
    assert not gates[f"{ARMS[0]}_lawa_beats_raw_scan"]


def test_stage_a_point_drop_fails_gate() -> None:
    gates = promotion_gates(_summaries(0.96, 0.95), _comparisons(0.001, 0.001, -0.001, -0.001))
    assert not gates[f"{ARMS[1]}_lawa_point_not_lower"]


def test_stage_a_ci_below_margin_fails_gate() -> None:
    gates = promotion_gates(_summaries(0.95, 0.95), _comparisons(0.001, 0.001, -0.003001, -0.003001))
    assert not gates[f"{ARMS[0]}_scan_ci_safe"]
    assert not gates[f"{ARMS[0]}_subject_ci_safe"]
