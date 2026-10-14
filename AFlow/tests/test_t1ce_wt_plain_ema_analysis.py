from __future__ import annotations

from scripts.evaluate_t1ce_wt_plain_ema_validation import compare


def _summary(shift: float, point: float = 0.9) -> dict:
    per_case = [
        {
            "case_id": f"BraTS-GLI-{index:05d}-100",
            "subject_id": f"BraTS-GLI-{index:05d}",
            "locator_dsc": 0.7 + shift,
            "automatic_point_hits_gt": index < round(10 * point)
        }
        for index in range(10)
    ]
    return {
        "locator_dsc_mean": 0.7 + shift,
        "locator_subject_dsc_mean": 0.7 + shift,
        "automatic_point_hit_rate": point,
        "per_case": per_case
    }


def test_ema_gates_pass_for_uniform_gain() -> None:
    result = compare(_summary(0.0), _summary(0.005))
    assert result["all_gates_passed"] is True


def test_ema_gates_reject_negative_gain() -> None:
    result = compare(_summary(0.0), _summary(-0.001))
    assert result["all_gates_passed"] is False


def test_ema_gates_reject_large_point_drop() -> None:
    result = compare(_summary(0.0, 0.9), _summary(0.005, 0.8))
    assert result["gates"]["point_hit_drop_at_most_0.01"] is False
