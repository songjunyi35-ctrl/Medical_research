from __future__ import annotations

from scripts.evaluate_t1ce_wt_lookahead5_stage_a_inner import pairing_gates, promotion_gates


def _summary(scan: float, subject: float, point: float) -> dict:
    return {"locator_dsc_mean": scan, "locator_subject_dsc_mean": subject, "automatic_point_hit_rate": point}


def _comparison(scan: float, subject: float, scan_low: float, subject_low: float) -> dict:
    return {"ema_minus_raw": {"scan": scan, "subject": subject, "scan_95ci": [scan_low, 0.01], "subject_95ci": [subject_low, 0.01]}}


def test_promotion_requires_all_six_gates() -> None:
    gates = promotion_gates(_summary(0.73, 0.72, 0.95), _summary(0.731, 0.721, 0.96), _comparison(0.001, 0.001, -0.002, -0.002), True)
    assert len(gates) == 6
    assert all(gates.values())


def test_promotion_rejects_point_drop_and_unsafe_ci() -> None:
    gates = promotion_gates(_summary(0.73, 0.72, 0.96), _summary(0.731, 0.721, 0.95), _comparison(0.001, 0.001, -0.003001, -0.003001), True)
    assert not gates["candidate_point_not_lower"]
    assert not gates["scan_ci_safe"]
    assert not gates["subject_ci_safe"]


def test_pairing_contract_requires_exact_counts_and_sequences() -> None:
    shared = {
        "initial_state_sha256": "i",
        "sampling_weight_sha256": "w",
        "sampler_epoch_sha256": [str(index) for index in range(168)],
        "first_augmented_batch": {"image_sha256": "x", "label_sha256": "y"},
    }
    control = {**shared, "optimizer_audit": {}}
    candidate = {**shared, "optimizer_audit": {"fast_steps": 20328, "slow_updates": 4065, "trailing_fast_steps": 3, "final_slow_loaded": True, "all_counts_exact": True}}
    gates = pairing_gates({"adamw_control": control, "lookahead5_candidate": candidate})
    assert len(gates) == 9
    assert all(gates.values())


def test_pairing_contract_rejects_sampler_mismatch() -> None:
    control = {"initial_state_sha256": "i", "sampling_weight_sha256": "w", "sampler_epoch_sha256": ["a"], "first_augmented_batch": {"x": 1}, "optimizer_audit": {}}
    candidate = {"initial_state_sha256": "i", "sampling_weight_sha256": "w", "sampler_epoch_sha256": ["b"], "first_augmented_batch": {"x": 1}, "optimizer_audit": {"fast_steps": 20328, "slow_updates": 4065, "trailing_fast_steps": 3, "final_slow_loaded": True, "all_counts_exact": True}}
    gates = pairing_gates({"adamw_control": control, "lookahead5_candidate": candidate})
    assert not gates["all_sampler_epochs_exact"]
