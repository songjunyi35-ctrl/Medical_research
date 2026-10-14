from scripts.evaluate_t1ce_wt_gamma_aug_stage_a_inner import pairing_gates, promotion_gates


def test_gamma_pairing_gates_pass():
    shared = {"initial_state_sha256": "i", "sampling_weight_sha256": "w", "sampler_epoch_sha256": [str(i) for i in range(168)], "first_pre_gamma_batch": {"x": 1}}
    control = {**shared}
    candidate = {**shared, "gamma_audit": {"total_samples": 162624, "applied_fraction": 0.3, "gamma_min": 0.71, "gamma_max": 1.49, "max_mean_abs_error": 1e-6, "max_population_std_abs_error": 1e-6}, "gamma_gates": {"first_label_unchanged": True, "global_rng_unchanged_by_gamma": True}}
    gates = pairing_gates({"baseline_control": control, "gamma_candidate": candidate})
    assert len(gates) == 9 and all(gates.values())


def test_gamma_pairing_rejects_fraction_and_rng():
    shared = {"initial_state_sha256": "i", "sampling_weight_sha256": "w", "sampler_epoch_sha256": ["a"], "first_pre_gamma_batch": {"x": 1}}
    candidate = {**shared, "gamma_audit": {"total_samples": 162624, "applied_fraction": 0.36, "gamma_min": 0.8, "gamma_max": 1.2, "max_mean_abs_error": 0, "max_population_std_abs_error": 0}, "gamma_gates": {"first_label_unchanged": True, "global_rng_unchanged_by_gamma": False}}
    gates = pairing_gates({"baseline_control": shared, "gamma_candidate": candidate})
    assert not gates["gamma_fraction_ok"] and not gates["label_and_global_rng_unchanged"]


def test_gamma_promotion_is_conjunctive():
    control = {"locator_dsc_mean": 0.73, "locator_subject_dsc_mean": 0.72, "automatic_point_hit_rate": 0.95}
    candidate = {"locator_dsc_mean": 0.731, "locator_subject_dsc_mean": 0.721, "automatic_point_hit_rate": 0.95}
    comparison = {"ema_minus_raw": {"scan_95ci": [-0.002, 0.01], "subject_95ci": [-0.002, 0.01]}}
    gates = promotion_gates(control, candidate, comparison, True)
    assert len(gates) == 6 and all(gates.values())


def test_gamma_promotion_rejects_point_drop_and_ci():
    control = {"locator_dsc_mean": 0.73, "locator_subject_dsc_mean": 0.72, "automatic_point_hit_rate": 0.96}
    candidate = {"locator_dsc_mean": 0.731, "locator_subject_dsc_mean": 0.721, "automatic_point_hit_rate": 0.95}
    comparison = {"ema_minus_raw": {"scan_95ci": [-0.00301, 0.01], "subject_95ci": [-0.00301, 0.01]}}
    gates = promotion_gates(control, candidate, comparison, True)
    assert not gates["candidate_point_not_lower"] and not gates["scan_ci_safe"] and not gates["subject_ci_safe"]
