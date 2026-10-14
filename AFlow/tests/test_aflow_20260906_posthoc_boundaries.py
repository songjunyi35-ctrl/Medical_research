from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AFRICA = ROOT / "archive/research_records/2026-09-06_brats_africa95_frozen_baselines"
CAPACITY = ROOT / "archive/reviews/2026-09-06_fresh_evidence_capacity"


def load(path: Path): return json.loads(path.read_text())


def test_fresh_capacity_is_exhausted_without_role_reassignment() -> None:
    value = load(CAPACITY / "CAPACITY_AUDIT.json")
    assert value["status"] == "exhausted_for_new_two_fold_inner_under_immutable_roles"
    assert value["source"] == {"scans": 972, "subjects": 444, "sha256": "6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65"}
    assert value["pre_ef"]["eligible_subjects"] == 38 and value["pre_ef"]["eligible_scans"] == 87
    assert value["remaining_never_evaluated"]["subjects"] == 2 and value["remaining_never_evaluated"]["scans"] == 3
    assert value["feasibility"]["new_42_scan_fold"] is False
    assert value["feasibility"]["role_reassignment_required_to_continue_internal"] is True
    assert value["model_inference"] is False and value["metrics_recomputed"] is False


def test_africa_posthoc_outputs_are_nonpromotional() -> None:
    names = [
        "POSTHOC_CENTER_DESCRIPTION.json",
        "POSTHOC_VOLUME_DESCRIPTION.json",
        "POSTHOC_ERROR_AUDIT.json",
        "POSTHOC_PRECISION_RECALL.json",
        "POSTHOC_CENTER_CLUSTER_SENSITIVITY.json",
        "POSTHOC_TRAIN_TARGET_SUPPORT.json",
        "CROSS_COHORT_UCSD_DESCRIPTION.json",
        "CHECKPOINT_HEAD_AUDIT.json",
    ]
    for name in names:
        value = load(AFRICA / name)
        text = json.dumps(value).lower()
        assert value.get("new_model_inference", value.get("model_forward", value.get("model_inference", False))) is False
        assert "tuning" in text or "selection" in text
        assert not value.get("authorized_for_test", False)


def test_target_support_is_train_defined_and_not_a_router() -> None:
    value = load(AFRICA / "POSTHOC_TRAIN_TARGET_SUPPORT.json")
    assert value["status"] == "descriptive_train_defined_thresholds_no_routing_or_tuning"
    assert value["africa_target_empirical_percentile_in_train972"]["median"] == 0.9547325102880658
    assert value["train_defined_splits"]["p90"]["africa_above_cases"] == 62
    assert value["africa_above_train_max_cases"] == 1
    assert value["selection_or_tuning"] is False


def test_center_sensitivity_keeps_four_center_limit() -> None:
    value = load(AFRICA / "POSTHOC_CENTER_CLUSTER_SENSITIVITY.json")
    assert len(value["gcsam_minus_full_identity"]["leave_one_center_out"]) == 4
    assert value["gcsam_minus_full_identity"]["all_leave_one_center_out_positive"] is True
    assert value["full_flip8_minus_identity"]["all_leave_one_center_out_positive"] is False
    assert value["tuning_or_promotion"] is False


def test_precision_recall_audit_does_not_authorize_thresholding() -> None:
    value = load(AFRICA / "POSTHOC_PRECISION_RECALL.json")
    gcsam = value["paired_minus_full_identity"]["gcsam184_identity"]
    flip = value["paired_minus_full_identity"]["full972_flip8"]
    assert gcsam["recall_95ci"][0] > 0
    assert gcsam["precision_95ci"][0] < 0 < gcsam["precision_95ci"][1]
    assert flip["precision_95ci"][0] > 0
    assert flip["recall_95ci"][0] < 0 < flip["recall_95ci"][1]
    assert value["threshold_or_model_selection"] is False
