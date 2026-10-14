from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scripts.build_ucsd_official_float_source_correction_cache as build
import scripts.evaluate_ucsd_official_float_source_correction as evaluate


def _summary(values: list[tuple[str, str, float, bool]]) -> dict:
    rows = [
        {
            "case_id": case,
            "subject_id": subject,
            "locator_dsc": dice,
            "automatic_point_hits_gt": hit,
            "target_wt_positive": True,
        }
        for case, subject, dice, hit in values
    ]
    return {"per_case": rows, "automatic_point_hits": sum(row[3] for row in values)}


def test_source_delta_is_paired_by_scan_and_subject() -> None:
    official = _summary([("a", "s1", 0.8, True), ("b", "s1", 0.4, False), ("c", "s2", 0.9, True)])
    mirror = _summary([("a", "s1", 0.7, False), ("b", "s1", 0.5, False), ("c", "s2", 0.6, True)])
    result = evaluate.compare_source_version(official, mirror, 7)
    delta = result["official_float_minus_mirror_uint8"]
    assert delta["scan_dsc_mean"] == pytest.approx(0.1)
    assert delta["subject_dsc_mean"] == pytest.approx(0.15)
    assert delta["positive_point_hits"] == 1
    assert result["scan_cases_improved"] == 2
    assert result["scan_cases_worse"] == 1
    assert len(result["paired_scan_dsc_mean_95ci"]) == 2


def test_source_delta_rejects_case_order_change() -> None:
    official = _summary([("a", "s1", 0.8, True), ("b", "s2", 0.4, False)])
    mirror = _summary([("b", "s2", 0.4, False), ("a", "s1", 0.8, True)])
    with pytest.raises(evaluate.SourceCorrectionEvaluationError, match="case order"):
        evaluate.compare_source_version(official, mirror, 7)


def test_cache_contract_is_explicitly_reused_not_new() -> None:
    source = Path(build.__file__).read_text()
    assert '"source_correction_reuses_old_ucsd75": True' in source
    assert '"independent_new_cohort": False' in source
    assert '"external_confirmation": False' in source
    assert '"split": "external_reused_source_correction"' in source


def test_evaluator_has_no_performance_promotion_gate() -> None:
    source = Path(evaluate.__file__).read_text()
    assert '"performance_promotion_gate": None' in source
    assert '"result_must_not_trigger_tuning_promotion_or_repeat": True' in source
    assert '"views": ["identity"]' in source


def test_frozen_historical_report_contract() -> None:
    path = Path("workspace/BraTS/ucsd_ptgbm_external_confirmation_20260902/UCSD_EXTERNAL_CONFIRMATION.json")
    report = evaluate._validate_old_report(path)
    assert report["primary"]["identifier"] == "full972_random_init_voxel_kd_epoch168"
    assert report["secondary"]["identifier"] == "gcsam_fit864_inner_selected_epoch184"


def test_builder_rejects_modified_frozen_metadata(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"cases": 75}))
    with pytest.raises(build.SourceCorrectionError, match="SHA-256 mismatch"):
        build._read_frozen(path, "0" * 64, "bad")
