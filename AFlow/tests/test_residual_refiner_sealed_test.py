from __future__ import annotations

import pytest

from scripts import build_t1ce_wt_gtcrop_n4_test_cache as cache
from scripts import evaluate_t1ce_wt_residual_refiner_test as evaluate


def _case(index: int, subject: str, coarse: float, refined: float, voxels: int) -> dict:
    return {
        "manifest_index": index,
        "case_id": f"case-{index}",
        "subject_id": subject,
        "target_voxels": voxels,
        "coarse": {"dsc": coarse},
        "refined": {"dsc": refined},
        "delta_dsc": refined - coarse,
    }


def test_frozen_model_and_manifest_bindings() -> None:
    assert cache.TEST_COUNT == 270
    assert cache.TEST_SUBJECTS == 119
    assert cache.TEST_MANIFEST_SHA256 == "a5ddc0ed2c0b269431758b14061da3c6163dd0c48133b60a2ef8efd7f23aa8f9"
    assert evaluate.EXPECTED_REFINER_EPOCH == 58
    assert evaluate.EXPECTED_REFINER_SHA256 == "4513ea4544ae153064575ec2ed5c56e1b78adf5e89e9d094754227b3bd429f21"
    assert evaluate.EXPECTED_VALIDATION_DSC == pytest.approx(0.6683989912099032)
    assert evaluate.EXPECTED_WORLD_SIZE == 2


def test_summary_keeps_scan_subject_delta_and_size_strata_separate() -> None:
    cases = [
        _case(0, "a", 0.4, 0.5, 4000),
        _case(1, "a", 0.6, 0.7, 12000),
        _case(2, "b", 0.8, 0.7, 45000),
    ]
    summary = evaluate.summarize(cases)
    assert summary["scan_macro"]["coarse"]["mean"] == pytest.approx(0.6)
    assert summary["scan_macro"]["refined"]["mean"] == pytest.approx(19 / 30)
    assert summary["scan_macro"]["paired_delta"]["mean"] == pytest.approx(1 / 30)
    assert summary["scan_macro"]["improved_cases"] == 2
    assert summary["scan_macro"]["worsened_cases"] == 1
    assert summary["subject_balanced"]["coarse"]["mean"] == pytest.approx(0.65)
    assert summary["subject_balanced"]["refined"]["mean"] == pytest.approx(0.65)
    assert summary["subject_balanced"]["coarse"]["subjects"] == 2
    assert [item["cases"] for item in summary["by_gt_crop_target_voxels"]] == [1, 1, 0, 1]


def test_authorized_builder_audit_is_subject_clean_without_opening_nifti() -> None:
    test_rows, development_rows = cache.read_and_audit()
    assert len(test_rows) == 270
    assert len({row["subject_id"] for row in test_rows}) == 119
    assert not ({row["case_id"] for row in test_rows} & {row["case_id"] for row in development_rows})
    assert not ({row["subject_id"] for row in test_rows} & {row["subject_id"] for row in development_rows})
