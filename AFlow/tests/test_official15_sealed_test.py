from __future__ import annotations

import pytest

from scripts import build_t1ce_wt_official15_test_cache as cache
from scripts import evaluate_t1ce_wt_upstream_literal_test as evaluate


def _row(**overrides):
    value = {
        "schema": "aflow.brats2024.random_subject_train_test_split",
        "schema_version": 1,
        "case_id": "BraTS-GLI-00006-100",
        "subject_id": "BraTS-GLI-00006",
        "split": "test",
        "modalities": {"t1c": "/data/case-t1c.nii"},
        "seg_path": "/data/case-seg.nii",
        "source_part": "Part1",
        "timepoint": "100",
    }
    value.update(overrides)
    return value


def test_sealed_row_adapter_locks_t1ce_wt_and_test() -> None:
    adapted = cache.adapt_test_row(_row())
    assert adapted["image_path"] == "/data/case-t1c.nii"
    assert adapted["label_path"] == "/data/case-seg.nii"
    assert adapted["foreground_labels"] == [1, 2, 3]
    assert adapted["excluded_labels"] == [4]
    assert adapted["split"] == "test"


@pytest.mark.parametrize(
    "change",
    [
        {"split": "validation"},
        {"schema": "unexpected"},
        {"modalities": {"t1c": "/a", "t2f": "/b"}},
    ],
)
def test_sealed_row_adapter_fails_closed(change) -> None:
    with pytest.raises(ValueError):
        cache.adapt_test_row(_row(**change))


def _case(index, subject, curve):
    return {
        "manifest_index": index,
        "case_id": f"case-{index}",
        "subject_id": subject,
        "click_dice": list(curve),
        "best_of_5_dice": max(curve),
        "best_click": curve.index(max(curve)) + 1,
        "final_click_5_dice": curve[-1],
    }


def test_summary_reports_primary_final_click_curve_and_subject_balance() -> None:
    cases = [
        _case(0, "a", [0.1, 0.2, 0.3, 0.4, 0.5]),
        _case(1, "a", [0.3, 0.4, 0.5, 0.6, 0.7]),
        _case(2, "b", [0.9, 0.8, 0.7, 0.6, 0.5]),
    ]
    summary = evaluate.summarize(cases)
    assert summary["primary_best_of_5"]["scan_macro"]["mean"] == pytest.approx(0.7)
    assert summary["final_click_5"]["scan_macro"]["mean"] == pytest.approx(17 / 30)
    assert summary["per_click"]["1"]["scan_macro"]["mean"] == pytest.approx(13 / 30)
    assert summary["per_click"]["5"]["subject_balanced"]["mean"] == pytest.approx(0.55)
    assert summary["per_click"]["5"]["subject_balanced"]["subjects"] == 2
    assert summary["best_click_counts"] == {"1": 1, "2": 0, "3": 0, "4": 0, "5": 2}


def test_frozen_bindings_are_the_completed_epoch21_selection() -> None:
    assert cache.TEST_COUNT == 270
    assert cache.TEST_MANIFEST_SHA256 == "a5ddc0ed2c0b269431758b14061da3c6163dd0c48133b60a2ef8efd7f23aa8f9"
    assert evaluate.EXPECTED_CHECKPOINT_EPOCH == 21
    assert evaluate.EXPECTED_CHECKPOINT_SHA256 == "c0c85a0fb90f6301344053f97cabe8a4972bd9febd4c6ae248f3a0908598fe31"
    assert evaluate.EXPECTED_VAL_BEST5 == 0.5713061351995669
    assert evaluate.upstream.VALIDATION_CLICKS == 5
