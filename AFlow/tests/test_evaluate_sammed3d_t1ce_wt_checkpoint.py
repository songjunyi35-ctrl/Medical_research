from scripts.evaluate_sammed3d_t1ce_wt_checkpoint import (
    _whole_volume_training_row,
    summarize,
)


def _case(case_id: str, subject_id: str, values: tuple[float, float, float]):
    return {
        "case_id": case_id,
        "subject_id": subject_id,
        "metrics": {
            str(index): {"dsc": value, "iou": value / (2.0 - value)}
            for index, value in enumerate(values, start=1)
        },
        "clicks": [
            {
                "click_index": index,
                "status": "sampled",
                "label": 1 if index < 3 else 0,
            }
            for index in range(1, 4)
        ],
    }


def test_summary_reports_scan_and_subject_balanced_metrics() -> None:
    cases = [
        _case("a-100", "a", (0.1, 0.2, 0.3)),
        _case("a-101", "a", (0.3, 0.4, 0.5)),
        _case("b-100", "b", (0.9, 0.8, 0.7)),
    ]
    result = summarize(cases)
    assert result["3"]["scan_macro"]["dsc_mean"] == 0.5
    assert result["3"]["subject_balanced"]["dsc_mean"] == 0.55
    assert result["3"]["subject_balanced"]["subjects"] == 2
    assert result["3"]["clicks"] == {
        "sampled": 3,
        "skipped_exact": 0,
        "positive": 0,
        "negative": 3,
    }

def test_whole_volume_test_row_adapter_selects_only_t1ce_and_wt() -> None:
    row = {
        "schema": "aflow.brats2024.random_subject_train_test_split",
        "case_id": "BraTS-GLI-00006-100",
        "subject_id": "BraTS-GLI-00006",
        "split": "test",
        "modalities": {
            "t1c": "/data/case-t1c.nii",
            "t1n": object(),
            "t2w": object(),
            "t2f": object(),
        },
        "seg_path": "/data/case-seg.nii",
    }

    adapted = _whole_volume_training_row(row)

    assert adapted["image_path"] == "/data/case-t1c.nii"
    assert adapted["label_path"] == "/data/case-seg.nii"
    assert adapted["foreground_labels"] == [1, 2, 3]
    assert adapted["excluded_labels"] == [4]
    assert adapted["split"] == "test"

