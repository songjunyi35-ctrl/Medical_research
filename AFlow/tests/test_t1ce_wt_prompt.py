import numpy as np
import pytest

from sammed3d.t1ce_wt_prompt import validate_row, wt_target


def test_wt_target_includes_123_and_excludes_rc4() -> None:
    raw = np.array(
        [
            [[0, 1], [2, 3]],
            [[4, 0], [4, 1]],
        ],
        dtype=np.uint8,
    )
    target = wt_target(raw)
    assert target.tolist() == [
        [[False, True], [True, True]],
        [[False, False], [False, True]],
    ]


def test_wt_target_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="unexpected labels"):
        wt_target(np.array([[[5]]], dtype=np.uint8))


def test_row_contract_rejects_rc_as_foreground() -> None:
    row = {
        "schema": "aflow.brats2024.sammed3d_t1ce_wt_inner_split",
        "case_id": "BraTS-GLI-00001-100",
        "subject_id": "BraTS-GLI-00001",
        "modality": "t1ce",
        "target": "WT",
        "image_path": "/data/t1c.nii",
        "label_path": "/data/seg.nii",
        "foreground_labels": [1, 2, 3, 4],
        "excluded_labels": [],
    }
    with pytest.raises(ValueError, match="foreground_labels"):
        validate_row(row)
