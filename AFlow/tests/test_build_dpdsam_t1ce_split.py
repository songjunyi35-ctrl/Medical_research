import json
from pathlib import Path

import pytest

from scripts.build_dpdsam_t1ce_split import (
    build_development_split,
    normalize_development_row,
)


def _row(subject: str, timepoint: int) -> dict:
    case = f"{subject}-{timepoint:03d}"
    return {
        "case_id": case,
        "subject_id": subject,
        "timepoint": f"{timepoint:03d}",
        "modalities": {
            "t1n": f"/{case}-t1n.nii",
            "t1c": f"/{case}-t1c.nii",
            "t2w": f"/{case}-t2w.nii",
            "t2f": f"/{case}-t2f.nii",
        },
        "seg_path": f"/{case}-seg.nii",
        "shape": [182, 218, 182],
        "spacing_mm": [1.0, 1.0, 1.0],
    }


def test_normalize_row_exposes_only_t1ce_and_fixed_binary_contract() -> None:
    normalized = normalize_development_row(_row("BraTS-GLI-00001", 100), "train")
    encoded = json.dumps(normalized)
    assert normalized["modality"] == "t1ce"
    assert normalized["image_path"].endswith("-t1c.nii")
    assert normalized["foreground_labels"] == [1, 2, 3]
    assert normalized["excluded_labels"] == [4]
    assert "t1n" not in encoded and "t2w" not in encoded and "t2f" not in encoded


def test_split_is_deterministic_and_subject_disjoint() -> None:
    rows = [
        _row("BraTS-GLI-00001", 100),
        _row("BraTS-GLI-00001", 101),
        _row("BraTS-GLI-00002", 100),
        _row("BraTS-GLI-00003", 100),
    ]
    left = build_development_split(rows, seed=17, validation_subjects=1)
    right = build_development_split(list(reversed(rows)), seed=17, validation_subjects=1)
    assert left == right
    train, val = left
    assert {row["subject_id"] for row in train}.isdisjoint(
        {row["subject_id"] for row in val}
    )
    assert len(train) + len(val) == len(rows)


def test_duplicate_case_fails_closed() -> None:
    row = _row("BraTS-GLI-00001", 100)
    with pytest.raises(ValueError, match="duplicate development case_id"):
        build_development_split([row, dict(row)], validation_subjects=1)


def test_missing_t1ce_fails_closed() -> None:
    row = _row("BraTS-GLI-00001", 100)
    del row["modalities"]["t1c"]
    with pytest.raises(ValueError, match="modalities.t1c"):
        normalize_development_row(row, "train")
