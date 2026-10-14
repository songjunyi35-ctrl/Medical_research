from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_brats2024_random_train_test_split.py"
)
SPEC = importlib.util.spec_from_file_location("random_train_test_split", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def rows() -> list[dict]:
    result = []
    counts = {"BraTS-GLI-00001": 1, "BraTS-GLI-00002": 2, "BraTS-GLI-00003": 3, "BraTS-GLI-00004": 4}
    for subject, count in counts.items():
        for index in range(count):
            case_id = f"{subject}-{100 + index}"
            result.append(
                {
                    "case_id": case_id,
                    "subject_id": subject,
                    "timepoint": str(100 + index),
                    "source_part": "Part1",
                    "source_case_dir": f"/data/{case_id}",
                    "modalities": {name: f"/data/{case_id}-{name}.nii" for name in module.MODALITIES},
                    "seg_path": f"/data/{case_id}-seg.nii",
                }
            )
    return result


def test_exact_subject_clean_split_is_deterministic() -> None:
    first_train, first_test = module.split_rows(rows(), seed=7, test_cases=4)
    second_train, second_test = module.split_rows(rows(), seed=7, test_cases=4)
    assert first_train == second_train
    assert first_test == second_test
    assert len(first_train) == 6
    assert len(first_test) == 4
    assert module.subject_ids(first_train).isdisjoint(module.subject_ids(first_test))
    assert {row["case_id"] for row in first_train + first_test} == {
        row["case_id"] for row in rows()
    }
    assert {row["split"] for row in first_train} == {"train"}
    assert {row["split"] for row in first_test} == {"test"}


def test_exact_split_fails_when_subject_counts_cannot_reach_target() -> None:
    with pytest.raises(RuntimeError, match="exactly 1 cases"):
        module.choose_exact_test_subjects({"a": 2, "b": 2}, seed=1, target_cases=1)


def test_normalize_metadata_rejects_case_subject_mismatch() -> None:
    value = {
        "case_id": "BraTS-GLI-00001-100",
        "patient_id": "BraTS-GLI-00002",
        "source_files": {},
    }
    with pytest.raises(ValueError, match="case/subject mismatch"):
        module.normalize_metadata(value, Path("case.json"))
