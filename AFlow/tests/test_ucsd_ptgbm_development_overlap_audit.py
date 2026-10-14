from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import scripts.audit_ucsd_ptgbm_development_overlap as audit


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "archive/research_records/2026-09-02_ucsd_ptgbm_external_confirmation"


def _real_args() -> object:
    source = RECORD / "source_metadata"
    return type("Args", (), {
        "participants": source / "participants.tsv",
        "release_json": [source / f"{tag}.release.json" for tag in audit.prepare.EXPECTED_RELEASES],
        "train_manifest": ROOT / "data/datasets/brats2024_random_subject_80_20_seed20260722_v1/train_cases.jsonl",
        "test_manifest": ROOT / "data/datasets/brats2024_random_subject_80_20_seed20260722_v1/sealed_test/test_cases.jsonl",
        "raw_root": Path("/mnt/hdfs/sjy/datasets/brats2024_gli_t1ce_seg_v1"),
    })()


def test_real_plan_excludes_every_reused_test_subject_before_download() -> None:
    plan = audit.build_plan(_real_args())
    assert plan["development_cases"] == 81
    assert plan["development_subjects"] == 63
    assert plan["reused_test_cases_excluded"] == 28
    assert plan["reused_test_subjects_excluded"] == 20
    selected_subjects = {row["brats_subject_id"] for row in plan["selected"]}
    excluded_subjects = {row["brats_subject_id"] for row in plan["excluded_reused_test_ids"]}
    assert selected_subjects.isdisjoint(excluded_subjects)
    assert plan["reused_test_label_or_image_download_authorized"] is False


def test_fixed_center_offset_recovers_development_grid(tmp_path: Path) -> None:
    development = np.zeros(audit.EXPECTED_BRATS_SHAPE, dtype=np.uint8)
    development[30:36, 50:60, 70:75] = 3
    official = np.zeros(audit.EXPECTED_UCSD_SHAPE, dtype=np.uint8)
    slices = tuple(slice(offset, offset + size) for offset, size in zip(audit.EXPECTED_CENTER_OFFSET, audit.EXPECTED_BRATS_SHAPE))
    official[slices] = development
    official_path = tmp_path / "official.nii.gz"
    development_path = tmp_path / "development.nii.gz"
    nib.save(nib.Nifti1Image(official, np.eye(4)), official_path)
    nib.save(nib.Nifti1Image(development, np.eye(4)), development_path)
    loaded_official, _ = audit._canonical_integer_label(official_path)
    loaded_development, _ = audit._canonical_integer_label(development_path)
    assert np.array_equal(loaded_official[slices], loaded_development)


def test_label_reader_rejects_out_of_contract_values(tmp_path: Path) -> None:
    label = np.zeros((8, 8, 8), dtype=np.uint8)
    label[1, 1, 1] = 5
    path = tmp_path / "bad.nii.gz"
    nib.save(nib.Nifti1Image(label, np.eye(4)), path)
    with pytest.raises(audit.OverlapAuditError, match="unexpected label"):
        audit._canonical_integer_label(path)
