from __future__ import annotations

from pathlib import Path

import pytest

import scripts.audit_rhuh_gbm_metadata as audit


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "archive/reviews/2026-09-02_rhuh_gbm_external_feasibility/source_metadata"


def test_official_rhuh_patient_metadata_contract() -> None:
    result = audit.audit(SOURCE / "clinical_data_TCIA_RHUH-GBM.csv", SOURCE / "GC_manifest_RHUH-GBM_20260326.csv")
    assert result["clinical_unique_patients"] == 40
    assert result["gdc_dicom_series_rows"] == 600
    assert result["gdc_series_per_patient"] == 15
    assert result["patient_id_sets_equal"] is True
    assert result["public_nifti_or_segmentation_inventory"] is False
    assert result["nifti_or_segmentation_voxel_access"] is False


def test_hash_mismatch_fails_before_csv_parse(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("not,the,official,file\n")
    with pytest.raises(audit.RhuhMetadataError, match="SHA-256"):
        audit._read(bad, audit.CLINICAL_SHA256)
