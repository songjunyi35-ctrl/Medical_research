from __future__ import annotations

from pathlib import Path

import pytest

import scripts.audit_mu_glioma_post_metadata as audit


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "archive/reviews/2026-09-02_mu_glioma_post_followup/source_metadata"


def test_column_index_handles_excel_base26() -> None:
    assert audit.column_index("A1") == 0
    assert audit.column_index("Z9") == 25
    assert audit.column_index("AA10") == 26
    assert audit.column_index("BV2") == 73


def test_official_metadata_counts_and_no_cross_dataset_bridge() -> None:
    result = audit.audit(SOURCE / "MU-Glioma-Post_ClinicalData-July2025.xlsx", SOURCE / "MU-Glioma-Post_Segmentation_Volumes.xlsx")
    assert result["clinical"] == {"rows": 335, "unique_patients": 203, "timepoint_columns": 6, "nonempty_timepoint_cells": 597}
    assert result["clinical_minus_reported_timepoints"] == 1
    assert result["cross_dataset_identifier_bridge_found"] is False
    assert result["nifti_or_segmentation_voxel_access"] is False


def test_hash_mismatch_fails_before_workbook_parse(tmp_path: Path) -> None:
    bad = tmp_path / "bad.xlsx"
    bad.write_bytes(b"not an xlsx")
    with pytest.raises(audit.MetadataAuditError, match="SHA-256"):
        audit.read_xlsx(bad, audit.CLINICAL_SHA256)
