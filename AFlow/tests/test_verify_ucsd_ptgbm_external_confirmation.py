from __future__ import annotations

from scripts.verify_ucsd_ptgbm_external_confirmation import validate


def test_current_ucsd_external_evidence_chain_is_valid() -> None:
    result = validate()
    assert result["status"] == "passed"
    assert result["external_cases"] == 75
    assert result["external_subjects"] == 53
    assert result["reused_test_files_accessed_by_overlap_audit"] == 0
    assert result["dual_namespace_subject_overlap"] == 0
    assert result["raw_assets_rehashed"] == 150
    assert result["raw_asset_bytes_rehashed"] == 103182525
