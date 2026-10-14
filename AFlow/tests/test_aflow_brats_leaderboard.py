from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.verify_aflow_brats_leaderboard import (
    ROOT,
    validate_leaderboard,
    validate_evidence_references,
    validate_ranked_rows,
)


def test_current_canonical_leaderboard_is_valid() -> None:
    value = validate_leaderboard(
        ROOT / "AFLOW_BRATS_LEADERBOARD.json",
        ROOT / "AFLOW_BRATS_LEADERBOARD.md",
    )
    assert value["test270_direct_locator_rank_by_scan"][0]["scan_dice"] > 0.74
    external = value["external_ucsd75_direct_locator_rank_by_scan"]
    assert external[0]["method"] == "gcsam_inner_fit864_epoch184_official_float32"
    assert len(external) == 4
    assert all(row["cohort"] == external[0]["cohort"] for row in external)
    assert {row["input_source_version"] for row in external} == {
        "direct_TCIA_package_1019_float32",
        "Pittsburgh_Fiber_Data_Hub_mirror_uint8",
    }
    assert external[0]["cohort"]["existing_subject_overlap"] == 0
    africa = value["external_brats_africa95_direct_locator_rank_by_scan"]
    assert len(africa) == 3
    assert africa[0]["method"] == "gcsam_inner_fit864_epoch184_identity"
    assert africa[0]["scan_dice"] > africa[1]["scan_dice"] > africa[2]["scan_dice"]
    assert africa[0]["cohort"]["institution_disjoint"] is True
    assert africa[0]["cohort"]["patient_id_bridge"] is False
    cptac = value["external_cptac33_direct_locator_rank_by_scan"]
    assert len(cptac) == 3 and cptac[0]["method"].endswith("flip8_tta")
    assert all(row["point_hits"] == row["point_denominator"] == 33 for row in cptac)
    assert cptac[0]["scan_dice"] > cptac[1]["scan_dice"] > cptac[2]["scan_dice"]
    waw = value["external_waw_gliomai152_direct_locator_rank_by_scan"]
    assert len(waw) == 3 and waw[0]["method"] == "gcsam_inner_fit864_epoch184_identity"
    assert waw[0]["scan_dice"] > waw[1]["scan_dice"] > waw[2]["scan_dice"]
    assert [row["point_hits"] for row in waw] == [140, 143, 141]
    assert waw[0]["cohort"]["institution_disjoint"] is True
    assert waw[0]["cohort"]["patient_id_bridge"] is False
    postop = value["external_waw_postop_only31_direct_locator_rank_by_scan"]
    assert len(postop) == 3 and postop[0]["method"].endswith("flip8_tta")
    assert postop[0]["scan_dice"] > postop[1]["scan_dice"] > postop[2]["scan_dice"]
    assert [row["point_hits"] for row in postop] == [15, 14, 15]
    assert postop[0]["cohort"]["preop_patient_overlap"] == 0


def test_unsorted_rows_are_rejected() -> None:
    rows = [
        {"rank": 1, "method": "a", "scan_dice": 0.7, "subject_dice": 0.7},
        {"rank": 2, "method": "b", "scan_dice": 0.8, "subject_dice": 0.8},
    ]
    with pytest.raises(ValueError, match="not sorted"):
        validate_ranked_rows("fake", rows)


def test_duplicate_methods_are_rejected() -> None:
    rows = [
        {"rank": 1, "method": "a", "scan_dice": 0.8, "subject_dice": 0.8},
        {"rank": 2, "method": "a", "scan_dice": 0.7, "subject_dice": 0.7},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        validate_ranked_rows("fake", rows)


def test_external_overlap_is_rejected() -> None:
    value = json.loads((ROOT / "AFLOW_BRATS_LEADERBOARD.json").read_text())
    value["external_ucsd75_direct_locator_rank_by_scan"][0]["cohort"]["existing_subject_overlap"] = 1
    with pytest.raises(ValueError, match="overlaps"):
        validate_evidence_references(value)


def test_external_point_fraction_must_match_counts() -> None:
    value = json.loads((ROOT / "AFLOW_BRATS_LEADERBOARD.json").read_text())
    value["external_ucsd75_direct_locator_rank_by_scan"][0]["point_hits"] -= 1
    with pytest.raises(ValueError, match="point fraction"):
        validate_evidence_references(value)


def test_report_hash_mismatch_is_rejected() -> None:
    value = json.loads((ROOT / "AFLOW_BRATS_LEADERBOARD.json").read_text())
    value["external_ucsd75_direct_locator_rank_by_scan"][0]["report_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        validate_evidence_references(value)


def test_external_source_version_pairing_is_required() -> None:
    value = json.loads((ROOT / "AFLOW_BRATS_LEADERBOARD.json").read_text())
    value["external_ucsd75_direct_locator_rank_by_scan"][0]["input_source_version"] = "unknown"
    with pytest.raises(ValueError, match="source versions"):
        validate_evidence_references(value)
