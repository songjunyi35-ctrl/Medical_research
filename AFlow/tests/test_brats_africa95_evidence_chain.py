from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "archive/research_records/2026-09-06_brats_africa95_frozen_baselines"
REVIEW = ROOT / "archive/reviews/2026-09-06_brats_africa_source_audit"


def load(path: Path): return json.loads(path.read_text())


def sha(path: Path): return hashlib.sha256(path.read_bytes()).hexdigest()


def test_official_inventory_and_frozen_categories_are_exact() -> None:
    inventory = load(REVIEW / "OFFICIAL_INVENTORY.json")
    cohort = load(REVIEW / "FROZEN_GLIOMA95_COHORT.json")
    assert len(inventory["files"]) == 730
    assert sum(row["size"] for row in inventory["files"]) == 1_727_632_847
    assert cohort["selection"]["cases"] == cohort["selection"]["subjects"] == 95
    assert cohort["selection"]["files"] == 190
    assert cohort["selection"]["bytes"] == 277_630_584
    assert cohort["excluded"]["cases"] == 51
    assert all("/95_Glioma/" in row[kind]["path"] for row in cohort["selection"]["rows"] for kind in ("t1ce", "seg"))


def test_dual_source_and_cache_chain_are_frozen() -> None:
    source = load(RECORD / "SOURCE_DOWNLOAD.json")
    cache = load(RECORD / "CACHE_RECEIPT.json")
    probe = load(RECORD / "FIRST_CASE_PROBE_V2.json")
    index = load(RECORD / "CACHE_INDEX_V2.json")
    assert source["status"] == "passed" and source["files_copy_exact"] == 190
    assert len(source["comparisons"]) == 190 and all(row["copy_exact"] for row in source["comparisons"])
    assert cache["all_gates_passed"] and cache["raw_label_value_union"] == [0, 1, 2, 3]
    assert probe["output_image_sha256"] == "35dfeeb829ee42472485bead18a7da966df1d7e1aa184602aa92bd87ee063c8a"
    assert probe["output_label_sha256"] == "fa65b9b7858946f85de7e19a14d6a1afb9bca6649ae5d9712637ac6f9e5bb315"
    assert index["gates"]["v2_probe_image_byte_exact"] and index["gates"]["v2_probe_label_byte_exact"]
    assert index["images_sha256"] == cache["images_sha256"]
    assert index["labels_sha256"] == cache["labels_sha256"]


def test_raw_result_center_correction_and_closed_result_are_consistent() -> None:
    raw = load(RECORD / "RAW_EXTERNAL95.json")
    correction = load(RECORD / "CENTER_PROVENANCE_CORRECTION.json")
    result = load(RECORD / "RESULT.json")
    assert sha(RECORD / "RAW_EXTERNAL95.json") == "c9fb07be4412c160129037104936e4c34b6390fa733b998b8a8f35524ccd30b3"
    assert raw["forwards"] == 855
    assert raw["gcsam184"]["identity"]["locator_dsc_mean"] == 0.5916192193392555
    assert raw["full972_kd168"]["identity"]["locator_dsc_mean"] == 0.5230025504563405
    assert raw["full972_kd168"]["flip8"]["automatic_point_hits"] == 88
    assert correction["package948_datainfo"]["glioma95_center_count"] == 4
    assert correction["metrics_changed"] is False and correction["case_selection_changed"] is False
    assert result["status"] == "completed_once_and_closed"
    assert result["decision"] == {"performance_promotion_gate": None, "tuning": False, "repeat": False, "cohort_closed": True}


def test_africa_leaderboard_is_separate_sorted_and_provenanced() -> None:
    board = load(ROOT / "AFLOW_BRATS_LEADERBOARD.json")
    rows = board["external_brats_africa95_direct_locator_rank_by_scan"]
    assert [row["rank"] for row in rows] == [1, 2, 3]
    assert [row["scan_dice"] for row in rows] == sorted([row["scan_dice"] for row in rows], reverse=True)
    assert all(row["cohort"]["source_centers"] == 4 for row in rows)
    assert all(row["report_sha256"] == "c9fb07be4412c160129037104936e4c34b6390fa733b998b8a8f35524ccd30b3" for row in rows)
    assert "external_brats_africa95_direct_locator_rank_by_scan" != "external_ucsd75_direct_locator_rank_by_scan"
