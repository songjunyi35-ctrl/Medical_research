#!/usr/bin/env python3
"""Fail fast when the canonical AFlow leaderboard is malformed or unsorted."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "AFLOW_BRATS_LEADERBOARD.json"
DEFAULT_MARKDOWN = ROOT / "AFLOW_BRATS_LEADERBOARD.md"
RANKED_KEYS = (
    "external_ucsd75_direct_locator_rank_by_scan",
    "external_brats_africa95_direct_locator_rank_by_scan",
    "external_cptac33_direct_locator_rank_by_scan",
    "external_waw_gliomai152_direct_locator_rank_by_scan",
    "external_waw_postop_only31_direct_locator_rank_by_scan",
    "external_ucsd_brats2024test56_direct_locator_rank_by_scan",
    "test270_direct_locator_rank_by_scan",
    "validation108_direct_locator_rank_by_scan",
)


def validate_ranked_rows(name: str, rows: Any) -> None:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{name} must be a non-empty list")
    expected_ranks = list(range(1, len(rows) + 1))
    actual_ranks = [row.get("rank") for row in rows]
    if actual_ranks != expected_ranks:
        raise ValueError(f"{name} ranks are not contiguous: {actual_ranks}")
    methods = [row.get("method") for row in rows]
    if any(not isinstance(method, str) or not method for method in methods):
        raise ValueError(f"{name} contains an empty method identifier")
    if len(methods) != len(set(methods)):
        raise ValueError(f"{name} contains duplicate methods")
    scans = [row.get("scan_dice") for row in rows]
    if any(not isinstance(value, (int, float)) for value in scans):
        raise ValueError(f"{name} contains a nonnumeric scan_dice")
    if any(left < right for left, right in zip(scans, scans[1:])):
        raise ValueError(f"{name} is not sorted by descending scan_dice")
    for row in rows:
        if not isinstance(row.get("subject_dice"), (int, float)):
            raise ValueError(f"{name}/{row.get('method')} lacks subject_dice")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_evidence_references(value: dict[str, Any], root: Path = ROOT) -> None:
    for name in RANKED_KEYS:
        for row in value[name]:
            report_path = row.get("report_path")
            report_sha256 = row.get("report_sha256")
            if report_sha256 is not None and report_path is None:
                raise ValueError(f"{name}/{row['method']} has report hash without path")
            if report_path is not None:
                relative = Path(str(report_path))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"{name}/{row['method']} report path escapes repository")
                resolved = root / relative
                if not resolved.is_file():
                    raise ValueError(f"{name}/{row['method']} report path is missing")
                if report_sha256 is not None and sha256_file(resolved) != report_sha256:
                    raise ValueError(f"{name}/{row['method']} report SHA-256 mismatch")
            cohort = row.get("cohort")
            if cohort is not None:
                if not isinstance(cohort, dict):
                    raise ValueError(f"{name}/{row['method']} cohort is not an object")
                for key in ("scans", "subjects"):
                    if not isinstance(cohort.get(key), int) or cohort[key] <= 0:
                        raise ValueError(f"{name}/{row['method']} has invalid cohort {key}")

    external = value["external_ucsd75_direct_locator_rank_by_scan"]
    frozen_cohort = external[0].get("cohort")
    for row in external:
        if row.get("cohort") != frozen_cohort:
            raise ValueError("external UCSD rows do not share one frozen cohort")
        cohort = row["cohort"]
        if (cohort.get("scans"), cohort.get("subjects")) != (75, 53):
            raise ValueError("external UCSD cohort is not 75 scans / 53 subjects")
        if cohort.get("existing_case_overlap") != 0 or cohort.get("existing_subject_overlap") != 0:
            raise ValueError("external UCSD cohort overlaps existing evidence roles")
        hits = row.get("point_hits")
        denominator = row.get("point_denominator")
        if not isinstance(hits, int) or not isinstance(denominator, int) or denominator <= 0:
            raise ValueError(f"external/{row['method']} point counts missing")
        if abs(float(row.get("point_hit")) - hits / denominator) > 1e-15:
            raise ValueError(f"external/{row['method']} point fraction disagrees with counts")
        evidence = str(row.get("evidence_status", ""))
        for token in ("external_confirmation", "no_external_tuning", "challenge_test_package_excluded"):
            if token not in evidence:
                raise ValueError(f"external/{row['method']} evidence status lacks {token}")
        caveat = str(row.get("comparability_note", ""))
        if "256" not in caveat or "182x218x182" not in caveat or "rasterization" not in caveat:
            raise ValueError(f"external/{row['method']} lacks grid/FOV caveat")
        for flag in ("tta", "sam", "postprocessing"):
            if row.get(flag) is not False:
                raise ValueError(f"external/{row['method']} unexpected {flag}")
    if "official_float_source_correction_20260906" in value:
        source_versions = [row.get("input_source_version") for row in external]
        expected_sources = {
            "direct_TCIA_package_1019_float32",
            "Pittsburgh_Fiber_Data_Hub_mirror_uint8",
        }
        if set(source_versions) != expected_sources:
            raise ValueError("external UCSD source versions are missing or unexpected")
        if any(source_versions.count(source) != 2 for source in expected_sources):
            raise ValueError("external UCSD source versions must each contain both frozen models")
        checkpoint_sources: dict[str, set[str]] = {}
        for row in external:
            checkpoint_sources.setdefault(str(row.get("checkpoint_sha256")), set()).add(str(row.get("input_source_version")))
            evidence = str(row.get("evidence_status", ""))
            if row.get("input_source_version") == "direct_TCIA_package_1019_float32" and "reused_source_correction" not in evidence:
                raise ValueError("official-float row is not labeled reused source correction")
            if row.get("input_source_version") == "Pittsburgh_Fiber_Data_Hub_mirror_uint8" and "mirror_uint8" not in evidence:
                raise ValueError("mirror row is not labeled mirror uint8")
        if len(checkpoint_sources) != 2 or any(sources != expected_sources for sources in checkpoint_sources.values()):
            raise ValueError("each frozen external model must have official-float and mirror rows")

    africa = value["external_brats_africa95_direct_locator_rank_by_scan"]
    if len(africa) != 3:
        raise ValueError("BraTS-Africa external list must contain three frozen rows")
    africa_cohort = africa[0].get("cohort")
    for row in africa:
        if row.get("cohort") != africa_cohort or (africa_cohort.get("scans"), africa_cohort.get("subjects")) != (95, 95):
            raise ValueError("BraTS-Africa rows do not share 95 scans/subjects")
        if africa_cohort.get("source_centers") != 4 or africa_cohort.get("institution_disjoint") is not True or africa_cohort.get("patient_id_bridge") is not False:
            raise ValueError("BraTS-Africa institution/patient-ID caveat missing")
        if row.get("input_source_version") != "TCIA_package948_official_NIfTI":
            raise ValueError("BraTS-Africa source version mismatch")
        if row.get("sam") is not False or row.get("postprocessing") is not False:
            raise ValueError("BraTS-Africa row has unexpected SAM/postprocessing")
        hits, denominator = row.get("point_hits"), row.get("point_denominator")
        if not isinstance(hits, int) or denominator != 95 or abs(float(row.get("point_hit")) - hits / denominator) > 1e-15:
            raise ValueError("BraTS-Africa point fraction mismatch")
        evidence = str(row.get("evidence_status", ""))
        for token in ("public_postchallenge", "not_hidden_blind", "no_external_tuning", "package948"):
            if token not in evidence:
                raise ValueError(f"BraTS-Africa evidence lacks {token}")
        caveat = str(row.get("comparability_note", ""))
        if "3-5mm" not in caveat or "128^3" not in caveat or "preoperative" not in caveat:
            raise ValueError("BraTS-Africa grid/target caveat missing")
        source_caveat = str(row.get("training_source_caveat", ""))
        if "Kaggle mirror" not in source_caveat or "parity unresolved" not in source_caveat:
            raise ValueError("BraTS-Africa training-source caveat missing")
    expected_views = [(1, False), (8, True), (1, False)]
    if [(row.get("locator_forwards"), row.get("tta")) for row in africa] != expected_views:
        raise ValueError("BraTS-Africa frozen forward/view rows mismatch")

    cptac = value["external_cptac33_direct_locator_rank_by_scan"]
    if len(cptac) != 3:
        raise ValueError("CPTAC external list must contain three rows")
    cohort = cptac[0].get("cohort")
    for row in cptac:
        if row.get("cohort") != cohort or (cohort.get("scans"), cohort.get("subjects")) != (33,33):
            raise ValueError("CPTAC rows do not share 33 cases")
        if cohort.get("source_disjoint") is not True or cohort.get("patient_id_bridge") is not False or cohort.get("hidden_blind") is not False:
            raise ValueError("CPTAC provenance caveat")
        if row.get("input_source_version") != "TCIA_BraTS2021_package636_official_NIfTI":
            raise ValueError("CPTAC source version")
        if row.get("point_hits") != 33 or row.get("point_denominator") != 33 or row.get("point_hit") != 1.0:
            raise ValueError("CPTAC point counts")
        if row.get("sam") is not False or row.get("postprocessing") is not False:
            raise ValueError("CPTAC protocol")
        evidence=str(row.get("evidence_status", ""))
        for token in ("africa_informed", "public_training", "not_hidden_blind", "no_external_tuning", "package636"):
            if token not in evidence: raise ValueError("CPTAC evidence lacks "+token)
        caveat=str(row.get("comparability_note", ""))
        if "public Training" not in caveat or "128^3" not in caveat or "patient-ID bridge" not in caveat:
            raise ValueError("CPTAC caveat")
        source_caveat=str(row.get("training_source_caveat", ""))
        if "Kaggle mirror" not in source_caveat or "parity unresolved" not in source_caveat:
            raise ValueError("CPTAC training source")
    if [(row.get("locator_forwards"),row.get("tta")) for row in cptac] != [(8,True),(1,False),(1,False)]:
        raise ValueError("CPTAC views")

    waw = value["external_waw_gliomai152_direct_locator_rank_by_scan"]
    if len(waw) != 3:
        raise ValueError("WAW-GliomAI external list must contain three rows")
    cohort = waw[0].get("cohort")
    for row in waw:
        if row.get("cohort") != cohort or (cohort.get("scans"), cohort.get("subjects")) != (152, 152):
            raise ValueError("WAW rows do not share 152 cases")
        if cohort.get("institution_disjoint") is not True or cohort.get("patient_id_bridge") is not False or cohort.get("hidden_blind") is not False:
            raise ValueError("WAW provenance caveat")
        if row.get("input_source_version") != "Zenodo22205631_final_coreg_delayed_postcontrast_t1c":
            raise ValueError("WAW source version")
        hits, denominator = row.get("point_hits"), row.get("point_denominator")
        if not isinstance(hits, int) or denominator != 152 or abs(float(row.get("point_hit")) - hits / denominator) > 1e-15:
            raise ValueError("WAW point fraction")
        if row.get("sam") is not False or row.get("postprocessing") is not False:
            raise ValueError("WAW protocol")
        evidence = str(row.get("evidence_status", ""))
        for token in ("public_external_to_project", "institution_disjoint_no_patient_id_bridge", "not_hidden_blind", "no_external_tuning", "Zenodo"):
            if token not in evidence:
                raise ValueError("WAW evidence lacks " + token)
        caveat = str(row.get("comparability_note", ""))
        for token in ("delayed post-contrast T1c", "N4", "128^3", "preoperative"):
            if token not in caveat:
                raise ValueError("WAW caveat lacks " + token)
        source_caveat = str(row.get("training_source_caveat", ""))
        if "Kaggle mirror" not in source_caveat or "parity unresolved" not in source_caveat:
            raise ValueError("WAW training source")
    if [(row.get("locator_forwards"), row.get("tta")) for row in waw] != [(1, False), (8, True), (1, False)]:
        raise ValueError("WAW views")

    postop = value["external_waw_postop_only31_direct_locator_rank_by_scan"]
    if len(postop) != 3:
        raise ValueError("WAW postop list must contain three rows")
    cohort = postop[0].get("cohort")
    for row in postop:
        if row.get("cohort") != cohort or (cohort.get("scans"), cohort.get("subjects")) != (31, 31):
            raise ValueError("WAW postop rows do not share31 cases")
        if cohort.get("preop_patient_overlap") != 0 or cohort.get("days_postop_min") != 0.0 or cohort.get("days_postop_max") != 22.0:
            raise ValueError("WAW postop cohort selection")
        if cohort.get("institution_disjoint") is not True or cohort.get("patient_id_bridge") is not False or cohort.get("hidden_blind") is not False:
            raise ValueError("WAW postop provenance")
        if row.get("input_source_version") != "Zenodo22205631_final_postop_coreg_t1c":
            raise ValueError("WAW postop source")
        hits, denominator = row.get("point_hits"), row.get("point_denominator")
        if not isinstance(hits, int) or denominator != 31 or abs(float(row.get("point_hit")) - hits / denominator) > 1e-15:
            raise ValueError("WAW postop point fraction")
        evidence = str(row.get("evidence_status", ""))
        for token in ("patient_disjoint_from_revealed_waw_preop", "not_hidden_blind", "no_external_tuning", "official_Zenodo_MD5"):
            if token not in evidence:
                raise ValueError("WAW postop evidence lacks " + token)
        caveat = str(row.get("comparability_note", ""))
        for token in ("day0-22", "resection-bed4 excluded", "128^3", "earliest numeric"):
            if token not in caveat:
                raise ValueError("WAW postop caveat lacks " + token)
        if row.get("sam") is not False or row.get("postprocessing") is not False:
            raise ValueError("WAW postop protocol")
    if [(row.get("locator_forwards"), row.get("tta")) for row in postop] != [(8, True), (1, False), (1, False)]:
        raise ValueError("WAW postop views")

    brats24 = value["external_ucsd_brats2024test56_direct_locator_rank_by_scan"]
    if len(brats24) != 3:
        raise ValueError("UCSD BraTS2024test list must contain three rows")
    cohort = brats24[0].get("cohort")
    for row in brats24:
        if row.get("cohort") != cohort or (cohort.get("scans"), cohort.get("subjects")) != (56, 41):
            raise ValueError("UCSD BraTS2024test rows do not share 56/41 cohort")
        if cohort.get("union_patient_overlap") != 0 or cohort.get("prior_ucsd_main_release_overlap") != 0:
            raise ValueError("UCSD BraTS2024test patient exclusion")
        if cohort.get("institution_disjoint") is not False or cohort.get("patient_id_bridge") is not True or cohort.get("hidden_blind") is not False:
            raise ValueError("UCSD BraTS2024test provenance flags")
        if row.get("input_source_version") != "TCIA_UCSD-PTGBM_package1221_V2_20260122_clinicalV3_20260313":
            raise ValueError("UCSD BraTS2024test source version")
        hits, denominator = row.get("point_hits"), row.get("point_denominator")
        if not isinstance(hits, int) or denominator != 56 or abs(float(row.get("point_hit")) - hits / denominator) > 1e-15:
            raise ValueError("UCSD BraTS2024test point fraction")
        evidence = str(row.get("evidence_status", ""))
        for token in ("public_post_challenge", "patient_disjoint_from_1350_union", "not_institution_disjoint", "not_hidden_blind", "no_external_tuning", "dual_transfer_sha256"):
            if token not in evidence:
                raise ValueError("UCSD BraTS2024test evidence lacks " + token)
        caveat = str(row.get("comparability_note", ""))
        for token in ("post-treatment", "resection cavity4 excluded", "128^3", "not cross-institution"):
            if token not in caveat:
                raise ValueError("UCSD BraTS2024test caveat lacks " + token)
        source_caveat = str(row.get("training_source_caveat", ""))
        if "Kaggle mirror" not in source_caveat or "parity unresolved" not in source_caveat:
            raise ValueError("UCSD BraTS2024test training source")
        if row.get("sam") is not False or row.get("postprocessing") is not False:
            raise ValueError("UCSD BraTS2024test protocol")
    if [(row.get("locator_forwards"), row.get("tta")) for row in brats24] != [(8, True), (1, False), (1, False)]:
        raise ValueError("UCSD BraTS2024test views")


def validate_leaderboard(json_path: Path, markdown_path: Path) -> dict[str, Any]:
    value = json.loads(json_path.read_text(encoding="utf-8"))
    if value.get("schema") != "aflow.brats.t1ce_wt_leaderboard":
        raise ValueError("unexpected leaderboard schema")
    if not value.get("updated_utc"):
        raise ValueError("leaderboard lacks updated_utc")
    for key in RANKED_KEYS:
        validate_ranked_rows(key, value.get(key))
    validate_evidence_references(value)
    warning = str(value.get("evidence_warning", "")).lower()
    if "reused" not in warning or "blind" not in warning:
        raise ValueError("leaderboard must warn that test270 is reused/not blind")
    source_warning = str(value.get("training_source_warning", ""))
    if "Kaggle mirror" not in source_warning or "parity is unresolved" not in source_warning:
        raise ValueError("leaderboard must preserve inherited training-source caveat")
    markdown = markdown_path.read_text(encoding="utf-8")
    required_markdown = (
        "# AFlow BraTS T1ce/WT Leaderboard",
        "External UCSD-PTGBM75",
        "External BraTS-Africa95",
        "External CPTAC33",
        "External WAW-GliomAI152",
        "External WAW-postop-only31",
        "Reused internal test270",
        "Fixed validation108",
        "非同口径test历史",
    )
    for text in required_markdown:
        if text not in markdown:
            raise ValueError(f"leaderboard Markdown lacks section/text: {text}")
    top_test = value["test270_direct_locator_rank_by_scan"][0]
    top_single = next(
        row
        for row in value["test270_direct_locator_rank_by_scan"]
        if row.get("locator_forwards") == 1
    )
    for row in (top_test, top_single):
        rendered = f"{float(row['scan_dice']):.6f}"
        if rendered not in markdown:
            raise ValueError(
                f"leaderboard Markdown does not contain headline scan value {rendered}"
            )
    top_external = value["external_ucsd75_direct_locator_rank_by_scan"][0]
    if f"{float(top_external['scan_dice']):.6f}" not in markdown:
        raise ValueError("leaderboard Markdown lacks external headline scan value")
    top_africa = value["external_brats_africa95_direct_locator_rank_by_scan"][0]
    if f"{float(top_africa['scan_dice']):.6f}" not in markdown:
        raise ValueError("leaderboard Markdown lacks BraTS-Africa headline scan value")
    top_cptac = value["external_cptac33_direct_locator_rank_by_scan"][0]
    if f"{float(top_cptac['scan_dice']):.6f}" not in markdown:
        raise ValueError("leaderboard Markdown lacks CPTAC headline scan value")
    top_waw = value["external_waw_gliomai152_direct_locator_rank_by_scan"][0]
    if f"{float(top_waw['scan_dice']):.6f}" not in markdown:
        raise ValueError("leaderboard Markdown lacks WAW headline scan value")
    top_postop = value["external_waw_postop_only31_direct_locator_rank_by_scan"][0]
    if f"{float(top_postop['scan_dice']):.6f}" not in markdown:
        raise ValueError("leaderboard Markdown lacks WAW postop headline scan value")
    top_brats24 = value["external_ucsd_brats2024test56_direct_locator_rank_by_scan"][0]
    if f"{float(top_brats24['scan_dice']):.6f}" not in markdown:
        raise ValueError("leaderboard Markdown lacks UCSD BraTS2024test headline scan value")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    value = validate_leaderboard(args.json, args.markdown)
    print(
        json.dumps(
            {
                "status": "passed",
                "updated_utc": value["updated_utc"],
                "test_rows": len(value["test270_direct_locator_rank_by_scan"]),
                "external_rows": len(value["external_ucsd75_direct_locator_rank_by_scan"]),
                "africa_external_rows": len(value["external_brats_africa95_direct_locator_rank_by_scan"]),
                "cptac_external_rows": len(value["external_cptac33_direct_locator_rank_by_scan"]),
                "waw_external_rows": len(value["external_waw_gliomai152_direct_locator_rank_by_scan"]),
                "waw_postop_external_rows": len(value["external_waw_postop_only31_direct_locator_rank_by_scan"]),
                "ucsd_brats2024test_external_rows": len(value["external_ucsd_brats2024test56_direct_locator_rank_by_scan"]),
                "validation_rows": len(
                    value["validation108_direct_locator_rank_by_scan"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
