#!/usr/bin/env python3
"""Validate the complete frozen UCSD-PTGBM external-confirmation evidence chain."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "archive/research_records/2026-09-02_ucsd_ptgbm_external_confirmation"
WORK = ROOT / "workspace/BraTS/ucsd_ptgbm_external_confirmation_20260902"


class ExternalEvidenceValidationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExternalEvidenceValidationError(message)


def require_hash(path: Path, expected: str, label: str) -> None:
    require(path.is_file(), f"missing {label}: {path}")
    require(sha256_file(path) == expected, f"{label} SHA-256 mismatch")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def recompute(summary: dict[str, Any]) -> dict[str, Any]:
    rows = summary["per_case"]
    by_subject: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_subject[str(row["subject_id"])].append(float(row["locator_dsc"]))
    positives = [row for row in rows if row["target_wt_positive"]]
    return {
        "cases": len(rows),
        "subjects": len(by_subject),
        "scan_dice": statistics.fmean(float(row["locator_dsc"]) for row in rows),
        "subject_dice": statistics.fmean(statistics.fmean(values) for values in by_subject.values()),
        "point_hits": sum(bool(row["automatic_point_hits_gt"]) for row in positives),
        "point_denominator": len(positives),
        "point_hit": statistics.fmean(float(row["automatic_point_hits_gt"]) for row in positives),
        "case_ids": [str(row["case_id"]) for row in rows],
        "subject_ids": [str(row["subject_id"]) for row in rows],
    }


def validate(root: Path = ROOT) -> dict[str, Any]:
    record = root / RECORD.relative_to(ROOT)
    work = root / WORK.relative_to(ROOT)
    result = json.loads((record / "RESULT.json").read_text())
    evidence = result["evidence"]
    paths = {
        "raw_report": root / evidence["raw_report_path"],
        "preaccess": record / "PREACCESS_RECEIPT.json",
        "inventory": record / "OFFICIAL_ASSET_INVENTORY.json",
        "download": work / "raw/DOWNLOAD_RECEIPT.json",
        "cache_index": work / "cache128/index.json",
        "postrun": root / evidence["postrun_audit_path"],
        "development_overlap": root / evidence["development_overlap_label_audit_path"],
        "development_delta": root / evidence["development_overlap_label_delta_audit_path"],
        "descriptive_sensitivity": root / evidence["descriptive_sensitivity_audit_path"],
        "power_planning": root / evidence["power_planning_audit_path"],
        "cache_reproduction": root / evidence["cache_reproduction_audit_path"],
    }
    hash_fields = {
        "raw_report": "raw_report_sha256",
        "preaccess": "preaccess_receipt_sha256",
        "inventory": "official_inventory_sha256",
        "download": "download_receipt_sha256",
        "cache_index": "cache_index_sha256",
        "postrun": "postrun_audit_sha256",
        "development_overlap": "development_overlap_label_audit_sha256",
        "development_delta": "development_overlap_label_delta_audit_sha256",
        "descriptive_sensitivity": "descriptive_sensitivity_audit_sha256",
        "power_planning": "power_planning_audit_sha256",
        "cache_reproduction": "cache_reproduction_audit_sha256",
    }
    for name, path in paths.items():
        require_hash(path, evidence[hash_fields[name]], name)

    preaccess = json.loads(paths["preaccess"].read_text())
    require(preaccess["status"] == "frozen_before_eligible_external_label_download", "preaccess status mismatch")
    require(preaccess["access"]["eligible_label_voxel_access"] is False, "preaccess receipt declares label access")
    for section, key in (("prepare_script", "prepare_script"), ("evaluator", "evaluator"), ("tests", "tests")):
        item = preaccess["implementation"][section]
        require_hash(root / item["path"], item["sha256"], key)
    for key in ("idea", "preregistration"):
        item = preaccess["method_documents"][key]
        require_hash(root / item["path"], item["sha256"], key)
    for key in ("synthetic_tests", "gpu_smoke"):
        item = preaccess["implementation"][key]
        require_hash(root / item["path"], item["sha256"], key)
    for role in ("primary", "secondary"):
        item = preaccess["checkpoints"][role]
        require_hash(root / item["path"], item["sha256"], f"{role} checkpoint")

    inventory = json.loads(paths["inventory"].read_text())
    download = json.loads(paths["download"].read_text())
    selection = inventory["selection"]
    require((selection["cases"], selection["subjects"], selection["case_overlap"], selection["subject_overlap"]) == (75, 53, 0, 0), "inventory cohort/overlap mismatch")
    require(inventory["source"]["separate_brats_challenge_test_package_used"] is False, "challenge test package entered inventory")
    selected_cases = [row["brats_case_id"] for row in selection["selected"]]
    selected_subjects = [row["brats_subject_id"] for row in selection["selected"]]
    downloaded_by_case = {row["ucsd_case_id"]: row for row in download["records"]}
    require(set(downloaded_by_case) == {row["ucsd_case_id"] for row in selection["selected"]}, "download receipt cases differ from inventory")
    raw_bytes = 0
    for row in selection["selected"]:
        receipt = downloaded_by_case[row["ucsd_case_id"]]
        for kind in ("image", "label"):
            asset = row[kind]
            item = receipt["files"][kind]
            path = Path(item["path"])
            require(path.name == asset["name"], f"raw {kind} filename differs from official asset")
            require(path.stat().st_size == asset["size_bytes"] == item["size_bytes"], f"raw {kind} size mismatch")
            require_hash(path, asset["sha256"], f"raw {kind} asset")
            require(item["sha256"] == asset["sha256"], f"download receipt {kind} hash differs from inventory")
            raw_bytes += path.stat().st_size
    require(raw_bytes == selection["selected_assets_bytes"] == download["downloaded_bytes"], "selected raw byte total mismatch")

    train_path = root / "data/datasets/brats2024_random_subject_80_20_seed20260722_v1/train_cases.jsonl"
    test_path = root / "data/datasets/brats2024_random_subject_80_20_seed20260722_v1/sealed_test/test_cases.jsonl"
    exposure = read_jsonl(train_path) + read_jsonl(test_path)
    exposed_cases = {str(row["case_id"]) for row in exposure}
    exposed_subjects = {str(row["subject_id"]) for row in exposure}
    require(not (set(selected_cases) & exposed_cases), "external selection now overlaps an exposed case")
    require(not (set(selected_subjects) & exposed_subjects), "external selection now overlaps an exposed subject")

    cache = json.loads(paths["cache_index"].read_text())
    require(cache["case_ids"] == selected_cases, "cache case order differs from inventory")
    require(cache["subject_ids"] == selected_subjects, "cache subject order differs from inventory")
    require(cache["count"] == 75 and cache["subjects"] == 53, "cache cohort count mismatch")
    require(cache["qc"]["all_raw_labels"] == [0, 1, 2, 3, 4], "cache raw label domain mismatch")
    require(cache["qc"]["all_finite"] is True, "cache finite gate failed")
    require_hash(work / "cache128/images.f32", cache["images_sha256"], "external image tensor")
    require_hash(work / "cache128/labels.u8", cache["labels_sha256"], "external label tensor")

    raw = json.loads(paths["raw_report"].read_text())
    require(raw["external_confirmation"] is True and raw["not_test_tuned"] is True, "raw report evidence scope mismatch")
    require(raw["data"]["existing_case_overlap"] == 0 and raw["data"]["existing_subject_overlap"] == 0, "raw report overlap mismatch")
    metrics = {role: recompute(raw[role]["summary"]) for role in ("primary", "secondary")}
    for role, result_key in (("primary", "primary_full972_kd"), ("secondary", "secondary_gcsam")):
        recomputed = metrics[role]
        recorded = result[result_key]
        require(recomputed["cases"] == 75 and recomputed["subjects"] == 53, f"{role} report cohort mismatch")
        for metric in ("scan_dice", "subject_dice", "point_hit"):
            require(recomputed[metric] == recorded[metric], f"{role} {metric} recomputation mismatch")
        require(recomputed["point_hits"] == recorded["point_hits"] and recomputed["point_denominator"] == recorded["point_denominator"], f"{role} point counts mismatch")
        require(recomputed["case_ids"] == selected_cases and recomputed["subject_ids"] == selected_subjects, f"{role} ID order mismatch")

    leaderboard = json.loads((root / "AFLOW_BRATS_LEADERBOARD.json").read_text())
    external = leaderboard["external_ucsd75_direct_locator_rank_by_scan"]
    require([row["rank"] for row in external] == list(range(1, len(external) + 1)), "external leaderboard ranks mismatch")
    mirror = [row for row in external if row.get("input_source_version") == "Pittsburgh_Fiber_Data_Hub_mirror_uint8"]
    require(len(mirror) == 2, "historical mirror leaderboard row count mismatch")
    require(
        [row["method"] for row in mirror]
        == ["gcsam_inner_fit864_epoch184_mirror_uint8", "full972_random_init_voxel_kd_epoch168_mirror_uint8"],
        "historical mirror leaderboard canonical method order mismatch",
    )
    require(
        raw["secondary"]["identifier"] == "gcsam_fit864_inner_selected_epoch184"
        and raw["primary"]["identifier"] == "full972_random_init_voxel_kd_epoch168",
        "raw report method identity mismatch",
    )
    expected = [("secondary", mirror[0]), ("primary", mirror[1])]
    for role, row in expected:
        computed = metrics[role]
        require(row["scan_dice"] == computed["scan_dice"], f"leaderboard {role} scan mismatch")
        require(row["subject_dice"] == computed["subject_dice"], f"leaderboard {role} subject mismatch")
        require(row["point_hit"] == computed["point_hit"], f"leaderboard {role} point mismatch")
        require(row["report_sha256"] == evidence["raw_report_sha256"], f"leaderboard {role} report hash mismatch")
        require(row["cohort"]["existing_case_overlap"] == 0 and row["cohort"]["existing_subject_overlap"] == 0, f"leaderboard {role} overlap mismatch")

    development = json.loads(paths["development_overlap"].read_text())
    delta = json.loads(paths["development_delta"].read_text())
    postrun = json.loads(paths["postrun"].read_text())
    require(development["reused_test_label_or_image_downloaded"] is False, "development audit touched reused-test files")
    require((development["centered_arrays_exact_cases"], delta["minor_revision_cases"]) == (76, 5), "development label identity summary mismatch")
    require(delta["minimum_wt_dice"] > 0.9997 and delta["maximum_changed_voxels"] == 36, "development label delta bound mismatch")
    subject_mapping = postrun["participants_subject_mapping_audit"]
    sensitivity = json.loads(paths["descriptive_sensitivity"].read_text())
    power = json.loads(paths["power_planning"].read_text())
    reproduction = json.loads(paths["cache_reproduction"].read_text())
    require(subject_mapping["selected_external_local_subject_overlap"] == 0, "external selection overlaps an exposed UCSD-local subject")
    require(subject_mapping["selected_external_brats_subject_overlap"] == 0, "external selection overlaps an exposed BraTS subject")
    require(subject_mapping["anomalous_local_subject_selected"] is False, "non-bijective participants subject entered external selection")
    require(sensitivity["source_raw_report_sha256"] == evidence["raw_report_sha256"], "descriptive sensitivity is bound to a different raw report")
    require(sensitivity["new_model_inference"] is False and sensitivity["result_dependent_tuning_or_rerun"] is False, "descriptive sensitivity crossed access boundary")
    require(power["source_raw_report_sha256"] == evidence["raw_report_sha256"], "power plan is bound to a different raw report")
    require(power["new_model_inference"] is False and power["result_dependent_tuning"] is False, "power plan crossed access boundary")
    require(reproduction["comparison"]["all_tensors_byte_exact"] is True, "external cache full reproduction failed")
    require(reproduction["temporary_directory_removed"] is True and reproduction["formal_cache_modified"] is False, "external cache reproduction cleanup/mutation gate failed")
    require(reproduction["model_inference"] is False and reproduction["reused_test270_access"] is False, "external cache reproduction crossed access boundary")

    return {
        "status": "passed",
        "external_cases": 75,
        "external_subjects": 53,
        "external_rows": 2,
        "raw_assets_rehashed": 150,
        "raw_asset_bytes_rehashed": raw_bytes,
        "development_overlap_labels_exact": 76,
        "development_overlap_labels_minor_revision": 5,
        "reused_test_files_accessed_by_overlap_audit": 0,
        "dual_namespace_subject_overlap": 0,
    }


def main() -> None:
    print(json.dumps(validate(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
