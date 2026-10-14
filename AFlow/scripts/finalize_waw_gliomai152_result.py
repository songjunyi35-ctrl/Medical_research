#!/usr/bin/env python3
"""Recompute WAW152 saved metrics and archive a terminal evidence receipt."""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

import scripts.evaluate_brats_africa95_frozen_baselines as metrics
import scripts.evaluate_waw_gliomai152_panel as evaluator
from scripts import waw_gliomai152_protocol as p


metrics.p = p
RAW_SHA = "ef35301a78f6e47ac0574e5727aa7c5015b2f30cd73e1c8c45487fa8591dde91"
STARTED_SHA = "78437f60a22d99c8399e88faed76b47a96aea96168001bf49903f997ba0c6251"
PREACCESS_V2_SHA = "bea3a59e70fab071b29924d087c2671e87ac34adf281b0942471d8ef2e726858"
CACHE_RECEIPT_SHA = "6f50b413dba75ecbc8a9b2ade60fda2fa709c77106e3dfb1d6d3d343db8fd9c4"
ARCHIVE_SHA = "680a90c8bad679dad5f2bbf8dfdd7efd992a635de6f9b3f7a91422349f2b5887"
EVALUATOR_SHA = "e4098901777011b82bcc9502a677b85aab15a23cbcae1371fe4cda792edb140c"


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def recompute_summary(summary: dict) -> None:
    rows = summary["per_case"]
    require(len(rows) == len({row["case_id"] for row in rows}) == p.CASES, "case summary")
    mean = statistics.fmean(row["locator_dsc"] for row in rows)
    hits = sum(row["automatic_point_hits_gt"] for row in rows)
    require(abs(mean - summary["locator_dsc_mean"]) < 1e-12, "mean mismatch")
    require(summary["locator_subject_dsc_mean"] == summary["locator_dsc_mean"], "subject mismatch")
    require(hits == summary["automatic_point_hits"], "point mismatch")


def main() -> None:
    result_path = p.RECORD / "RESULT.json"
    receipt_path = p.RECORD / "VERIFICATION_RECEIPT.json"
    cache_archive_path = p.RECORD / "CACHE_INDEX_FINAL.json"
    if any(path.exists() for path in (result_path, receipt_path, cache_archive_path)):
        raise FileExistsError("terminal outputs already exist")
    p.docs()
    raw_path = p.RECORD / "RAW_WAW152.json"
    started_path = p.RECORD / "RAW_WAW152_STARTED.json"
    p.checked(raw_path, RAW_SHA)
    p.checked(started_path, STARTED_SHA)
    p.checked(p.MODEL_PREACCESS, PREACCESS_V2_SHA)
    p.checked(p.SOURCE_RECEIPT, p.SOURCE_RECEIPT_SHA)
    p.checked(p.CACHE_RECEIPT, CACHE_RECEIPT_SHA)
    p.checked(p.FULL, p.FULL_SHA)
    p.checked(p.GCSAM, p.GCSAM_SHA)

    raw = json.loads(raw_path.read_text())
    require(raw["forwards"] == 1368 and raw["evaluator_sha256"] == EVALUATOR_SHA, "raw contract")
    sections = [
        ("full972_kd168", "identity", p.BOOTSTRAP_SEED),
        ("full972_kd168", "flip8", p.BOOTSTRAP_SEED + 10),
        ("gcsam184", "identity", p.BOOTSTRAP_SEED + 20),
    ]
    for model, view, seed in sections:
        summary = raw[model][view]
        recompute_summary(summary)
        require(metrics.absolute(summary, seed) == raw[model][view + "_bootstrap"], "absolute bootstrap")
    full_identity = raw["full972_kd168"]["identity"]
    full_flip8 = raw["full972_kd168"]["flip8"]
    gcsam = raw["gcsam184"]["identity"]
    require(
        metrics.compare(full_identity, full_flip8, p.BOOTSTRAP_SEED + 100, "flip8_minus_identity")
        == raw["comparisons"]["full_flip8_minus_identity"],
        "Flip8 paired bootstrap",
    )
    require(
        metrics.compare(full_identity, gcsam, p.BOOTSTRAP_SEED + 200, "gcsam_minus_full")
        == raw["comparisons"]["gcsam_identity_minus_full_identity"],
        "GCSAM paired bootstrap",
    )
    require(
        evaluator.interpretation(raw["comparisons"]["full_flip8_minus_identity"], "flip8_minus_identity")
        == raw["interpretation_gates"]["full_flip8_minus_identity"],
        "Flip8 interpretation",
    )
    require(
        evaluator.interpretation(raw["comparisons"]["gcsam_identity_minus_full_identity"], "gcsam_minus_full")
        == raw["interpretation_gates"]["gcsam_identity_minus_full_identity"],
        "GCSAM interpretation",
    )

    source = json.loads(p.SOURCE_RECEIPT.read_text())
    archive = p.SOURCE_ROOT / "2_1_preop_scans_coreg.zip"
    require(archive.stat().st_size == 3164137716, "archive bytes")
    require(md5(archive) == "0d24a3a2c76bc6fa9b5b6468679676b1", "archive official MD5")
    p.checked(archive, ARCHIVE_SHA)
    source_files = 0
    source_bytes = 0
    for row in source["selected"]["files"]:
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])
        p.checked(image_path, row["image_sha256"])
        p.checked(mask_path, row["mask_sha256"])
        source_files += 2
        source_bytes += image_path.stat().st_size + mask_path.stat().st_size
    require(source_files == 304, "source file count")

    cache_receipt = json.loads(p.CACHE_RECEIPT.read_text())
    index_path = p.CACHE / "index.json"
    p.checked(index_path, cache_receipt["index_sha256"])
    p.checked(p.CACHE / "images.f32", cache_receipt["images_sha256"])
    p.checked(p.CACHE / "labels.u8", cache_receipt["labels_sha256"])
    cache_index = json.loads(index_path.read_text())
    p.write_new(cache_archive_path, cache_index)

    board_path = p.ROOT / "AFLOW_BRATS_LEADERBOARD.json"
    board = json.loads(board_path.read_text())
    rows = board["external_waw_gliomai152_direct_locator_rank_by_scan"]
    mapping = {
        "gcsam_inner_fit864_epoch184_identity": gcsam,
        "full972_random_init_voxel_kd_epoch168_flip8_tta": full_flip8,
        "full972_random_init_voxel_kd_epoch168_identity": full_identity,
    }
    require([row["rank"] for row in rows] == [1, 2, 3], "leaderboard ranks")
    require([row["scan_dice"] for row in rows] == sorted([row["scan_dice"] for row in rows], reverse=True), "leaderboard sort")
    for row in rows:
        summary = mapping[row["method"]]
        require(row["scan_dice"] == summary["locator_dsc_mean"], "leaderboard Dice")
        require(row["point_hits"] == summary["automatic_point_hits"], "leaderboard Point")
        require(row["report_sha256"] == RAW_SHA, "leaderboard report SHA")

    result = {
        "schema": "aflow.waw_gliomai152.result",
        "schema_version": 1,
        "status": "completed_and_cohort_closed",
        "cohort": {
            "cases": 152,
            "subjects": 152,
            "source": "WAW-GliomAI final Zenodo record22205631",
            "evidence": "public external-to-project; institution-disjoint/no-known-overlap; not hidden blind; no patient-ID bridge",
        },
        "absolute": {
            "full_identity": {
                "scan_subject_dice": full_identity["locator_dsc_mean"],
                "scan_subject_dice_95ci": raw["full972_kd168"]["identity_bootstrap"]["scan_dsc_mean_95ci"],
                "point_hits": full_identity["automatic_point_hits"],
            },
            "full_flip8": {
                "scan_subject_dice": full_flip8["locator_dsc_mean"],
                "scan_subject_dice_95ci": raw["full972_kd168"]["flip8_bootstrap"]["scan_dsc_mean_95ci"],
                "point_hits": full_flip8["automatic_point_hits"],
            },
            "gcsam_identity": {
                "scan_subject_dice": gcsam["locator_dsc_mean"],
                "scan_subject_dice_95ci": raw["gcsam184"]["identity_bootstrap"]["scan_dsc_mean_95ci"],
                "point_hits": gcsam["automatic_point_hits"],
            },
        },
        "comparisons": raw["comparisons"],
        "interpretation_gates": raw["interpretation_gates"],
        "cost": {
            "training": 0,
            "cache_seconds": cache_receipt["elapsed_seconds"],
            "formal_seconds": raw["elapsed_seconds"],
            "formal_forwards": raw["forwards"],
            "peak_gpu_memory_bytes": raw["peak_gpu_memory_bytes"],
            "source_archive_bytes": archive.stat().st_size,
            "selected_source_bytes_rehashed": source_bytes,
            "cache_bytes": (p.CACHE / "images.f32").stat().st_size + (p.CACHE / "labels.u8").stat().st_size,
        },
        "access": raw["access"],
        "training_source_caveat": raw["training_source_caveat"],
        "raw_report_sha256": RAW_SHA,
        "preaccess_v2_sha256": PREACCESS_V2_SHA,
        "cache_index_sha256": cache_receipt["index_sha256"],
        "source_receipt_sha256": p.SOURCE_RECEIPT_SHA,
        "result_must_not_trigger_tuning_or_repeat": True,
    }
    p.write_new(result_path, result)
    receipt = {
        "schema": "aflow.waw_gliomai152.verification",
        "schema_version": 1,
        "status": "passed",
        "cases": 152,
        "subjects": 152,
        "official_archive_md5_exact": True,
        "archive_sha256": ARCHIVE_SHA,
        "source_files_rehashed": source_files,
        "source_bytes_rehashed": source_bytes,
        "source_receipt_sha256": p.SOURCE_RECEIPT_SHA,
        "cache_receipt_sha256": CACHE_RECEIPT_SHA,
        "cache_index_sha256": cache_receipt["index_sha256"],
        "cache_index_archive_sha256": p.sha256(cache_archive_path),
        "cache_images_sha256": cache_receipt["images_sha256"],
        "cache_labels_sha256": cache_receipt["labels_sha256"],
        "full_checkpoint_sha256": p.FULL_SHA,
        "gcsam_checkpoint_sha256": p.GCSAM_SHA,
        "preaccess_v2_sha256": PREACCESS_V2_SHA,
        "started_sha256": STARTED_SHA,
        "raw_sha256": RAW_SHA,
        "result_sha256": p.sha256(result_path),
        "leaderboard_sha256": p.sha256(board_path),
        "recomputed": {"absolute_summaries": 3, "absolute_bootstraps": 3, "paired_bootstraps": 2, "leaderboard_rows": 3},
        "model_inference": False,
        "new_metrics": False,
        "verifier_sha256": p.sha256(Path(__file__)),
    }
    p.write_new(receipt_path, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
