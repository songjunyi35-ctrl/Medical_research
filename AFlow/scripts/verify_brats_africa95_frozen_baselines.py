#!/usr/bin/env python3
"""Verify the complete BraTS-Africa95 evidence chain without model inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import scripts.evaluate_brats_africa95_frozen_baselines as evaluate
from scripts import brats_africa95_protocol as p


RAW_SHA = "c9fb07be4412c160129037104936e4c34b6390fa733b998b8a8f35524ccd30b3"
STARTED_SHA = "7591379287b649b16b29de9a86d899338995e4f0d8d825b66942dae35ebb39d6"
CACHE_RECEIPT_SHA = "a9efa53d56b5087ab8d56ad4f69bd808b6e51d8cae70aefc372a59417e341497"
CACHE_INDEX_SHA = "1493bfde0e82309c3bfc73e2b2b9e1ab631921a6aeb988288f16ddbd9cba1c2e"
SOURCE_DOWNLOAD_SHA = "f5c129efcf073f24f39f94129e85f3e81c5614f67eb522ead651feac9a30324f"
PREACCESS_SHA = "59f2ab8c746fb57d4480ebe105975bf5e2e53a4111b51370b72d0d299b0e45e8"


def require(value: bool, message: str) -> None:
    if not value: raise ValueError(message)


def summary(rows: list[dict]) -> dict:
    require(len(rows) == len({x["case_id"] for x in rows}) == 95, "case count")
    require(len({x["subject_id"] for x in rows}) == 95, "subject count")
    mean = statistics.fmean(float(x["locator_dsc"]) for x in rows); hits = sum(bool(x["automatic_point_hits_gt"]) for x in rows)
    return {"locator_dsc_mean": mean, "locator_subject_dsc_mean": mean, "automatic_point_hits": hits, "automatic_point_hit_rate_positive_only": hits / 95}


def close(left, right, label):
    require(abs(float(left) - float(right)) <= 1e-12, label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--receipt", type=Path, required=True); parser.add_argument("--cache-index-archive", type=Path, required=True); args = parser.parse_args()
    if args.receipt.exists() or args.cache_index_archive.exists(): raise FileExistsError("verification output exists")
    raw_path = p.RECORD / "RAW_EXTERNAL95.json"; started_path = p.RECORD / "RAW_EXTERNAL95_STARTED.json"
    p.checked(raw_path, RAW_SHA); p.checked(started_path, STARTED_SHA); p.checked(p.SOURCE_DOWNLOAD, SOURCE_DOWNLOAD_SHA); p.checked(p.CACHE_RECEIPT, CACHE_RECEIPT_SHA); p.checked(p.PREACCESS, PREACCESS_SHA); p.checked(p.CACHE / "index.json", CACHE_INDEX_SHA)
    p.checked(p.FULL972, p.FULL972_SHA); p.checked(p.GCSAM184, p.GCSAM184_SHA)
    source_receipt = json.loads(p.SOURCE_DOWNLOAD.read_text())
    require(source_receipt["files_copy_exact"] == 190 and len(source_receipt["copies"]) == 2, "source receipt counts")
    raw_source_files = 0; raw_source_bytes = 0
    for copy in source_receipt["copies"]:
        require(len(copy["files"]) == 190 and copy["all_sizes_exact"], "source copy contract")
        for item in copy["files"]:
            path = Path(item["path"]); require(path.stat().st_size == item["bytes"] == item["expected_bytes"], "source file bytes"); p.checked(path, item["sha256"]); raw_source_files += 1; raw_source_bytes += item["bytes"]
    cache = json.loads((p.CACHE / "index.json").read_text()); p.checked(p.CACHE / "images.f32", cache["images_sha256"]); p.checked(p.CACHE / "labels.u8", cache["labels_sha256"])
    raw = json.loads(raw_path.read_text()); require(raw["forwards"] == 855 and raw["evidence_scope"].startswith("one-time public post-challenge"), "scope/forwards")
    require(raw["evaluator_sha256"] == p.sha(Path(evaluate.__file__)), "evaluator SHA")
    sections = [("full972_kd168", "identity"), ("full972_kd168", "flip8"), ("gcsam184", "identity")]
    for model, view in sections:
        recorded = raw[model][view]; recomputed = summary(recorded["per_case"])
        for key, value in recomputed.items(): close(recorded[key], value, model + view + key)
        expected_bootstrap = evaluate.absolute(recorded, p.BOOTSTRAP_SEED + ({("full972_kd168","identity"):0,("full972_kd168","flip8"):10,("gcsam184","identity"):20}[(model,view)]))
        require(expected_bootstrap == raw[model][view + "_bootstrap"], model + view + " bootstrap")
    full_identity = raw["full972_kd168"]["identity"]; full_flip = raw["full972_kd168"]["flip8"]; gcsam = raw["gcsam184"]["identity"]
    require(evaluate.compare(full_identity, full_flip, p.BOOTSTRAP_SEED + 100, "flip8_minus_identity") == raw["comparisons"]["full_flip8_minus_identity"], "flip comparison")
    require(evaluate.compare(full_identity, gcsam, p.BOOTSTRAP_SEED + 200, "gcsam_minus_full") == raw["comparisons"]["gcsam_identity_minus_full_identity"], "gcsam comparison")
    board = json.loads((p.ROOT / "AFLOW_BRATS_LEADERBOARD.json").read_text()); rows = board["external_brats_africa95_direct_locator_rank_by_scan"]
    mapping = {"gcsam_inner_fit864_epoch184_identity": gcsam, "full972_random_init_voxel_kd_epoch168_flip8_tta": full_flip, "full972_random_init_voxel_kd_epoch168_identity": full_identity}
    require([x["rank"] for x in rows] == [1,2,3], "leaderboard ranks")
    for row in rows:
        source = mapping[row["method"]]; close(row["scan_dice"], source["locator_dsc_mean"], "leader scan"); close(row["subject_dice"], source["locator_subject_dsc_mean"], "leader subject"); require(row["point_hits"] == source["automatic_point_hits"] and row["report_sha256"] == RAW_SHA, "leader point/hash")
    p.write_new(args.cache_index_archive, cache)
    receipt = {"schema": "aflow.brats_africa95.verification", "schema_version": 1, "status": "passed", "cases": 95, "subjects": 95, "raw_report_sha256": RAW_SHA, "started_sha256": STARTED_SHA, "source_download_sha256": SOURCE_DOWNLOAD_SHA, "raw_source_files_rehashed": raw_source_files, "raw_source_bytes_rehashed": raw_source_bytes, "raw_source_two_copies_exact": raw_source_files == 380 and raw_source_bytes == 2*p.BYTES_PER_COPY, "full972_checkpoint_sha256": p.FULL972_SHA, "gcsam184_checkpoint_sha256": p.GCSAM184_SHA, "evaluator_sha256": raw["evaluator_sha256"], "cache_receipt_sha256": CACHE_RECEIPT_SHA, "cache_index_sha256": CACHE_INDEX_SHA, "cache_index_archive_sha256": p.sha(args.cache_index_archive), "images_sha256": cache["images_sha256"], "labels_sha256": cache["labels_sha256"], "preaccess_sha256": PREACCESS_SHA, "recomputed": {"three_absolute_summaries": True, "three_absolute_bootstraps": True, "two_paired_bootstraps": True, "leaderboard_rows": 3}, "model_inference": False, "new_metric_access": False, "verifier_sha256": p.sha(Path(__file__))}
    p.write_new(args.receipt, receipt); print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__": main()
