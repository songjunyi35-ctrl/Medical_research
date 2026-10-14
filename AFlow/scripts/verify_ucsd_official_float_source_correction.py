#!/usr/bin/env python3
"""Verify and independently summarize the UCSD75 official-float correction."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.evaluate_ucsd_ptgbm_external_confirmation as historical_eval  # noqa: E402
import scripts.evaluate_ucsd_official_float_source_correction as correction  # noqa: E402
from scripts.prepare_ucsd_ptgbm_external_confirmation import sha256_file, write_json_atomic  # noqa: E402


RAW_SHA256 = "a1155914b16b648f97f11a7d7a20cb27096d7eed9f9662db73f4905860da04d2"
STARTED_SHA256 = "553cdd5bb8395638698d14068ca4fda456a45f2c4fca5976661e2dc8e09125c1"
CACHE_INDEX_SHA256 = "2540d5015008e22b40f3f94cc3f33b0d96cc12ee70a14b12c6d15a48e5a0856f"
OLD_REPORT_SHA256 = correction.OLD_REPORT_SHA256


def _assert_close(left: float, right: float, label: str) -> None:
    if abs(float(left) - float(right)) > 1e-12:
        raise ValueError(f"{label} mismatch: {left} != {right}")


def _recompute_summary(summary: dict[str, Any], label: str) -> None:
    rows = summary["per_case"]
    if len(rows) != 75 or len({row["case_id"] for row in rows}) != 75:
        raise ValueError(f"{label} case count/uniqueness mismatch")
    by_subject: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_subject[row["subject_id"]].append(float(row["locator_dsc"]))
    if len(by_subject) != 53:
        raise ValueError(f"{label} subject count mismatch")
    _assert_close(statistics.fmean(float(row["locator_dsc"]) for row in rows), summary["locator_dsc_mean"], f"{label} scan")
    _assert_close(statistics.fmean(statistics.fmean(values) for values in by_subject.values()), summary["locator_subject_dsc_mean"], f"{label} subject")
    hits = sum(bool(row["automatic_point_hits_gt"]) for row in rows if row["target_wt_positive"])
    if hits != summary["automatic_point_hits"] or summary["wt_positive_cases"] != 75 or summary["wt_negative_cases"] != 0:
        raise ValueError(f"{label} point/target counts mismatch")


def validate(args: argparse.Namespace) -> dict[str, Any]:
    if sha256_file(args.raw.resolve()) != RAW_SHA256:
        raise ValueError("raw report SHA mismatch")
    if sha256_file(args.started.resolve()) != STARTED_SHA256:
        raise ValueError("started receipt SHA mismatch")
    if sha256_file(args.old_report.resolve()) != OLD_REPORT_SHA256:
        raise ValueError("old report SHA mismatch")
    cache_index_path = args.cache.resolve() / "index.json"
    if sha256_file(cache_index_path) != CACHE_INDEX_SHA256:
        raise ValueError("cache index SHA mismatch")
    cache_index = json.loads(cache_index_path.read_text())
    if sha256_file(args.cache.resolve() / "images.f32") != cache_index["images_sha256"]:
        raise ValueError("cache image SHA mismatch")
    if sha256_file(args.cache.resolve() / "labels.u8") != cache_index["labels_sha256"]:
        raise ValueError("cache label SHA mismatch")
    raw = json.loads(args.raw.read_text())
    old = json.loads(args.old_report.read_text())
    for key in ("full972_kd168", "gcsam184"):
        _recompute_summary(raw[key]["summary"], key)
    full_delta = correction.compare_source_version(raw["full972_kd168"]["summary"], old["primary"]["summary"], correction.BOOTSTRAP_SEED + 100)
    gcsam_delta = correction.compare_source_version(raw["gcsam184"]["summary"], old["secondary"]["summary"], correction.BOOTSTRAP_SEED + 200)
    if full_delta != raw["full972_kd168"]["source_delta"] or gcsam_delta != raw["gcsam184"]["source_delta"]:
        raise ValueError("source delta recomputation mismatch")
    model_comparison = historical_eval.compare(raw["full972_kd168"]["summary"], raw["gcsam184"]["summary"])
    if model_comparison != raw["official_float_model_comparison_full972_minus_gcsam"]:
        raise ValueError("model comparison recomputation mismatch")

    leaderboard = json.loads(args.leaderboard.read_text())
    rows = leaderboard["external_ucsd75_direct_locator_rank_by_scan"]
    if len(rows) != 4 or [row["rank"] for row in rows] != [1, 2, 3, 4]:
        raise ValueError("leaderboard external row count/ranks mismatch")
    expected = {
        "gcsam_inner_fit864_epoch184_official_float32": (raw["gcsam184"], RAW_SHA256),
        "gcsam_inner_fit864_epoch184_mirror_uint8": (old["secondary"], OLD_REPORT_SHA256),
        "full972_random_init_voxel_kd_epoch168_official_float32": (raw["full972_kd168"], RAW_SHA256),
        "full972_random_init_voxel_kd_epoch168_mirror_uint8": (old["primary"], OLD_REPORT_SHA256),
    }
    for row in rows:
        report_model, report_sha = expected[row["method"]]
        summary = report_model["summary"]
        _assert_close(row["scan_dice"], summary["locator_dsc_mean"], row["method"] + " scan")
        _assert_close(row["subject_dice"], summary["locator_subject_dsc_mean"], row["method"] + " subject")
        if row["point_hits"] != summary["automatic_point_hits"] or row["report_sha256"] != report_sha:
            raise ValueError(f"{row['method']} leaderboard point/report mismatch")
    if any(left["scan_dice"] < right["scan_dice"] for left, right in zip(rows, rows[1:])):
        raise ValueError("leaderboard not sorted")

    return {
        "schema": "aflow.ucsd75_official_float_source_correction.verification",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "raw_report_sha256": RAW_SHA256,
        "started_receipt_sha256": STARTED_SHA256,
        "cache_index_sha256": CACHE_INDEX_SHA256,
        "images_sha256": cache_index["images_sha256"],
        "labels_sha256": cache_index["labels_sha256"],
        "old_mirror_report_sha256": OLD_REPORT_SHA256,
        "recomputed": {"absolute_summaries": True, "source_paired_bootstrap": True, "model_paired_bootstrap": True, "leaderboard_four_rows": True},
        "cases": 75,
        "subjects": 53,
        "model_inference": False,
        "new_metric_access": False,
        "verifier": str(Path(__file__).resolve().relative_to(ROOT)),
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--started", type=Path, required=True)
    parser.add_argument("--old-report", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--leaderboard", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--cache-index-archive", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists() or args.cache_index_archive.exists():
        raise FileExistsError("refusing to overwrite verification artifacts")
    result = validate(args)
    write_json_atomic(args.cache_index_archive.resolve(), json.loads((args.cache.resolve() / "index.json").read_text()))
    result["cache_index_archive"] = str(args.cache_index_archive.resolve().relative_to(ROOT))
    result["cache_index_archive_sha256"] = sha256_file(args.cache_index_archive.resolve())
    write_json_atomic(args.receipt.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
