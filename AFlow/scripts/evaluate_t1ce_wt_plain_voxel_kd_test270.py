#!/usr/bin/env python3
"""Evaluate the validation-selected single KD epoch200 checkpoint on test270."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator  # noqa: E402
from scripts.evaluate_t1ce_wt_plain_ema_validation import compare  # noqa: E402
from scripts.evaluate_t1ce_wt_plain_fixed_test270 import intervals  # noqa: E402
from scripts.evaluate_t1ce_wt_plain_locator_test270 import (  # noqa: E402
    TEST_CASES,
    TEST_MANIFEST_SHA256,
    TEST_SUBJECTS,
    TestBrainCropDataset,
)
from scripts.train_t1ce_wt_capg3d import sha256_file, validate, write_json_atomic  # noqa: E402
from scripts.train_t1ce_wt_plain_voxel_kd_continuation import CHECKPOINT_SCHEMA  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_voxel_kd_test270"
AUTHORIZATION = "USER_CONTINUE_FRONTIER_RESEARCH_VOXEL_KD_TEST270_2026-08-19"
CHECKPOINT_SHA256 = "56f54898ded11a0208e75f8000d57d4d053e80d6dedd760402730ad345c111b7"
VALIDATION_SHA256 = "db7c3bc6b5ff9e1076827d25c9f4de0dea88781bfc5de45da0fb2f79a09bfb57"
HISTORICAL_REPORT_SHA256 = "91433c4b6b88b233d827e41b0d038f459c25ef41dec224c84e154e90f370bb8c"
ENSEMBLE_REPORT_SHA256 = "b49cf72a7d749afd28bc3009d0f7ad27b63ce5642424cc8eeb183e4eec7dbc7c"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--historical-report", type=Path, required=True)
    parser.add_argument("--ensemble-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.authorization != AUTHORIZATION:
        raise RuntimeError("explicit selected KD test authorization is required")
    for path, expected, name in (
        (args.checkpoint, CHECKPOINT_SHA256, "checkpoint"),
        (args.validation_report, VALIDATION_SHA256, "validation"),
        (args.historical_report, HISTORICAL_REPORT_SHA256, "historical report"),
        (args.ensemble_report, ENSEMBLE_REPORT_SHA256, "ensemble report"),
        (args.test_manifest, TEST_MANIFEST_SHA256, "test manifest")
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"{name} SHA-256 mismatch")
    validation_report = json.loads(args.validation_report.read_text(encoding="utf-8"))
    if validation_report.get("selected_kd_epoch") != 200:
        raise ValueError("validation did not select global epoch200")
    if validation_report.get("kd_candidate_authorized_for_test") is not True:
        raise ValueError("KD validation gates did not authorize test")
    started = args.output.with_suffix(".started.json")
    if args.output.exists() or started.exists():
        raise FileExistsError("refusing repeated KD test270")
    rows = [
        json.loads(line) for line in args.test_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    case_ids = [str(row["case_id"]) for row in rows]
    if len(case_ids) != TEST_CASES or len(set(case_ids)) != TEST_CASES:
        raise ValueError("unexpected test case count")
    if len({str(row["subject_id"]) for row in rows}) != TEST_SUBJECTS:
        raise ValueError("unexpected test subject count")
    data = TestBrainCropDataset(args.cache, case_ids)
    if data.index.get("manifest_sha256") != TEST_MANIFEST_SHA256:
        raise ValueError("test cache binding mismatch")
    for filename, field in (("images.f32", "images_sha256"), ("labels.u8", "labels_sha256")):
        if sha256_file(args.cache / filename) != data.index[field]:
            raise ValueError(f"test cache {filename} SHA-256 mismatch")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unexpected KD checkpoint schema")
    if checkpoint.get("arm") != "kd" or checkpoint.get("global_epoch") != 200:
        raise ValueError("checkpoint is not selected KD global epoch200")
    if checkpoint.get("distillation", {}).get("temperature") != 2.0:
        raise ValueError("KD temperature mismatch")
    if checkpoint.get("distillation", {}).get("kd_weight") != 0.3:
        raise ValueError("KD weight mismatch")
    if any(
        checkpoint.get(key) is not False
        for key in ("validation_access", "outer_selector_access", "confirmation_access", "test_access")
    ):
        raise ValueError("KD checkpoint declares prior evaluation access")
    historical_report = json.loads(args.historical_report.read_text(encoding="utf-8"))
    ensemble_report = json.loads(args.ensemble_report.read_text(encoding="utf-8"))
    historical = historical_report["summary"]
    ensemble = ensemble_report["ensemble"]
    write_json_atomic(started, {
        "schema": SCHEMA + ".started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": AUTHORIZATION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "validation_report_sha256": VALIDATION_SHA256,
        "global_epoch": 200,
        "temperature": 2.0,
        "kd_weight": 0.3,
        "threshold": 0.0,
        "result_will_not_trigger_temperature_weight_or_epoch_tuning": True
    })
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("KD test270 requires CUDA")
    torch.cuda.set_device(device)
    architecture = checkpoint["architecture"]
    model = AutomaticPromptLocator(
        center_aware=False,
        init_filters=int(architecture["init_filters"]),
        dropout=float(architecture["dropout"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0
    )
    kd_summary = validate(model, loader, device)
    versus_historical = compare(historical, kd_summary, seed=20260855)
    versus_ensemble = compare(ensemble, kd_summary, seed=20260856)
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "validation-selected 2025/2026 voxel-KD single model on reused test270",
        "not_pristine_blind_test": True,
        "authorization": AUTHORIZATION,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": CHECKPOINT_SHA256,
            "global_epoch": 200
        },
        "distillation": checkpoint["distillation"],
        "validation_selection": {
            "report": str(args.validation_report.resolve()),
            "sha256": VALIDATION_SHA256
        },
        "data": {
            "manifest": str(args.test_manifest.resolve()),
            "manifest_sha256": TEST_MANIFEST_SHA256,
            "cache": str(args.cache.resolve()),
            "cache_index_sha256": sha256_file(args.cache / "index.json"),
            "cases": TEST_CASES,
            "subjects": TEST_SUBJECTS
        },
        "kd": kd_summary,
        "kd_bootstrap": intervals(kd_summary),
        "historical": historical,
        "ensemble": ensemble,
        "kd_minus_historical": versus_historical,
        "kd_minus_ensemble": versus_ensemble,
        "new_single_model_internal_test_high": (
            versus_historical["ema_minus_raw"]["scan"] > 0.0
            and versus_historical["ema_minus_raw"]["subject"] > 0.0
        ),
        "result_will_not_trigger_temperature_weight_or_epoch_tuning": True,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": True
    }
    write_json_atomic(args.output, report)
    print(json.dumps({
        "kd": {
            "scan": kd_summary["locator_dsc_mean"],
            "subject": kd_summary["locator_subject_dsc_mean"],
            "point_hit": kd_summary["automatic_point_hit_rate"]
        },
        "kd_minus_historical": versus_historical["ema_minus_raw"],
        "kd_minus_ensemble": versus_ensemble["ema_minus_raw"],
        "new_single_model_internal_test_high": report[
            "new_single_model_internal_test_high"
        ]
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
