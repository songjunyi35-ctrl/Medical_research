#!/usr/bin/env python3
"""Evaluate the fixed EMA168 single locator once on reused test270."""
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
from scripts.train_t1ce_wt_plain_ema_full_cohort import CHECKPOINT_SCHEMA  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_ema168_test270"
AUTHORIZATION = "USER_AUTHORIZED_SINGLE_EMA168_TEST270_2026-08-19"
EMA168_SHA256 = "4ea97dacb9a538993e1c16593bc56b03ce6dc3df2f457c1f0426439c6252197e"
HISTORICAL_REPORT_SHA256 = "91433c4b6b88b233d827e41b0d038f459c25ef41dec224c84e154e90f370bb8c"


def run(args: argparse.Namespace):
    if args.authorization != AUTHORIZATION:
        raise RuntimeError("explicit single EMA168 test270 authorization is required")
    if sha256_file(args.checkpoint) != EMA168_SHA256:
        raise ValueError("EMA168 checkpoint SHA-256 mismatch")
    if sha256_file(args.historical_report) != HISTORICAL_REPORT_SHA256:
        raise ValueError("historical epoch168 test report SHA-256 mismatch")
    if sha256_file(args.test_manifest) != TEST_MANIFEST_SHA256:
        raise ValueError("test270 manifest SHA-256 mismatch")
    started = args.output.with_suffix(".started.json")
    if args.output.exists() or started.exists():
        raise FileExistsError("refusing repeated EMA168 test270 evaluation")
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
        raise ValueError("unexpected EMA checkpoint schema")
    if checkpoint.get("kind") != "ema" or checkpoint.get("epoch") != 168:
        raise ValueError("checkpoint is not fixed EMA168")
    if checkpoint.get("resolution") != 128:
        raise ValueError("checkpoint is not true-128^3")
    if checkpoint.get("ema", {}).get("decay") != 0.999:
        raise ValueError("EMA decay differs from frozen 0.999")
    if any(
        checkpoint.get(key) is not False
        for key in ("validation_access", "outer_selector_access", "confirmation_access", "test_access")
    ):
        raise ValueError("EMA checkpoint declares prior evaluation access")
    historical_report = json.loads(args.historical_report.read_text(encoding="utf-8"))
    if historical_report.get("test_access") is not True:
        raise ValueError("historical baseline is not a test report")
    write_json_atomic(started, {
        "schema": SCHEMA + ".started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": AUTHORIZATION,
        "checkpoint_sha256": EMA168_SHA256,
        "epoch": 168,
        "ema_decay": 0.999,
        "threshold": 0.0,
        "tta": False,
        "postprocessing": False,
        "result_will_not_trigger_ema_or_threshold_tuning": True
    })
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("EMA168 test270 requires CUDA")
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
    ema_summary = validate(model, loader, device)
    historical_summary = historical_report["summary"]
    paired = compare(historical_summary, ema_summary, seed=20260843)
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "fixed EMA168 single-model evaluation on reused internal test270",
        "not_pristine_blind_test": True,
        "authorization": AUTHORIZATION,
        "protocol": {
            "model": "single true-128^3 EMA168 locator",
            "ema_decay": 0.999,
            "ema_start_after_epoch": 100,
            "threshold": "raw logit > 0",
            "point": "argmax raw logit",
            "tta": False,
            "postprocessing": False,
            "sam": False
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": EMA168_SHA256
        },
        "data": {
            "manifest": str(args.test_manifest.resolve()),
            "manifest_sha256": TEST_MANIFEST_SHA256,
            "cache": str(args.cache.resolve()),
            "cache_index_sha256": sha256_file(args.cache / "index.json"),
            "cases": TEST_CASES,
            "subjects": TEST_SUBJECTS
        },
        "historical": {
            "report": str(args.historical_report.resolve()),
            "report_sha256": HISTORICAL_REPORT_SHA256,
            "summary": historical_summary
        },
        "ema": ema_summary,
        "ema_bootstrap": intervals(ema_summary),
        "paired": paired,
        "new_internal_test_point_estimate_high": (
            paired["ema_minus_raw"]["scan"] > 0.0
            and paired["ema_minus_raw"]["subject"] > 0.0
        ),
        "result_will_not_trigger_ema_or_threshold_tuning": True,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": True
    }
    write_json_atomic(args.output, report)
    print(json.dumps({
        "historical": paired["raw"],
        "ema168": paired["ema"],
        "ema_minus_historical": paired["ema_minus_raw"],
        "new_internal_test_point_estimate_high": report[
            "new_internal_test_point_estimate_high"
        ]
    }, indent=2, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--historical-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
