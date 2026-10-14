#!/usr/bin/env python3
"""Evaluate the validation-selected w=0.25 logit ensemble once on test270."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_t1ce_wt_plain_ema_validation import compare  # noqa: E402
from scripts.evaluate_t1ce_wt_plain_logit_ensemble_validation import (  # noqa: E402
    EMA168_SHA256,
    HISTORICAL_SHA256,
    _model,
    blend_logits,
    point_from_logits,
)
from scripts.evaluate_t1ce_wt_plain_locator_test270 import (  # noqa: E402
    TEST_CASES,
    TEST_MANIFEST_SHA256,
    TEST_SUBJECTS,
    TestBrainCropDataset,
)
from scripts.train_t1ce_wt_capg3d import (  # noqa: E402
    dice_per_case,
    read_case_ids,
    sha256_file,
    summarize_cases,
    write_json_atomic,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_logit_ensemble_test270"
AUTHORIZATION = "USER_CONTINUE_RESEARCH_LOGIT_ENSEMBLE_TEST270_2026-08-19"
VALIDATION_REPORT_SHA256 = "66681f5d811f25c6fc67eeafa12b7356b5e0962f98c20be66d5e60b53254460a"
EMA_WEIGHT = 0.25
HISTORICAL_SCAN = 0.7412039492417265
HISTORICAL_SUBJECT = 0.7231794858721793
HISTORICAL_POINT = 0.9296296296296296


@torch.inference_mode()
def run(args: argparse.Namespace):
    if args.authorization != AUTHORIZATION:
        raise RuntimeError("explicit fixed ensemble test270 authorization is required")
    if sha256_file(args.validation_report) != VALIDATION_REPORT_SHA256:
        raise ValueError("ensemble validation report SHA-256 mismatch")
    validation = json.loads(args.validation_report.read_text(encoding="utf-8"))
    if validation.get("selected_weight") != EMA_WEIGHT:
        raise ValueError("validation did not select the frozen EMA weight 0.25")
    if validation.get("ensemble_candidate_authorized_for_test") is not True:
        raise ValueError("validation gates did not authorize ensemble test")
    if sha256_file(args.historical_checkpoint) != HISTORICAL_SHA256:
        raise ValueError("historical checkpoint SHA-256 mismatch")
    if sha256_file(args.ema_checkpoint) != EMA168_SHA256:
        raise ValueError("EMA168 checkpoint SHA-256 mismatch")
    if sha256_file(args.test_manifest) != TEST_MANIFEST_SHA256:
        raise ValueError("test270 manifest SHA-256 mismatch")
    started = args.output.with_suffix(".started.json")
    if args.output.exists() or started.exists():
        raise FileExistsError("refusing repeated ensemble test270 evaluation")
    rows = [
        json.loads(line) for line in args.test_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    case_ids = [str(row["case_id"]) for row in rows]
    if len(case_ids) != TEST_CASES or len(set(case_ids)) != TEST_CASES:
        raise ValueError("unexpected test270 case count")
    if len({str(row["subject_id"]) for row in rows}) != TEST_SUBJECTS:
        raise ValueError("unexpected test270 subject count")
    data = TestBrainCropDataset(args.cache, case_ids)
    if data.index.get("manifest_sha256") != TEST_MANIFEST_SHA256:
        raise ValueError("test cache manifest binding mismatch")
    for filename, field in (("images.f32", "images_sha256"), ("labels.u8", "labels_sha256")):
        if sha256_file(args.cache / filename) != data.index[field]:
            raise ValueError(f"test cache {filename} SHA-256 mismatch")
    historical_checkpoint = torch.load(
        args.historical_checkpoint, map_location="cpu", weights_only=False
    )
    ema_checkpoint = torch.load(
        args.ema_checkpoint, map_location="cpu", weights_only=False
    )
    for checkpoint in (historical_checkpoint, ema_checkpoint):
        if any(
            checkpoint.get(key) is not False
            for key in ("validation_access", "outer_selector_access", "confirmation_access", "test_access")
        ):
            raise ValueError("checkpoint declares prior evaluation access")
    write_json_atomic(started, {
        "schema": SCHEMA + ".started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": AUTHORIZATION,
        "validation_report_sha256": VALIDATION_REPORT_SHA256,
        "historical_checkpoint_sha256": HISTORICAL_SHA256,
        "ema_checkpoint_sha256": EMA168_SHA256,
        "ema_weight": EMA_WEIGHT,
        "threshold": 0.0,
        "result_will_not_trigger_weight_or_threshold_tuning": True
    })
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ensemble test270 requires CUDA")
    torch.cuda.set_device(device)
    historical_model = _model(historical_checkpoint, device)
    ema_model = _model(ema_checkpoint, device)
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0
    )
    cases: dict[str, list[dict[str, Any]]] = {"historical": [], "ensemble": []}
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            historical_logits = historical_model(image).segmentation_logits.float()
            ema_logits = ema_model(image).segmentation_logits.float()
        values = {
            "historical": historical_logits,
            "ensemble": blend_logits(historical_logits, ema_logits, EMA_WEIGHT)
        }
        for name, logits in values.items():
            dsc = dice_per_case(logits, label)
            points = point_from_logits(logits)
            for index, case_id in enumerate(batch["case_id"]):
                z, y, x = (int(value) for value in points[index].tolist())
                cases[name].append({
                    "case_id": str(case_id),
                    "subject_id": str(batch["subject_id"][index]),
                    "locator_dsc": float(dsc[index]),
                    "automatic_point": [z, y, x],
                    "automatic_point_hits_gt": bool(label[index, 0, z, y, x] > 0.5)
                })
    historical = summarize_cases(cases["historical"])
    ensemble = summarize_cases(cases["ensemble"])
    for actual, expected, name in (
        (historical["locator_dsc_mean"], HISTORICAL_SCAN, "scan"),
        (historical["locator_subject_dsc_mean"], HISTORICAL_SUBJECT, "subject"),
        (historical["automatic_point_hit_rate"], HISTORICAL_POINT, "point")
    ):
        if abs(float(actual) - expected) > 1e-9:
            raise RuntimeError(f"historical test reproduction changed: {name}")
    paired = compare(historical, ensemble, seed=20260842)
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "validation-selected fixed ensemble on reused internal test270",
        "not_pristine_blind_test": True,
        "authorization": AUTHORIZATION,
        "method": {
            "formula": "0.75 * historical_epoch168_logit + 0.25 * EMA168_logit",
            "ema_weight": EMA_WEIGHT,
            "threshold": "raw blended logit > 0",
            "point": "argmax blended logit",
            "tta": False,
            "postprocessing": False
        },
        "validation_selection": {
            "report": str(args.validation_report.resolve()),
            "sha256": VALIDATION_REPORT_SHA256
        },
        "checkpoints": {
            "historical": {
                "path": str(args.historical_checkpoint.resolve()),
                "sha256": HISTORICAL_SHA256
            },
            "ema168": {
                "path": str(args.ema_checkpoint.resolve()),
                "sha256": EMA168_SHA256
            }
        },
        "data": {
            "manifest": str(args.test_manifest.resolve()),
            "manifest_sha256": TEST_MANIFEST_SHA256,
            "cache": str(args.cache.resolve()),
            "cache_index_sha256": sha256_file(args.cache / "index.json"),
            "cases": TEST_CASES,
            "subjects": TEST_SUBJECTS
        },
        "historical": historical,
        "ensemble": ensemble,
        "paired": paired,
        "new_internal_test_point_estimate_high": (
            paired["ema_minus_raw"]["scan"] > 0.0
            and paired["ema_minus_raw"]["subject"] > 0.0
        ),
        "result_will_not_trigger_weight_or_threshold_tuning": True,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": True
    }
    write_json_atomic(args.output, report)
    print(json.dumps({
        "historical": paired["raw"],
        "ensemble": paired["ema"],
        "ensemble_minus_historical": paired["ema_minus_raw"],
        "new_internal_test_point_estimate_high": report[
            "new_internal_test_point_estimate_high"
        ]
    }, indent=2, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--historical-checkpoint", type=Path, required=True)
    parser.add_argument("--ema-checkpoint", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
