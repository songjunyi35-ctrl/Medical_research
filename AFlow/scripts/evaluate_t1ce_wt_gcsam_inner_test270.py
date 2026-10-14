#!/usr/bin/env python3
"""One-shot user-authorized exploratory test270 for frozen GCSAM inner epoch184."""
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
from scripts.train_t1ce_wt_voxel_kd_972_draw_budget import state_dict_sha256  # noqa: E402
import scripts.train_t1ce_wt_gcsam_inner as train  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_gcsam_inner_failed_gate_test270"
OVERRIDE_SCHEMA = "aflow.brats.t1ce_wt_gcsam_inner_user_test_override"
CHECKPOINT_SHA256 = "acf2fce112370e3cdb58fe295642680d5db1db3ceb619defdc47824ef4016d0d"
INNER_DECISION_SHA256 = "78851d7b15bc4414e404f968dce816241c3d7656fc27c80250059e1a86341577"
IMPLEMENTATION_RECEIPT_SHA256 = "74536ec022d2094732be68841b454ef0a32f57434c0c743fa9e221220cf175fb"
CURRENT_BEST_REPORT_SHA256 = "9827a75b88dc9239902c7e473a7d75e2d6058f4f515bfd2b4d93003ba7bd34cd"
TEST_CACHE_INDEX_SHA256 = "41bd6cf44553b8d156de49f2c57bb424cab60cfa26c06881bdfb96ff91de39d3"


def require_access_seal(value: dict[str, Any]) -> None:
    expected = {
        "train_only_inner": True,
        "fixed_validation108": False,
        "outer": False,
        "confirmation": False,
        "test270": False,
    }
    if value != expected:
        raise ValueError("checkpoint access seal mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inner-decision", type=Path, required=True)
    parser.add_argument("--implementation-receipt", type=Path, required=True)
    parser.add_argument("--current-best-report", type=Path, required=True)
    parser.add_argument("--override-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    started = args.output.with_suffix(".started.json")
    if args.output.exists() or started.exists():
        raise FileExistsError("refusing repeated GCSAM exploratory test270")
    evaluator_sha = sha256_file(Path(__file__).resolve())
    expected = (
        (args.checkpoint, CHECKPOINT_SHA256),
        (args.inner_decision, INNER_DECISION_SHA256),
        (args.implementation_receipt, IMPLEMENTATION_RECEIPT_SHA256),
        (args.current_best_report, CURRENT_BEST_REPORT_SHA256),
        (args.test_manifest, TEST_MANIFEST_SHA256),
        (args.cache / "index.json", TEST_CACHE_INDEX_SHA256),
    )
    for path, digest in expected:
        if sha256_file(path) != digest:
            raise ValueError(f"frozen SHA mismatch: {path}")

    override = json.loads(args.override_receipt.read_text())
    if (
        override.get("schema") != OVERRIDE_SCHEMA
        or override.get("status") != "frozen_before_test270"
        or override.get("test_access") is not False
    ):
        raise ValueError("invalid GCSAM user override receipt")
    bindings = {
        "evaluator_sha256": evaluator_sha,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "inner_decision_sha256": INNER_DECISION_SHA256,
        "implementation_receipt_sha256": IMPLEMENTATION_RECEIPT_SHA256,
        "current_best_report_sha256": CURRENT_BEST_REPORT_SHA256,
        "test_manifest_sha256": TEST_MANIFEST_SHA256,
        "test_cache_index_sha256": TEST_CACHE_INDEX_SHA256,
    }
    for key, value in bindings.items():
        if override.get(key) != value:
            raise ValueError(f"override receipt binding mismatch: {key}")
    if override.get("user_authorization_quote") != "可以去test上评测了":
        raise ValueError("override lacks exact user authorization")

    decision = json.loads(args.inner_decision.read_text())
    if (
        decision.get("selected_epoch") != 184
        or decision.get("final_retraining_authorized") is not False
        or decision.get("all_gates_passed") is not False
    ):
        raise ValueError("expected failed-gate inner-selected epoch184")
    if decision.get("access", {}).get("test270") is not False:
        raise ValueError("inner decision declares prior test access")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema") != train.CHECKPOINT_SCHEMA
        or checkpoint.get("arm") != "gcsam_candidate"
        or checkpoint.get("epoch") != 184
        or checkpoint.get("window") != "post100"
    ):
        raise ValueError("GCSAM checkpoint contract mismatch")
    if checkpoint.get("model_state_sha256") != state_dict_sha256(checkpoint["model_state_dict"]):
        raise ValueError("GCSAM checkpoint model-state mismatch")
    require_access_seal(checkpoint.get("access", {}))

    current_best_report = json.loads(args.current_best_report.read_text())
    if (
        current_best_report.get("test_access") is not True
        or current_best_report.get("failed_gate_exploratory") is not True
    ):
        raise ValueError("current-best report evidence scope mismatch")
    current_best = current_best_report["candidate"]

    write_json_atomic(started, {
        "schema": SCHEMA + ".started",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "override_receipt_sha256": sha256_file(args.override_receipt),
        "evaluator_sha256": evaluator_sha,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "training_role": "fit864 after holding108 train-only inner cases out",
        "selection_role": "train-only inner108",
        "selected_epoch": 184,
        "threshold": 0.0,
        "tta": False,
        "sam": False,
        "postprocessing": False,
        "failed_gate_exploratory": True,
        "not_full972_final": True,
        "not_pristine_blind_test": True,
        "test_access": True,
    })

    rows = [
        json.loads(line)
        for line in args.test_manifest.read_text().splitlines()
        if line.strip()
    ]
    case_ids = [str(row["case_id"]) for row in rows]
    if len(case_ids) != TEST_CASES or len(set(case_ids)) != TEST_CASES:
        raise ValueError("unexpected test case count")
    if len({str(row["subject_id"]) for row in rows}) != TEST_SUBJECTS:
        raise ValueError("unexpected test subject count")
    data = TestBrainCropDataset(args.cache, case_ids)
    if data.index.get("manifest_sha256") != TEST_MANIFEST_SHA256:
        raise ValueError("test cache manifest binding mismatch")
    for filename, field in (("images.f32", "images_sha256"), ("labels.u8", "labels_sha256")):
        if sha256_file(args.cache / filename) != data.index[field]:
            raise ValueError(f"test cache tensor mismatch: {filename}")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GCSAM test270 requires CUDA")
    torch.cuda.set_device(device)
    model = AutomaticPromptLocator(center_aware=False, init_filters=16, dropout=0.1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    loader = DataLoader(
        data,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    with torch.no_grad():
        summary = validate(model, loader, device)
    versus_current_best = compare(current_best, summary, seed=20261006, samples=10_000)
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "user-authorized failed-gate exploratory reused internal test270 for fit864 train-inner-selected GCSAM; not Full972 final",
        "not_pristine_blind_test": True,
        "failed_gate_exploratory": True,
        "not_full972_final": True,
        "result_must_not_trigger_tuning": True,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": CHECKPOINT_SHA256,
            "model_state_sha256": checkpoint["model_state_sha256"],
            "epoch": 184,
            "fit_cases": 864,
            "train_inner_selection_cases": 108,
        },
        "data": {
            "manifest": str(args.test_manifest.resolve()),
            "manifest_sha256": TEST_MANIFEST_SHA256,
            "cache_index_sha256": TEST_CACHE_INDEX_SHA256,
            "cases": TEST_CASES,
            "subjects": TEST_SUBJECTS,
        },
        "gcsam": summary,
        "gcsam_bootstrap": intervals(summary),
        "current_internal_best": current_best,
        "gcsam_minus_current_internal_best": versus_current_best,
        "protocol": {
            "raw_logit_threshold": 0.0,
            "point": "argmax raw logit",
            "locator_forwards": 1,
            "tta": False,
            "sam": False,
            "postprocessing": False,
        },
        "authorization": {
            "quote": override["user_authorization_quote"],
            "receipt": str(args.override_receipt.resolve()),
            "receipt_sha256": sha256_file(args.override_receipt),
        },
        "validation_access": False,
        "outer_access": False,
        "confirmation_access": False,
        "test_access": True,
    }
    write_json_atomic(args.output, report)
    print(json.dumps({
        "gcsam": {
            "scan": summary["locator_dsc_mean"],
            "subject": summary["locator_subject_dsc_mean"],
            "point": summary["automatic_point_hit_rate"],
        },
        "gcsam_minus_current_internal_best": versus_current_best["ema_minus_raw"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
