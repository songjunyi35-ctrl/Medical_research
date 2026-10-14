#!/usr/bin/env python3
"""One-shot fixed-validation comparison of Full972 raw168, LAWA6, and archived champion."""
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
from scripts.train_t1ce_wt_capg3d import (  # noqa: E402
    BrainCropDataset,
    read_case_ids,
    sha256_file,
    validate,
    write_json_atomic,
)
from scripts.train_t1ce_wt_voxel_kd_972_draw_budget import state_dict_sha256  # noqa: E402
from scripts.train_t1ce_wt_lawa6_stage_b_final import (  # noqa: E402
    ARM,
    CHECKPOINT_SCHEMA,
    SCHEMA as TRAIN_SCHEMA,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_lawa6_stage_b_validation"
PREACCESS_SCHEMA = SCHEMA + ".preaccess"
VALIDATION_MANIFEST_SHA256 = "f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb"
VALIDATION_CACHE_INDEX_SHA256 = "9339e4575561447dd5bd99bc29e909ae1e9f566721d351a244d087a40622f322"
ARCHIVED_VALIDATION_SHA256 = "30cd65c78ad1ed2a2b5748652f48ec05cde3b54468b44a66eba77d062d3ce129"
EXPECTED_ACCESS = {
    "train_only_inner": False,
    "fixed_validation108": False,
    "outer": False,
    "confirmation": False,
    "test270": False,
}


def load_train_report(path: Path, runner_sha: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if (
        value.get("schema") != TRAIN_SCHEMA
        or value.get("status") != "complete"
        or value.get("arm") != ARM
        or value.get("seed") != 20261063
        or value.get("smoke") is not False
        or value.get("all_gates_passed") is not True
        or value.get("runner_sha256") != runner_sha
        or value.get("access") != EXPECTED_ACCESS
        or len(value.get("history", [])) != 168
        or value.get("lawa", {}).get("captured_epochs") != [163, 164, 165, 166, 167, 168]
    ):
        raise ValueError("Stage B training report contract mismatch")
    sampling = value.get("sampling_contract", {})
    if (
        sampling.get("population_count"),
        sampling.get("requested_draws"),
        sampling.get("consumed_draws"),
        sampling.get("optimizer_steps"),
    ) != (972, 972, 968, 121):
        raise ValueError("Stage B Full972 compute contract mismatch")
    return value


def load_model(
    path: Path,
    *,
    kind: str,
    report: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    report_key = "raw" if kind == "raw_epoch168" else "lawa6"
    if sha256_file(path) != report["checkpoints"][report_key]["sha256"]:
        raise ValueError("Stage B checkpoint SHA mismatch")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if (
        value.get("schema") != CHECKPOINT_SCHEMA
        or value.get("arm") != ARM
        or value.get("seed") != 20261063
        or value.get("kind") != kind
        or value.get("epoch") != 168
        or value.get("access") != EXPECTED_ACCESS
        or value.get("lawa_contract", {}).get("captured_epochs") != [163, 164, 165, 166, 167, 168]
    ):
        raise ValueError("Stage B checkpoint metadata/access mismatch")
    if value.get("model_state_sha256") != state_dict_sha256(value["model_state_dict"]):
        raise ValueError("Stage B checkpoint state hash mismatch")
    model = AutomaticPromptLocator(center_aware=False, init_filters=16, dropout=0.1).to(device)
    model.load_state_dict(value["model_state_dict"], strict=True)
    return model.eval().requires_grad_(False), value


def promotion_gates(
    raw: dict[str, Any],
    lawa: dict[str, Any],
    archived: dict[str, Any],
    lawa_minus_raw: dict[str, Any],
    lawa_minus_archived: dict[str, Any],
    engineering_ok: bool,
) -> dict[str, bool]:
    dr = lawa_minus_raw["ema_minus_raw"]
    da = lawa_minus_archived["ema_minus_raw"]
    lawa_hits = int(round(float(lawa["automatic_point_hit_rate"]) * 108))
    raw_hits = int(round(float(raw["automatic_point_hit_rate"]) * 108))
    return {
        "lawa_beats_raw_scan": float(lawa["locator_dsc_mean"]) > float(raw["locator_dsc_mean"]),
        "lawa_beats_raw_subject": float(lawa["locator_subject_dsc_mean"]) > float(raw["locator_subject_dsc_mean"]),
        "lawa_beats_archived_scan": float(lawa["locator_dsc_mean"]) > float(archived["locator_dsc_mean"]),
        "lawa_beats_archived_subject": float(lawa["locator_subject_dsc_mean"]) > float(archived["locator_subject_dsc_mean"]),
        "lawa_point_at_least_raw": lawa_hits >= raw_hits,
        "lawa_point_at_least_103": lawa_hits >= 103,
        "lawa_vs_raw_scan_ci_safe": float(dr["scan_95ci"][0]) >= -0.003,
        "lawa_vs_raw_subject_ci_safe": float(dr["subject_95ci"][0]) >= -0.003,
        "lawa_vs_archived_scan_ci_safe": float(da["scan_95ci"][0]) >= -0.003,
        "lawa_vs_archived_subject_ci_safe": float(da["subject_95ci"][0]) >= -0.003,
        "engineering_gates": engineering_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--train-report", type=Path, required=True)
    parser.add_argument("--raw-checkpoint", type=Path, required=True)
    parser.add_argument("--lawa-checkpoint", type=Path, required=True)
    parser.add_argument("--archived-validation", type=Path, required=True)
    parser.add_argument("--preaccess-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    started = args.output.with_suffix(".started.json")
    if args.output.exists() or started.exists():
        raise FileExistsError("refusing repeated LAWA6 Stage B validation")
    evaluator_sha = sha256_file(Path(__file__).resolve())
    if sha256_file(args.validation_manifest) != VALIDATION_MANIFEST_SHA256:
        raise ValueError("fixed validation manifest SHA mismatch")
    if sha256_file(args.cache / "index.json") != VALIDATION_CACHE_INDEX_SHA256:
        raise ValueError("fixed validation cache SHA mismatch")
    if sha256_file(args.archived_validation) != ARCHIVED_VALIDATION_SHA256:
        raise ValueError("archived Full972 validation SHA mismatch")
    runner_sha = sha256_file(ROOT / "scripts/train_t1ce_wt_lawa6_stage_b_final.py")
    report = load_train_report(args.train_report, runner_sha)
    expected_checkpoints = {
        "raw": sha256_file(args.raw_checkpoint),
        "lawa6": sha256_file(args.lawa_checkpoint),
    }
    preaccess = json.loads(args.preaccess_receipt.read_text())
    if (
        preaccess.get("schema") != PREACCESS_SCHEMA
        or preaccess.get("status") != "frozen_before_fixed_validation"
        or preaccess.get("evaluator_sha256") != evaluator_sha
        or preaccess.get("runner_sha256") != runner_sha
        or preaccess.get("train_report_sha256") != sha256_file(args.train_report)
        or preaccess.get("checkpoint_sha256") != expected_checkpoints
        or preaccess.get("archived_validation_sha256") != ARCHIVED_VALIDATION_SHA256
        or preaccess.get("access") != EXPECTED_ACCESS
    ):
        raise ValueError("Stage B validation preaccess receipt mismatch")

    case_ids = read_case_ids(args.validation_manifest)
    data = BrainCropDataset(args.cache, expected_split="validation", case_ids=case_ids)
    if len(data) != 108 or len({case.rsplit("-", 1)[0] for case in data.case_ids}) != 50:
        raise ValueError("fixed validation cohort mismatch")
    write_json_atomic(started, {
        "schema": SCHEMA + ".started",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preaccess_receipt_sha256": sha256_file(args.preaccess_receipt),
        "train_only_inner_access": True,
        "fixed_validation108_access": True,
        "outer_access": False,
        "confirmation_access": False,
        "test270_access": False,
    })

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage B fixed validation requires CUDA")
    torch.cuda.set_device(device)
    loader = DataLoader(
        data,
        batch_size=8,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    summaries: dict[str, Any] = {}
    checkpoint_meta: dict[str, Any] = {}
    for name, path, kind in (
        ("raw", args.raw_checkpoint, "raw_epoch168"),
        ("lawa6", args.lawa_checkpoint, "lawa6_epochs163_168"),
    ):
        model, checkpoint = load_model(path, kind=kind, report=report, device=device)
        summaries[name] = validate(model, loader, device)
        checkpoint_meta[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "model_state_sha256": checkpoint["model_state_sha256"],
        }
        del model
        torch.cuda.empty_cache()
    archived_report = json.loads(args.archived_validation.read_text())
    archived = archived_report["summaries"]["full972_candidate"]
    raw_comparison = compare(summaries["raw"], summaries["lawa6"], seed=20261065, samples=10_000)
    archived_comparison = compare(archived, summaries["lawa6"], seed=20261065, samples=10_000)
    gates = promotion_gates(
        summaries["raw"],
        summaries["lawa6"],
        archived,
        raw_comparison,
        archived_comparison,
        bool(report["all_gates_passed"]),
    )
    result = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summaries": summaries,
        "archived_full972_seed32": archived,
        "lawa6_minus_raw": raw_comparison,
        "lawa6_minus_archived": archived_comparison,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "candidate_authorized_for_test": all(gates.values()),
        "inputs": {
            "train_report_sha256": sha256_file(args.train_report),
            "checkpoint_sha256": expected_checkpoints,
            "archived_validation_sha256": ARCHIVED_VALIDATION_SHA256,
            "preaccess_receipt_sha256": sha256_file(args.preaccess_receipt),
            "evaluator_sha256": evaluator_sha,
            "runner_sha256": runner_sha,
        },
        "access": {
            "train_only_inner": True,
            "fixed_validation108": True,
            "outer": False,
            "confirmation": False,
            "test270": False,
        },
    }
    write_json_atomic(args.output, result)
    print(json.dumps({
        "raw": {
            "scan": summaries["raw"]["locator_dsc_mean"],
            "subject": summaries["raw"]["locator_subject_dsc_mean"],
            "point": summaries["raw"]["automatic_point_hit_rate"],
        },
        "lawa6": {
            "scan": summaries["lawa6"]["locator_dsc_mean"],
            "subject": summaries["lawa6"]["locator_subject_dsc_mean"],
            "point": summaries["lawa6"]["automatic_point_hit_rate"],
        },
        "archived": {
            "scan": archived["locator_dsc_mean"],
            "subject": archived["locator_subject_dsc_mean"],
            "point": archived["automatic_point_hit_rate"],
        },
        "gates": gates,
        "candidate_authorized_for_test": result["candidate_authorized_for_test"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
