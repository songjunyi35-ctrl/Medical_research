#!/usr/bin/env python3
"""Reload and evaluate a frozen intensity-residual checkpoint without test access."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from scripts.train_t1ce_wt_capg3d import BrainCropDataset, read_case_ids, sha256_file, write_json_atomic
from scripts.train_t1ce_wt_intensity_residual import (
    ARMS,
    CHECKPOINT_SCHEMA,
    build_model,
    validate,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_intensity_residual_evaluation"
OUTER_SELECTOR_SHA256 = "3710d63842acaccb1e4fe81c5fade578b6f1a306363d969ea2d799edd6c32b07"
OUTER_AUTHORIZATION = "LOCKED_INTENSITY_OUTER_SELECTOR_2026-08-17"


def maximum_case_difference(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_cases = {case["case_id"]: case for case in left["per_case"]}
    right_cases = {case["case_id"]: case for case in right["per_case"]}
    if set(left_cases) != set(right_cases):
        raise RuntimeError("reload report case identities differ")
    maximum = 0.0
    for case_id in left_cases:
        for field in ("dsc", "base_dsc", "residual_abs_mean"):
            maximum = max(
                maximum,
                abs(float(left_cases[case_id][field]) - float(right_cases[case_id][field])),
            )
        if left_cases[case_id]["automatic_point"] != right_cases[case_id]["automatic_point"]:
            raise RuntimeError("reload automatic point differs")
        if left_cases[case_id]["automatic_point_hits_gt"] != right_cases[case_id]["automatic_point_hits_gt"]:
            raise RuntimeError("reload point-hit differs")
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--case-ids", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-case-ids-sha256", required=True)
    parser.add_argument("--role", choices=("inner_tune", "outer_selector"), required=True)
    parser.add_argument("--decision-report", type=Path)
    parser.add_argument("--authorization")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    lowered = str(args.case_ids.resolve()).lower()
    if "confirmation" in lowered or "sealed" in lowered or "test" in lowered:
        raise RuntimeError("confirmation/test manifests are forbidden")
    checkpoint = args.checkpoint.resolve()
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise RuntimeError("checkpoint SHA-256 mismatch")
    case_path = args.case_ids.resolve()
    case_sha = sha256_file(case_path)
    if case_sha != args.expected_case_ids_sha256:
        raise RuntimeError("case-role SHA-256 mismatch")
    if args.role == "outer_selector":
        if case_sha != OUTER_SELECTOR_SHA256:
            raise RuntimeError("outer selector role SHA-256 mismatch")
        if args.authorization != OUTER_AUTHORIZATION or args.decision_report is None:
            raise RuntimeError("locked inner decision and explicit selector authorization required")
        decision = json.loads(args.decision_report.read_text())
        if decision.get("all_inner_gates_passed") is not True:
            raise RuntimeError("inner decision did not authorize selector access")
        if decision.get("selected_arm") != "intensity_residual":
            raise RuntimeError("inner decision did not select intensity_residual")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA or payload.get("test_access") is not False:
        raise RuntimeError("unexpected or test-accessible checkpoint")
    arm = str(payload["arm"])
    if arm not in ARMS:
        raise RuntimeError("unexpected checkpoint arm")
    hyper = payload["hyperparameters"]
    namespace = argparse.Namespace(
        arm=arm,
        init_filters=int(hyper["init_filters"]),
        dropout=float(hyper["dropout"]),
        residual_channels=int(hyper["residual_channels"]),
        intensity_kernel=int(hyper["intensity_kernel"]),
    )
    model = build_model(namespace)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal reload evaluation requires CUDA")
    torch.cuda.set_device(device)
    model = model.to(device).eval()

    case_ids = read_case_ids(case_path)
    assert case_ids is not None
    dataset = BrainCropDataset(args.cache, expected_split="train", case_ids=case_ids)
    loader = DataLoader(
        dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    summary = validate(model, loader, device)
    embedded_difference = None
    if args.role == "inner_tune":
        embedded_difference = maximum_case_difference(payload["validation"], summary)
        if embedded_difference != 0.0:
            raise RuntimeError(f"inner-tune reload differs by {embedded_difference}")
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": args.role,
        "arm": arm,
        "fixed_logit_threshold": 0.0,
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha, "epoch": int(payload["epoch"])},
        "case_role": {"path": str(case_path), "sha256": case_sha, "cases": len(case_ids)},
        "cache_index_sha256": sha256_file(args.cache / "index.json"),
        "embedded_reload_max_abs_difference": embedded_difference,
        "summary": summary,
        "test_access": False,
        "confirmation_access": False,
    }
    write_json_atomic(output, report)
    print(json.dumps({
        "arm": arm, "role": args.role, "epoch": payload["epoch"],
        "scan_dsc_mean": summary["scan_dsc_mean"],
        "subject_dsc_mean": summary["subject_dsc_mean"],
        "point_hit": summary["automatic_point_hit_rate"],
        "reload_max_abs_difference": embedded_difference,
    }, indent=2))


if __name__ == "__main__":
    main()

