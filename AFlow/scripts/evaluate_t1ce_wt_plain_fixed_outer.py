#!/usr/bin/env python3
"""Evaluate the inner-authorized fixed-epoch plain locator once on the outer selector."""
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
from scripts.train_t1ce_wt_capg3d import (  # noqa: E402
    BrainCropDataset,
    read_case_ids,
    sha256_file,
    validate,
    write_json_atomic,
)
from scripts.train_t1ce_wt_plain_fixed_epoch import (  # noqa: E402
    CHECKPOINT_SCHEMA,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_fixed_epoch_outer_evaluation"
OUTER_SELECTOR_SHA256 = "3710d63842acaccb1e4fe81c5fade578b6f1a306363d969ea2d799edd6c32b07"
INNER_DECISION_SHA256 = "6308c5e3807e513cc0eb3aebf0c4718e29c7e04e5748b314da6f3b28a7b227b4"
TRAIN_ROLE_SHA256 = "a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16"
AUTHORIZATION = "LOCKED_PLAIN_LONG_OUTER_SELECTOR_2026-08-18"


def verify_authorization(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite one-shot evaluation: {output}")
    lowered = str(args.case_ids.resolve()).lower()
    if any(token in lowered for token in ("confirmation", "sealed", "test")):
        raise ValueError("confirmation/test manifests are forbidden")
    case_sha = sha256_file(args.case_ids)
    if case_sha != args.expected_case_ids_sha256 or case_sha != OUTER_SELECTOR_SHA256:
        raise ValueError("outer-selector manifest SHA-256 mismatch")
    decision_sha = sha256_file(args.inner_decision)
    if decision_sha != args.expected_inner_decision_sha256 or decision_sha != INNER_DECISION_SHA256:
        raise ValueError("inner decision SHA-256 mismatch")
    if args.authorization != AUTHORIZATION:
        raise ValueError("explicit frozen outer-selector authorization is required")
    decision = json.loads(args.inner_decision.read_text())
    if decision.get("all_inner_gates_passed") is not True:
        raise ValueError("inner gates did not pass")
    if decision.get("outer_selector_authorized") is not True:
        raise ValueError("inner decision did not authorize the outer selector")
    if int(decision["post100"]["epoch"]) != 168:
        raise ValueError("inner decision did not select epoch 168")
    if any(decision.get(key) is not False for key in (
        "outer_selector_access", "confirmation_access", "test_access"
    )):
        raise ValueError("inner decision access flags were not sealed")

    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise ValueError("fixed-epoch checkpoint SHA-256 mismatch")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unexpected fixed-epoch checkpoint schema")
    if int(checkpoint.get("epoch", -1)) != 168:
        raise ValueError("fixed checkpoint is not epoch 168")
    if checkpoint.get("inner_decision_sha256") != decision_sha:
        raise ValueError("checkpoint was not authorized by the frozen inner decision")
    if checkpoint.get("data_role", {}).get("train", {}).get("sha256") != TRAIN_ROLE_SHA256:
        raise ValueError("checkpoint was not trained on the frozen 756-case role")
    if int(checkpoint.get("data_role", {}).get("cases", -1)) != 756:
        raise ValueError("checkpoint training case count mismatch")
    if any(checkpoint.get(key) is not False for key in (
        "validation_access", "outer_selector_access", "confirmation_access", "test_access"
    )):
        raise ValueError("checkpoint access flags are not sealed")
    return decision, checkpoint


def run(args: argparse.Namespace) -> dict[str, Any]:
    _, checkpoint = verify_authorization(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal one-shot outer evaluation requires CUDA")
    torch.cuda.set_device(device)

    case_ids = read_case_ids(args.case_ids)
    assert case_ids is not None
    dataset = BrainCropDataset(args.cache, expected_split="train", case_ids=case_ids)
    if len(dataset) != args.expected_cases:
        raise ValueError("unexpected outer-selector case count")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    architecture = checkpoint["architecture"]
    model = AutomaticPromptLocator(
        center_aware=False,
        init_filters=int(architecture["init_filters"]),
        dropout=float(architecture["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    summary = validate(model, loader, device)

    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "outer_selector",
        "fixed_epoch": 168,
        "fixed_logit_threshold": 0.0,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": args.expected_checkpoint_sha256,
        },
        "inner_decision": {
            "path": str(args.inner_decision.resolve()),
            "sha256": args.expected_inner_decision_sha256,
        },
        "case_manifest": {
            "path": str(args.case_ids.resolve()),
            "sha256": args.expected_case_ids_sha256,
            "cases": len(dataset),
        },
        "cache_index_sha256": sha256_file(args.cache / "index.json"),
        "summary": summary,
        "outer_selector_access": True,
        "confirmation_access": False,
        "test_access": False,
    }
    write_json_atomic(args.output, report)
    print(json.dumps({
        "role": report["role"],
        "epoch": report["fixed_epoch"],
        "scan_dsc_mean": summary["locator_dsc_mean"],
        "subject_dsc_mean": summary["locator_subject_dsc_mean"],
        "point_hit": summary["automatic_point_hit_rate"],
        "confirmation_access": False,
        "test_access": False,
    }, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--case-ids", type=Path, required=True)
    parser.add_argument("--expected-case-ids-sha256", required=True)
    parser.add_argument("--expected-cases", type=int, default=108)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--inner-decision", type=Path, required=True)
    parser.add_argument("--expected-inner-decision-sha256", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
