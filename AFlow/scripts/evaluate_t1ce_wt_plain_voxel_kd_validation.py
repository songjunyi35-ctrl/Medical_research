#!/usr/bin/env python3
"""Evaluate matched control/KD continuation checkpoints on validation108."""
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
from scripts.train_t1ce_wt_capg3d import (  # noqa: E402
    BrainCropDataset,
    read_case_ids,
    sha256_file,
    validate,
    write_json_atomic,
)
from scripts.train_t1ce_wt_plain_voxel_kd_continuation import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    SAVE_GLOBAL_EPOCHS,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_voxel_kd_validation"
VALIDATION_MANIFEST_SHA256 = "f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb"
HISTORICAL_COMPARISON_SHA256 = "3404d598ade63ec6d2b7ccd44f67e284ca7f2c7c7a904546fdfcd1ed8fb95f12"
ENSEMBLE_VALIDATION_SHA256 = "66681f5d811f25c6fc67eeafa12b7356b5e0962f98c20be66d5e60b53254460a"


def load_model(path: Path, arm: str, epoch: int, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"unexpected continuation checkpoint schema: {path}")
    if checkpoint.get("arm") != arm or checkpoint.get("global_epoch") != epoch:
        raise ValueError(f"unexpected arm/epoch checkpoint: {path}")
    if any(
        checkpoint.get(key) is not False
        for key in ("validation_access", "outer_selector_access", "confirmation_access", "test_access")
    ):
        raise ValueError("continuation checkpoint declares prior evaluation access")
    architecture = checkpoint["architecture"]
    model = AutomaticPromptLocator(
        center_aware=False,
        init_filters=int(architecture["init_filters"]),
        dropout=float(architecture["dropout"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--kd-dir", type=Path, required=True)
    parser.add_argument("--historical-comparison", type=Path, required=True)
    parser.add_argument("--ensemble-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if sha256_file(args.validation_manifest) != VALIDATION_MANIFEST_SHA256:
        raise ValueError("validation manifest SHA-256 mismatch")
    if sha256_file(args.historical_comparison) != HISTORICAL_COMPARISON_SHA256:
        raise ValueError("historical comparison SHA-256 mismatch")
    if sha256_file(args.ensemble_validation) != ENSEMBLE_VALIDATION_SHA256:
        raise ValueError("ensemble validation SHA-256 mismatch")
    if args.output.exists() or args.output.with_suffix(".started.json").exists():
        raise FileExistsError("refusing repeated KD validation")
    historical_report = json.loads(
        args.historical_comparison.read_text(encoding="utf-8")
    )
    ensemble_report = json.loads(
        args.ensemble_validation.read_text(encoding="utf-8")
    )
    historical = historical_report["historical"]["summary"]
    ensemble = ensemble_report["summaries"]["0.25"]
    case_ids = read_case_ids(args.validation_manifest)
    assert case_ids is not None
    data = BrainCropDataset(
        args.cache, expected_split="validation", case_ids=case_ids
    )
    if len(data) != 108 or len({case.rsplit("-", 1)[0] for case in data.case_ids}) != 50:
        raise ValueError("unexpected validation case/subject count")
    write_json_atomic(args.output.with_suffix(".started.json"), {
        "schema": SCHEMA + ".started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "global_epochs": list(SAVE_GLOBAL_EPOCHS),
        "selection": "maximum KD scan Dice, subject tie-break",
        "test_access": False
    })
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("KD validation requires CUDA")
    torch.cuda.set_device(device)
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0
    )
    summaries = {"control": {}, "kd": {}}
    checkpoints = {"control": {}, "kd": {}}
    for arm, directory in (("control", args.control_dir), ("kd", args.kd_dir)):
        for epoch in SAVE_GLOBAL_EPOCHS:
            path = directory / f"plain_128_{arm}_global_epoch{epoch}.pth"
            model, checkpoint = load_model(path, arm, epoch, device)
            summaries[arm][str(epoch)] = validate(model, loader, device)
            checkpoints[arm][str(epoch)] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "distillation": checkpoint["distillation"]
            }
            del model
            torch.cuda.empty_cache()
    matched = {}
    versus_historical = {}
    for index, epoch in enumerate(SAVE_GLOBAL_EPOCHS):
        key = str(epoch)
        matched[key] = compare(
            summaries["control"][key], summaries["kd"][key], seed=20260847 + index
        )
        versus_historical[key] = compare(
            historical, summaries["kd"][key], seed=20260851 + index
        )
    selected_epoch = max(
        SAVE_GLOBAL_EPOCHS,
        key=lambda epoch: (
            summaries["kd"][str(epoch)]["locator_dsc_mean"],
            summaries["kd"][str(epoch)]["locator_subject_dsc_mean"]
        )
    )
    selected_key = str(selected_epoch)
    selected_summary = summaries["kd"][selected_key]
    historical_scan = float(historical["locator_dsc_mean"])
    historical_subject = float(historical["locator_subject_dsc_mean"])
    ensemble_scan = float(ensemble["locator_dsc_mean"])
    ensemble_subject = float(ensemble["locator_subject_dsc_mean"])
    ensemble_point = float(ensemble["automatic_point_hit_rate"])
    gates = {
        "beats_matched_control_scan": matched[selected_key]["ema_minus_raw"]["scan"] > 0.0,
        "beats_matched_control_subject": matched[selected_key]["ema_minus_raw"]["subject"] > 0.0,
        "beats_historical_scan": float(selected_summary["locator_dsc_mean"]) > historical_scan,
        "beats_historical_subject": float(selected_summary["locator_subject_dsc_mean"]) > historical_subject,
        "absorbs_80pct_ensemble_scan_gain": float(selected_summary["locator_dsc_mean"]) >= (
            historical_scan + 0.8 * (ensemble_scan - historical_scan)
        ),
        "absorbs_80pct_ensemble_subject_gain": float(
            selected_summary["locator_subject_dsc_mean"]
        ) >= (historical_subject + 0.8 * (ensemble_subject - historical_subject)),
        "point_within_one_case_of_ensemble": float(
            selected_summary["automatic_point_hit_rate"]
        ) >= ensemble_point - 1.0 / 108.0,
        "paired_subject_ci_lower_at_least_minus_0.003_vs_history": float(
            versus_historical[selected_key]["ema_minus_raw"]["subject_95ci"][0]
        ) >= -0.003
    }
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summaries": summaries,
        "checkpoints": checkpoints,
        "matched_kd_minus_control": matched,
        "kd_minus_historical": versus_historical,
        "historical": historical,
        "ensemble_teacher": ensemble,
        "selected_kd_epoch": selected_epoch,
        "selected_kd_summary": selected_summary,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "kd_candidate_authorized_for_test": all(gates.values()),
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False
    }
    write_json_atomic(args.output, report)
    print(json.dumps({
        "selected_kd_epoch": selected_epoch,
        "selected_kd": {
            "scan": selected_summary["locator_dsc_mean"],
            "subject": selected_summary["locator_subject_dsc_mean"],
            "point_hit": selected_summary["automatic_point_hit_rate"]
        },
        "historical": {
            "scan": historical_scan,
            "subject": historical_subject,
            "point_hit": historical["automatic_point_hit_rate"]
        },
        "ensemble_teacher": {
            "scan": ensemble_scan,
            "subject": ensemble_subject,
            "point_hit": ensemble_point
        },
        "gates": gates,
        "kd_candidate_authorized_for_test": report["kd_candidate_authorized_for_test"]
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
