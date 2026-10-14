#!/usr/bin/env python3
"""Compare ReCo-inspired KD checkpoints against base voxel-KD on validation108."""
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
from scripts.train_t1ce_wt_plain_reco_kd_continuation import CHECKPOINT_SCHEMA  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_reco_kd_validation"
BASE_KD_REPORT_SHA256 = "db7c3bc6b5ff9e1076827d25c9f4de0dea88781bfc5de45da0fb2f79a09bfb57"
VALIDATION_MANIFEST_SHA256 = "f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb"
EPOCHS = (176, 184, 192, 200)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--reco-dir", type=Path, required=True)
    parser.add_argument("--base-kd-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if sha256_file(args.validation_manifest) != VALIDATION_MANIFEST_SHA256:
        raise ValueError("validation manifest SHA-256 mismatch")
    if sha256_file(args.base_kd_report) != BASE_KD_REPORT_SHA256:
        raise ValueError("base KD validation report SHA-256 mismatch")
    if args.output.exists():
        raise FileExistsError("refusing repeated ReCo validation")
    base_report = json.loads(args.base_kd_report.read_text(encoding="utf-8"))
    historical = base_report["historical"]
    case_ids = read_case_ids(args.validation_manifest)
    assert case_ids is not None
    data = BrainCropDataset(args.cache, expected_split="validation", case_ids=case_ids)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ReCo validation requires CUDA")
    torch.cuda.set_device(device)
    loader = DataLoader(
        data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0
    )
    summaries, checkpoints, versus_base, versus_history = {}, {}, {}, {}
    for index, epoch in enumerate(EPOCHS):
        path = args.reco_dir / f"plain_128_reco_kd_global_epoch{epoch}.pth"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema") != CHECKPOINT_SCHEMA or checkpoint.get("global_epoch") != epoch:
            raise ValueError("unexpected ReCo checkpoint")
        architecture = checkpoint["architecture"]
        model = AutomaticPromptLocator(
            center_aware=False,
            init_filters=int(architecture["init_filters"]),
            dropout=float(architecture["dropout"])
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        key = str(epoch)
        summaries[key] = validate(model, loader, device)
        checkpoints[key] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
        versus_base[key] = compare(
            base_report["summaries"]["kd"][key], summaries[key], seed=20260860 + index
        )
        versus_history[key] = compare(historical, summaries[key], seed=20260864 + index)
        del model
        torch.cuda.empty_cache()
    selected_epoch = max(
        EPOCHS,
        key=lambda epoch: (
            summaries[str(epoch)]["locator_dsc_mean"],
            summaries[str(epoch)]["locator_subject_dsc_mean"]
        )
    )
    key = str(selected_epoch)
    base_delta = versus_base[key]["ema_minus_raw"]
    history_delta = versus_history[key]["ema_minus_raw"]
    gates = {
        "scan_gain_ge_0.001_or_point_plus_one_case": (
            base_delta["scan"] >= 0.001 or base_delta["point_hit"] >= 1.0 / 108.0
        ),
        "subject_drop_at_most_0.001_vs_base": base_delta["subject"] >= -0.001,
        "beats_historical_scan": history_delta["scan"] > 0.0,
        "beats_historical_subject": history_delta["subject"] > 0.0
    }
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summaries": summaries,
        "checkpoints": checkpoints,
        "reco_minus_base_kd": versus_base,
        "reco_minus_historical": versus_history,
        "selected_epoch": selected_epoch,
        "selected_summary": summaries[key],
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "reco_candidate_authorized_for_test": all(gates.values()),
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False
    }
    write_json_atomic(args.output, report)
    print(json.dumps({
        "selected_epoch": selected_epoch,
        "selected": {
            "scan": summaries[key]["locator_dsc_mean"],
            "subject": summaries[key]["locator_subject_dsc_mean"],
            "point_hit": summaries[key]["automatic_point_hit_rate"]
        },
        "reco_minus_base_kd": base_delta,
        "gates": gates,
        "reco_candidate_authorized_for_test": report["reco_candidate_authorized_for_test"]
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
