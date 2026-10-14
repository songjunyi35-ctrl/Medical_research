#!/usr/bin/env python3
"""Compare EMA candidates with the historical best on the same validation108."""
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
from scripts.evaluate_t1ce_wt_plain_ema_validation import (  # noqa: E402
    VALIDATION_MANIFEST_SHA256,
    compare,
)
from scripts.train_t1ce_wt_capg3d import (  # noqa: E402
    BrainCropDataset,
    read_case_ids,
    sha256_file,
    validate,
    write_json_atomic,
)
from scripts.train_t1ce_wt_plain_fixed_epoch import (  # noqa: E402
    CHECKPOINT_SCHEMA as HISTORICAL_CHECKPOINT_SCHEMA,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_ema_vs_historical_validation"
HISTORICAL_CHECKPOINT_SHA256 = "f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1"
EMA_REPORT_SHA256 = "a97d91a113e565a4ac2e5304feeedca4b56ea77c7a8b3d1c329fec41e0347242"


def run(args: argparse.Namespace):
    if sha256_file(args.historical_checkpoint) != HISTORICAL_CHECKPOINT_SHA256:
        raise ValueError("historical best checkpoint SHA-256 mismatch")
    if sha256_file(args.ema_report) != EMA_REPORT_SHA256:
        raise ValueError("EMA validation report SHA-256 mismatch")
    if sha256_file(args.validation_manifest) != VALIDATION_MANIFEST_SHA256:
        raise ValueError("validation manifest SHA-256 mismatch")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    ema_report = json.loads(args.ema_report.read_text(encoding="utf-8"))
    if ema_report.get("test_access") is not False:
        raise ValueError("EMA report accessed test")
    case_ids = read_case_ids(args.validation_manifest)
    assert case_ids is not None
    data = BrainCropDataset(
        args.cache, expected_split="validation", case_ids=case_ids
    )
    if data.index.get("manifest_sha256") != VALIDATION_MANIFEST_SHA256:
        raise ValueError("validation cache binding mismatch")
    checkpoint = torch.load(
        args.historical_checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint.get("schema") != HISTORICAL_CHECKPOINT_SCHEMA:
        raise ValueError("unexpected historical checkpoint schema")
    if checkpoint.get("epoch") != 168:
        raise ValueError("historical checkpoint is not epoch168")
    if any(
        checkpoint.get(key) is not False
        for key in ("validation_access", "outer_selector_access", "confirmation_access", "test_access")
    ):
        raise ValueError("historical checkpoint access flags are not sealed")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("historical validation comparison requires CUDA")
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
    historical = validate(model, loader, device)
    comparisons = {
        "ema168_minus_historical": compare(
            historical, ema_report["summaries"]["ema_168"], seed=20260835
        ),
        "ema200_minus_historical": compare(
            historical, ema_report["summaries"]["ema_200"], seed=20260836
        ),
        "raw168_minus_historical": compare(
            historical, ema_report["summaries"]["raw_168"], seed=20260837
        ),
        "raw200_minus_historical": compare(
            historical, ema_report["summaries"]["raw_200"], seed=20260838
        )
    }
    candidates = {
        name: value
        for name, value in comparisons.items()
        if name.startswith("ema")
    }
    selected = max(
        candidates,
        key=lambda name: (
            candidates[name]["ema"]["scan"],
            candidates[name]["ema"]["subject"]
        )
    )
    selected_better = (
        candidates[selected]["ema_minus_raw"]["scan"] > 0.0
        and candidates[selected]["ema_minus_raw"]["subject"] > 0.0
    )
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "same historical fixed validation108/50 subjects",
        "historical": {
            "checkpoint": str(args.historical_checkpoint.resolve()),
            "checkpoint_sha256": HISTORICAL_CHECKPOINT_SHA256,
            "summary": historical
        },
        "ema_report": {
            "path": str(args.ema_report.resolve()),
            "sha256": EMA_REPORT_SHA256
        },
        "comparisons": comparisons,
        "best_ema_by_validation_scan_then_subject": selected,
        "best_ema_has_positive_scan_and_subject_delta_vs_historical": selected_better,
        "test_evaluation_recommended": selected_better,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False
    }
    write_json_atomic(args.output, report)
    print(json.dumps({
        "historical": {
            "scan": historical["locator_dsc_mean"],
            "subject": historical["locator_subject_dsc_mean"],
            "point_hit": historical["automatic_point_hit_rate"]
        },
        "comparisons": comparisons,
        "selected": selected,
        "test_evaluation_recommended": selected_better
    }, indent=2, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--historical-checkpoint", type=Path, required=True)
    parser.add_argument("--ema-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
