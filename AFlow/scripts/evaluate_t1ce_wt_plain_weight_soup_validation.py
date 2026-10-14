#!/usr/bin/env python3
"""Evaluate the fixed parameter soup on validation108 against historical best."""
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
from scripts.make_t1ce_wt_plain_weight_soup import EMA_WEIGHT, SCHEMA as SOUP_SCHEMA  # noqa: E402
from scripts.train_t1ce_wt_capg3d import (  # noqa: E402
    BrainCropDataset,
    read_case_ids,
    sha256_file,
    validate,
    write_json_atomic,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_weight_soup_validation"
VALIDATION_MANIFEST_SHA256 = "f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb"
HISTORICAL_COMPARISON_SHA256 = "3404d598ade63ec6d2b7ccd44f67e284ca7f2c7c7a904546fdfcd1ed8fb95f12"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--soup", type=Path, required=True)
    parser.add_argument("--historical-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if sha256_file(args.validation_manifest) != VALIDATION_MANIFEST_SHA256:
        raise ValueError("validation manifest SHA-256 mismatch")
    if sha256_file(args.historical_comparison) != HISTORICAL_COMPARISON_SHA256:
        raise ValueError("historical validation report SHA-256 mismatch")
    if args.output.exists() or args.output.with_suffix(".started.json").exists():
        raise FileExistsError("refusing repeated soup validation")
    soup = torch.load(args.soup, map_location="cpu", weights_only=False)
    if soup.get("schema") != SOUP_SCHEMA or soup.get("ema_weight") != EMA_WEIGHT:
        raise ValueError("unexpected fixed soup checkpoint")
    if any(
        soup.get(key) is not False
        for key in ("validation_access", "outer_selector_access", "confirmation_access", "test_access")
    ):
        raise ValueError("soup checkpoint declares prior evaluation access")
    historical_report = json.loads(
        args.historical_comparison.read_text(encoding="utf-8")
    )
    historical = historical_report["historical"]["summary"]
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
        "soup_sha256": sha256_file(args.soup),
        "ema_weight": EMA_WEIGHT,
        "threshold": 0.0,
        "test_access": False
    })
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("soup validation requires CUDA")
    torch.cuda.set_device(device)
    architecture = soup["architecture"]
    model = AutomaticPromptLocator(
        center_aware=False,
        init_filters=int(architecture["init_filters"]),
        dropout=float(architecture["dropout"])
    ).to(device)
    model.load_state_dict(soup["model_state_dict"], strict=True)
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0
    )
    summary = validate(model, loader, device)
    paired = compare(historical, summary, seed=20260845)
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": soup["formula"],
        "soup": {
            "path": str(args.soup.resolve()),
            "sha256": sha256_file(args.soup),
            "ema_weight": EMA_WEIGHT
        },
        "historical": historical,
        "soup_summary": summary,
        "paired": paired,
        "candidate_authorized_for_test": paired["all_gates_passed"],
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False
    }
    write_json_atomic(args.output, report)
    print(json.dumps({
        "historical": paired["raw"],
        "soup": paired["ema"],
        "soup_minus_historical": paired["ema_minus_raw"],
        "candidate_authorized_for_test": report["candidate_authorized_for_test"]
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
