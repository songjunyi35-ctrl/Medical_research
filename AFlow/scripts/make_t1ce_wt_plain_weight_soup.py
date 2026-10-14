#!/usr/bin/env python3
"""Create the fixed 0.75 historical + 0.25 EMA168 parameter soup."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator  # noqa: E402
from scripts.train_t1ce_wt_capg3d import sha256_file  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_weight_soup_checkpoint"
HISTORICAL_SHA256 = "f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1"
EMA168_SHA256 = "4ea97dacb9a538993e1c16593bc56b03ce6dc3df2f457c1f0426439c6252197e"
EMA_WEIGHT = 0.25


def interpolate_state_dict(
    historical: dict[str, torch.Tensor],
    ema: dict[str, torch.Tensor],
    ema_weight: float,
) -> dict[str, torch.Tensor]:
    if historical.keys() != ema.keys():
        raise ValueError("checkpoint state_dict keys differ")
    if not 0.0 <= ema_weight <= 1.0:
        raise ValueError("EMA weight must be in [0,1]")
    result = {}
    for key in historical:
        left = historical[key]
        right = ema[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"state tensor metadata differs: {key}")
        if left.is_floating_point():
            result[key] = left * (1.0 - ema_weight) + right * ema_weight
        else:
            result[key] = left.clone()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--ema168", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ema-weight", type=float, default=EMA_WEIGHT)
    args = parser.parse_args()
    if args.ema_weight != EMA_WEIGHT:
        raise ValueError("frozen soup protocol requires EMA weight 0.25")
    if sha256_file(args.historical) != HISTORICAL_SHA256:
        raise ValueError("historical checkpoint SHA-256 mismatch")
    if sha256_file(args.ema168) != EMA168_SHA256:
        raise ValueError("EMA168 checkpoint SHA-256 mismatch")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    historical = torch.load(args.historical, map_location="cpu", weights_only=False)
    ema = torch.load(args.ema168, map_location="cpu", weights_only=False)
    if historical["architecture"] != ema["architecture"]:
        raise ValueError("checkpoint architectures differ")
    state = interpolate_state_dict(
        historical["model_state_dict"], ema["model_state_dict"], EMA_WEIGHT
    )
    architecture = historical["architecture"]
    model = AutomaticPromptLocator(
        center_aware=False,
        init_filters=int(architecture["init_filters"]),
        dropout=float(architecture["dropout"])
    )
    model.load_state_dict(state, strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != 4_700_897:
        raise ValueError("unexpected soup model parameter count")
    value = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "resolution": 128,
        "epoch_labels": [168, 168],
        "formula": "0.75 * historical_epoch168_parameters + 0.25 * EMA168_parameters",
        "ema_weight": EMA_WEIGHT,
        "model_state_dict": state,
        "architecture": architecture,
        "components": {
            "historical": {
                "path": str(args.historical.resolve()),
                "sha256": HISTORICAL_SHA256
            },
            "ema168": {
                "path": str(args.ema168.resolve()),
                "sha256": EMA168_SHA256
            }
        },
        "selection": "fixed weight copied from validation-selected logit ensemble",
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "sha256": sha256_file(args.output),
        "ema_weight": EMA_WEIGHT,
        "parameter_count": 4_700_897
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
