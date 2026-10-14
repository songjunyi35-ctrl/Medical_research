#!/usr/bin/env python3
"""Development-only eight-reflection TTA evaluation for a residual refiner."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calibrate_t1ce_wt_residual_refiner_postprocess import (  # noqa: E402
    binary_dice,
    filter_components,
)
from scripts.train_sammed3d_t1ce_wt_residual_refiner import (  # noqa: E402
    RefinerDataset,
    ResidualRefiner,
    sha256_file,
    write_json_atomic,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_residual_refiner_tta_validation"
SCHEMA_VERSION = 1


def reflection_combinations() -> list[tuple[int, ...]]:
    spatial_axes = (-3, -2, -1)
    return [
        tuple(axis for axis, enabled in zip(spatial_axes, choices, strict=True) if enabled)
        for choices in itertools.product((False, True), repeat=3)
    ]


@torch.inference_mode()
def tta_logits(
    model: torch.nn.Module,
    image: torch.Tensor,
    coarse: torch.Tensor,
) -> torch.Tensor:
    total = torch.zeros_like(coarse, dtype=torch.float32)
    combinations = reflection_combinations()
    for axes in combinations:
        transformed_image = torch.flip(image, dims=axes) if axes else image
        transformed_coarse = torch.flip(coarse, dims=axes) if axes else coarse
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            transformed_logits = model(transformed_image, transformed_coarse)
        logits = torch.flip(transformed_logits, dims=axes) if axes else transformed_logits
        total.add_(logits.float())
    return total / len(combinations)


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    data = RefinerDataset(args.validation_cache)
    if data.index["split"] != "validation" or data.index.get("test_access") is not False:
        raise ValueError("TTA evaluation requires development-only validation cache")
    checkpoint_sha256 = sha256_file(args.checkpoint)
    if args.expected_checkpoint_sha256 and checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise RuntimeError("checkpoint hash mismatch")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    model = ResidualRefiner(
        int(architecture["init_filters"]),
        float(architecture["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("TTA evaluation requires CUDA")
    torch.cuda.set_device(device)
    model.to(device).eval()
    loader = DataLoader(
        data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    plain_values: list[float] = []
    tta_raw_values: list[float] = []
    tta_locked_postprocess_values: list[float] = []
    per_case: list[dict[str, Any]] = []
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        coarse = batch["coarse"].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            plain_logits = model(image, coarse)
        averaged_logits = tta_logits(model, image, coarse)
        target = batch["label"].numpy()[0, 0] > 0.5
        plain_prediction = plain_logits.float().cpu().numpy()[0, 0] > 0.0
        tta_logits_np = averaged_logits.cpu().numpy()[0, 0]
        tta_raw_prediction = tta_logits_np > 0.0
        tta_locked_prediction = filter_components(
            tta_logits_np > args.locked_threshold,
            args.locked_mode,
        )
        plain_dsc = binary_dice(plain_prediction, target)
        tta_raw_dsc = binary_dice(tta_raw_prediction, target)
        tta_locked_dsc = binary_dice(tta_locked_prediction, target)
        plain_values.append(plain_dsc)
        tta_raw_values.append(tta_raw_dsc)
        tta_locked_postprocess_values.append(tta_locked_dsc)
        per_case.append(
            {
                "case_id": str(batch["case_id"][0]),
                "plain_threshold0_raw_dsc": plain_dsc,
                "tta_threshold0_raw_dsc": tta_raw_dsc,
                "tta_locked_postprocess_dsc": tta_locked_dsc,
            }
        )

    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "split": "validation",
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": checkpoint_sha256,
        },
        "tta": {
            "method": "float32 mean of logits after all 8 combinations of 3D axis reflection and inverse reflection",
            "combinations": [list(axes) for axes in reflection_combinations()],
            "uses_label": False,
        },
        "locked_postprocess": {
            "threshold": args.locked_threshold,
            "mode": args.locked_mode,
            "source": "prior frozen postprocess calibration",
        },
        "metrics": {
            "plain_threshold0_raw_scan_macro_dsc": float(np.mean(plain_values)),
            "tta_threshold0_raw_scan_macro_dsc": float(np.mean(tta_raw_values)),
            "tta_locked_postprocess_scan_macro_dsc": float(
                np.mean(tta_locked_postprocess_values)
            ),
            "tta_raw_minus_plain": float(np.mean(tta_raw_values) - np.mean(plain_values)),
        },
        "per_case": per_case,
        "validation_cache_index_sha256": sha256_file(args.validation_cache / "index.json"),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": time.time() - started,
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report["metrics"], indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--locked-threshold", type=float, default=-1.0)
    parser.add_argument("--locked-mode", default="remove_lt_500")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
