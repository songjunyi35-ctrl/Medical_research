#!/usr/bin/env python3
"""Validation-only threshold and connected-component calibration for a refiner.

The input cache must explicitly be development-only.  This script has no test
mode and records every candidate, avoiding hidden postprocessing choices.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy import ndimage
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_sammed3d_t1ce_wt_residual_refiner import (  # noqa: E402
    RefinerDataset,
    ResidualRefiner,
    sha256_file,
    write_json_atomic,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_residual_refiner_postprocess_calibration"
SCHEMA_VERSION = 1
CONNECTIVITY_26 = ndimage.generate_binary_structure(3, 3)


def filter_components(prediction: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return prediction
    components, count = ndimage.label(prediction, structure=CONNECTIVITY_26)
    if count == 0:
        return prediction
    sizes = np.bincount(components.reshape(-1))
    sizes[0] = 0
    if mode == "largest_component":
        return components == int(np.argmax(sizes))
    if mode.startswith("remove_lt_"):
        minimum = int(mode.removeprefix("remove_lt_"))
        keep = sizes >= minimum
        keep[0] = False
        return keep[components]
    raise ValueError(f"unknown postprocessing mode: {mode}")


def binary_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = int(np.count_nonzero(prediction & target))
    denominator = int(np.count_nonzero(prediction)) + int(np.count_nonzero(target))
    return 2.0 * intersection / denominator if denominator else 1.0


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    validation_data = RefinerDataset(args.validation_cache)
    if validation_data.index["split"] != "validation":
        raise ValueError("postprocess calibration requires validation split")
    if validation_data.index.get("test_access") is not False:
        raise ValueError("postprocess calibration cannot consume test data")
    checkpoint_sha256 = sha256_file(args.checkpoint)
    if args.expected_checkpoint_sha256 and checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise RuntimeError("refiner checkpoint hash mismatch")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    model = ResidualRefiner(
        int(architecture["init_filters"]),
        float(architecture["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("postprocess calibration requires CUDA")
    torch.cuda.set_device(device)
    model.to(device).eval()
    loader = DataLoader(
        validation_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    thresholds = list(args.thresholds)
    modes = list(args.modes)
    values: dict[tuple[float, str], list[float]] = {
        (threshold, mode): [] for threshold in thresholds for mode in modes
    }
    case_ids: list[str] = []
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        coarse = batch["coarse"].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(image, coarse)
        logits_np = logits.float().cpu().numpy()[0, 0]
        target = batch["label"].numpy()[0, 0] > 0.5
        for threshold in thresholds:
            prediction = logits_np > threshold
            for mode in modes:
                filtered = filter_components(prediction, mode)
                values[(threshold, mode)].append(binary_dice(filtered, target))
        case_ids.append(str(batch["case_id"][0]))

    candidates: list[dict[str, Any]] = []
    for threshold in thresholds:
        for mode in modes:
            dice_values = values[(threshold, mode)]
            candidates.append(
                {
                    "threshold": threshold,
                    "mode": mode,
                    "scan_macro_dsc": float(np.mean(dice_values)),
                    "median_dsc": float(np.median(dice_values)),
                    "per_case": [
                        {"case_id": case_id, "dsc": dsc}
                        for case_id, dsc in zip(case_ids, dice_values, strict=True)
                    ],
                }
            )
    candidates.sort(key=lambda item: item["scan_macro_dsc"], reverse=True)
    threshold_zero_raw = next(
        item for item in candidates if item["threshold"] == 0.0 and item["mode"] == "raw"
    )
    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "selection_split": "validation",
        "selection_metric": "scan-macro Dice",
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": checkpoint_sha256,
            "stored_validation": checkpoint.get("validation", {}).get("refined_dsc_mean"),
        },
        "validation_cache": {
            "path": str(args.validation_cache.resolve()),
            "count": len(validation_data),
            "index_sha256": sha256_file(args.validation_cache / "index.json"),
        },
        "candidate_grid": {"thresholds": thresholds, "modes": modes},
        "threshold_zero_raw": threshold_zero_raw,
        "best": candidates[0],
        "candidates": candidates,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": time.time() - started,
    }
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "threshold_zero_raw": threshold_zero_raw["scan_macro_dsc"],
                "best": {
                    key: candidates[0][key]
                    for key in ("threshold", "mode", "scan_macro_dsc")
                },
            },
            indent=2,
        ),
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[-1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["raw", "remove_lt_100", "remove_lt_500", "largest_component"],
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
