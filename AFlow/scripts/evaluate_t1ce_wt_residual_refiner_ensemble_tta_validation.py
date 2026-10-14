#!/usr/bin/env python3
"""Development-only equal model-and-reflection logit ensemble evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from scripts.evaluate_t1ce_wt_residual_refiner_ensemble_validation import (  # noqa: E402
    load_model,
    mean_logits,
)
from scripts.evaluate_t1ce_wt_residual_refiner_tta_validation import (  # noqa: E402
    reflection_combinations,
)
from scripts.train_sammed3d_t1ce_wt_residual_refiner import (  # noqa: E402
    RefinerDataset,
    sha256_file,
    write_json_atomic,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_residual_refiner_ensemble_tta_validation"
SCHEMA_VERSION = 1


@torch.inference_mode()
def ensemble_tta_logits(
    models: list[torch.nn.Module],
    image: torch.Tensor,
    coarse: torch.Tensor,
) -> torch.Tensor:
    if not models:
        raise ValueError("at least one model is required")
    predictions: list[torch.Tensor] = []
    for axes in reflection_combinations():
        transformed_image = torch.flip(image, dims=axes) if axes else image
        transformed_coarse = torch.flip(coarse, dims=axes) if axes else coarse
        for model in models:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                transformed_logits = model(transformed_image, transformed_coarse)
            logits = torch.flip(transformed_logits, dims=axes) if axes else transformed_logits
            predictions.append(logits.float())
    return mean_logits(predictions)


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    if len(args.checkpoints) != len(args.expected_sha256):
        raise ValueError("checkpoints and expected hashes must have equal length")
    if len(args.checkpoints) < 2:
        raise ValueError("ensemble requires at least two checkpoints")
    data = RefinerDataset(args.validation_cache)
    if data.index["split"] != "validation" or data.index.get("test_access") is not False:
        raise ValueError("ensemble TTA requires development-only validation cache")
    actual_hashes = [sha256_file(path) for path in args.checkpoints]
    if actual_hashes != list(args.expected_sha256):
        raise RuntimeError("ensemble checkpoint hash mismatch")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ensemble TTA requires CUDA")
    torch.cuda.set_device(device)
    models = [load_model(path, device) for path in args.checkpoints]
    loader = DataLoader(
        data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    raw_values: list[float] = []
    locked_values: list[float] = []
    per_case: list[dict[str, Any]] = []
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        coarse = batch["coarse"].to(device=device, dtype=torch.float32, non_blocking=True)
        averaged = ensemble_tta_logits(models, image, coarse).cpu().numpy()[0, 0]
        target = batch["label"].numpy()[0, 0] > 0.5
        raw_dsc = binary_dice(averaged > 0.0, target)
        locked_prediction = filter_components(
            averaged > args.locked_threshold,
            args.locked_mode,
        )
        locked_dsc = binary_dice(locked_prediction, target)
        raw_values.append(raw_dsc)
        locked_values.append(locked_dsc)
        per_case.append(
            {
                "case_id": str(batch["case_id"][0]),
                "ensemble_tta_threshold0_raw_dsc": raw_dsc,
                "ensemble_tta_locked_postprocess_dsc": locked_dsc,
            }
        )

    metrics = {
        "ensemble_tta_threshold0_raw_scan_macro_dsc": float(np.mean(raw_values)),
        "ensemble_tta_locked_postprocess_scan_macro_dsc": float(np.mean(locked_values)),
    }
    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "split": "validation",
        "aggregation": "equal float32 logit mean over 3 models x 8 spatial reflections",
        "members": [
            {"path": str(path.resolve()), "sha256": digest}
            for path, digest in zip(args.checkpoints, actual_hashes, strict=True)
        ],
        "reflections": [list(axes) for axes in reflection_combinations()],
        "locked_postprocess": {
            "threshold": args.locked_threshold,
            "mode": args.locked_mode,
            "new_threshold_search": False,
        },
        "metrics": metrics,
        "per_case": per_case,
        "validation_cache_index_sha256": sha256_file(args.validation_cache / "index.json"),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": time.time() - started,
    }
    write_json_atomic(args.output, report)
    print(json.dumps(metrics, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-sha256", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--locked-threshold", type=float, default=-1.0)
    parser.add_argument("--locked-mode", default="remove_lt_500")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
