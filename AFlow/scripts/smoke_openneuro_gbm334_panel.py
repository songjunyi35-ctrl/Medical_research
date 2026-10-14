#!/usr/bin/env python3
"""Synthetic-only CUDA smoke for the frozen OpenNeuro-GBM334 panel."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

import scripts.evaluate_ucsd_ptgbm_external_confirmation as model_io
from scripts import openneuro_gbm334_protocol as p


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = p.RECORD / "SMOKE_RECEIPT.json"
    if output.exists():
        raise FileExistsError(output)
    p.docs()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.cuda.set_device(device)
    generator = torch.Generator().manual_seed(p.BOOTSTRAP_SEED)
    tensor = torch.randn((1, 1, 128, 128, 128), generator=generator).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    full, _ = model_io.load_model(p.FULL, p.FULL_SHA, "primary", device)
    total = None
    for axes in p.AXES:
        view = torch.flip(tensor, axes) if axes else tensor
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = full(view).segmentation_logits.float()
        logits = torch.flip(logits, axes) if axes else logits
        total = logits.clone() if total is None else total + logits
    gcsam, _ = model_io.load_model(p.GCSAM, p.GCSAM_SHA, "secondary", device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        gcsam_logits = gcsam(tensor).segmentation_logits.float()
    if (
        total is None
        or total.shape != tensor.shape
        or gcsam_logits.shape != tensor.shape
        or not torch.isfinite(total).all()
        or not torch.isfinite(gcsam_logits).all()
    ):
        raise RuntimeError("synthetic output contract")
    receipt = {
        "schema": "aflow.openneuro_gbm334.synthetic_smoke",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "synthetic_only": True,
        "openneuro_payload_access": False,
        "forwards": 9,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "full_checkpoint_sha256": p.FULL_SHA,
        "gcsam_checkpoint_sha256": p.GCSAM_SHA,
        "runner_sha256": p.sha256(Path(__file__)),
    }
    p.write_new(output, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

