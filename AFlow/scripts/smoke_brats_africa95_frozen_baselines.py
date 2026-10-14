#!/usr/bin/env python3
"""Synthetic-only CUDA smoke for the frozen Africa95 model/view panel."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

import scripts.evaluate_ucsd_ptgbm_external_confirmation as model_io
from scripts import brats_africa95_protocol as p


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--device", default="cuda:0"); args = parser.parse_args()
    output = p.RECORD / "SMOKE_RECEIPT.json"
    if output.exists(): raise FileExistsError(output)
    p.documents(); device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
    image = torch.randn((1, 1, 128, 128, 128), generator=torch.Generator().manual_seed(2026090601)).to(device)
    full, _ = model_io.load_model(p.FULL972, p.FULL972_SHA, "primary", device); total = None; identity = None
    for axes in p.AXES:
        view = torch.flip(image, axes) if axes else image
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): logits = full(view).segmentation_logits.float()
        restored = torch.flip(logits, axes) if axes else logits
        if identity is None: identity = restored.clone(); total = torch.zeros_like(restored)
        total.add_(restored)
    del full; torch.cuda.empty_cache(); gcsam, _ = model_io.load_model(p.GCSAM184, p.GCSAM184_SHA, "secondary", device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16): gcsam_logits = gcsam(image).segmentation_logits.float()
    if identity.shape != image.shape or total.shape != image.shape or gcsam_logits.shape != image.shape or not torch.isfinite(total).all() or not torch.isfinite(gcsam_logits).all(): raise RuntimeError("synthetic output gate")
    receipt = {"schema": "aflow.brats_africa95.synthetic_cuda_smoke", "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "passed", "synthetic_only": True, "source_or_label_access": False, "full972_forwards": 8, "gcsam184_forwards": 1, "outputs": {"identity_shape": list(identity.shape), "flip8_shape": list((total / 8).shape), "gcsam_shape": list(gcsam_logits.shape)}, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device), "checkpoint_sha256": [p.FULL972_SHA, p.GCSAM184_SHA], "runner_sha256": p.sha(Path(__file__))}
    p.write_new(output, receipt); print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__": main()
