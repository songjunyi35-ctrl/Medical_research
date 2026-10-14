#!/usr/bin/env python3
"""Independently reload and validate a frozen-base safe-adapter checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sammed3d.t1ce_wt_safe_adapter import FrozenBaseSafeAdapter
from scripts.train_sammed3d_t1ce_wt_residual_refiner import RefinerDataset, ResidualRefiner
from scripts.train_t1ce_wt_safe_adapter import CHECKPOINT_SCHEMA, sha256_file, validate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("reload validation requires CUDA")
    torch.cuda.set_device(device)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unexpected safe-adapter checkpoint schema")
    architecture = checkpoint["architecture"]
    base = ResidualRefiner(init_filters=16, dropout=0.1)
    model = FrozenBaseSafeAdapter(
        base,
        mode=architecture["mode"],
        init_filters=int(architecture["adapter_init_filters"]),
        dropout=float(architecture["adapter_dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    dataset = RefinerDataset(args.validation_cache)
    if dataset.index["split"] != "validation" or dataset.index.get("test_access") is not False:
        raise ValueError("only the non-test validation cache is accepted")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    result = validate(model, loader, device)
    embedded = checkpoint["validation"]
    parity = {
        "scan_abs_diff": abs(result["final_dsc_mean"] - embedded["final_dsc_mean"]),
        "subject_abs_diff": abs(
            result["final_subject_dsc_mean"] - embedded["final_subject_dsc_mean"]
        ),
    }
    if parity["scan_abs_diff"] > 1e-9 or parity["subject_abs_diff"] > 1e-9:
        raise RuntimeError(f"reload validation differs from embedded metrics: {parity}")
    print(
        json.dumps(
            {
                "schema": "aflow.sammed3d.t1ce_wt_safe_adapter_reload_validation",
                "test_access": False,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "mode": architecture["mode"],
                "checkpoint_epoch": checkpoint["epoch"],
                "validation": result,
                "parity": parity,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
