#!/usr/bin/env python3
"""Reload and validate a boundary-safe-adapter checkpoint on validation only."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sammed3d.t1ce_wt_boundary_aux import boundary_dice_per_case, boundary_target
from sammed3d.t1ce_wt_safe_adapter import FrozenBaseSafeAdapter
from scripts.train_sammed3d_t1ce_wt_residual_refiner import (
    RefinerDataset,
    ResidualRefiner,
    dice_per_case,
    sha256_file,
)


CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_boundary_safe_adapter_checkpoint"


@torch.inference_mode()
def validate(model, loader, device, *, boundary_radius):
    model.eval()
    final_values, base_values, boundary_values, case_ids = [], [], [], []
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        coarse = batch["coarse"].to(device=device, dtype=torch.float32, non_blocking=True)
        label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model.forward_with_aux(image, coarse)
        target = boundary_target(label, radius=boundary_radius)
        final_values.extend(dice_per_case(output.logits.float(), label).cpu().tolist())
        base_values.extend(dice_per_case(output.base_logits.float(), label).cpu().tolist())
        boundary_values.extend(
            boundary_dice_per_case(output.gate_logits.float(), target).cpu().tolist()
        )
        case_ids.extend(list(batch["case_id"]))
    grouped_final, grouped_base = defaultdict(list), defaultdict(list)
    for case_id, final, base in zip(case_ids, final_values, base_values, strict=True):
        subject = case_id.rsplit("-", 1)[0]
        grouped_final[subject].append(final)
        grouped_base[subject].append(base)
    return {
        "final_dsc_mean": float(np.mean(final_values)),
        "final_subject_dsc_mean": float(
            np.mean([np.mean(values) for values in grouped_final.values()])
        ),
        "base_subject_dsc_mean": float(
            np.mean([np.mean(values) for values in grouped_base.values()])
        ),
        "boundary_head_dsc_mean": float(np.mean(boundary_values)),
    }


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
        raise ValueError("unexpected checkpoint schema")
    architecture = checkpoint["architecture"]
    base = ResidualRefiner(init_filters=16, dropout=0.1)
    model = FrozenBaseSafeAdapter(
        base,
        mode="plain",
        init_filters=int(architecture["adapter_init_filters"]),
        dropout=float(architecture["adapter_dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    dataset = RefinerDataset(args.validation_cache)
    if dataset.index["split"] != "validation" or dataset.index.get("test_access") is not False:
        raise ValueError("only the non-test validation cache is accepted")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    result = validate(
        model,
        loader,
        device,
        boundary_radius=int(checkpoint["objective"]["boundary_radius"]),
    )
    embedded = checkpoint["validation"]
    parity = {
        "scan_abs_diff": abs(result["final_dsc_mean"] - embedded["final_dsc_mean"]),
        "subject_abs_diff": abs(
            result["final_subject_dsc_mean"] - embedded["final_subject_dsc_mean"]
        ),
        "boundary_abs_diff": abs(
            result["boundary_head_dsc_mean"] - embedded["boundary_head_dsc_mean"]
        ),
    }
    if max(parity.values()) > 1e-9:
        raise RuntimeError(f"reload metrics differ from embedded values: {parity}")
    print(
        json.dumps(
            {
                "schema": "aflow.sammed3d.t1ce_wt_boundary_safe_adapter_reload_validation",
                "test_access": False,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "checkpoint_epoch": checkpoint["epoch"],
                "boundary_weight": checkpoint["objective"]["boundary_weight"],
                "validation": result,
                "parity": parity,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
