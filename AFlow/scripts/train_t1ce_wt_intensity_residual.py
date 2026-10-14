#!/usr/bin/env python3
"""Train matched plain/control/intensity uncertainty-residual locator arms."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import time
from typing import Any

import numpy as np
import torch
from monai.losses import DiceCELoss
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from sammed3d.t1ce_wt_intensity_residual import IntensityUncertaintyResidualLocator
from scripts.train_t1ce_wt_capg3d import (
    BrainCropDataset,
    augment_batch,
    case_view_metadata,
    dice_per_case,
    read_case_ids,
    sha256_file,
    subject_id,
    write_json_atomic,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_intensity_residual_training"
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_intensity_residual_checkpoint"
ARMS = ("plain", "residual_control", "intensity_residual")


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.arm == "plain":
        return AutomaticPromptLocator(
            center_aware=False, init_filters=args.init_filters, dropout=args.dropout
        )
    return IntensityUncertaintyResidualLocator(
        use_intensity=args.arm == "intensity_residual",
        init_filters=args.init_filters,
        dropout=args.dropout,
        residual_channels=args.residual_channels,
        intensity_kernel=args.intensity_kernel,
    )


def output_logits(output: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    final_logits = output.segmentation_logits
    base_logits = getattr(output, "base_logits", final_logits)
    residual_logits = getattr(output, "residual_logits", torch.zeros_like(final_logits))
    return final_logits, base_logits, residual_logits


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_subject: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        by_subject[str(case["subject_id"])].append(float(case["dsc"]))
    values = [float(case["dsc"]) for case in cases]
    base_values = [float(case["base_dsc"]) for case in cases]
    return {
        "cases": len(cases),
        "subjects": len(by_subject),
        "scan_dsc_mean": statistics.fmean(values),
        "scan_dsc_median": statistics.median(values),
        "scan_dsc_std_population": statistics.pstdev(values),
        "subject_dsc_mean": statistics.fmean(
            statistics.fmean(items) for items in by_subject.values()
        ),
        "base_scan_dsc_mean": statistics.fmean(base_values),
        "automatic_point_hit_rate": statistics.fmean(
            float(case["automatic_point_hits_gt"]) for case in cases
        ),
        "per_case": cases,
    }


@torch.inference_mode()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    cases: list[dict[str, Any]] = []
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(image)
        logits, base_logits, residual_logits = output_logits(output)
        dsc = dice_per_case(logits.float(), label)
        base_dsc = dice_per_case(base_logits.float(), label)
        flat_index = torch.argmax(logits[:, 0].flatten(1), dim=1)
        height, width = logits.shape[-2:]
        z = torch.div(flat_index, height * width, rounding_mode="floor")
        remainder = flat_index % (height * width)
        y = torch.div(remainder, width, rounding_mode="floor")
        x = remainder % width
        residual_abs = residual_logits.float().abs().flatten(1).mean(dim=1)
        for index, case_id in enumerate(batch["case_id"]):
            zi, yi, xi = int(z[index]), int(y[index]), int(x[index])
            cases.append({
                "case_id": str(case_id),
                "subject_id": str(batch["subject_id"][index]),
                "dsc": float(dsc[index]),
                "base_dsc": float(base_dsc[index]),
                "residual_abs_mean": float(residual_abs[index]),
                "automatic_point": [zi, yi, xi],
                "automatic_point_hits_gt": bool(label[index, 0, zi, yi, xi] > 0.5),
            })
    return summarize(cases)


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal intensity-residual training requires CUDA")
    torch.cuda.set_device(device)

    fit_ids = read_case_ids(args.fit_case_ids)
    tune_ids = read_case_ids(args.inner_tune_case_ids)
    assert fit_ids is not None and tune_ids is not None
    fit_data = BrainCropDataset(
        args.cache, expected_split="train", case_ids=fit_ids,
        limit=args.limit_fit_cases,
    )
    tune_data = BrainCropDataset(
        args.cache, expected_split="train", case_ids=tune_ids,
        limit=args.limit_tune_cases,
    )
    if set(fit_data.case_ids) & set(tune_data.case_ids):
        raise ValueError("fit/inner-tune case overlap")
    if {subject_id(value) for value in fit_data.case_ids} & {
        subject_id(value) for value in tune_data.case_ids
    }:
        raise ValueError("fit/inner-tune subject overlap")

    generator = torch.Generator().manual_seed(args.seed)
    fit_loader = DataLoader(
        fit_data,
        batch_size=min(args.batch_size, len(fit_data)),
        sampler=WeightedRandomSampler(
            fit_data.sampling_weights(), num_samples=len(fit_data), replacement=True,
            generator=generator,
        ),
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0, drop_last=True,
    )
    tune_loader = DataLoader(
        tune_data, batch_size=min(args.validation_batch_size, len(tune_data)),
        shuffle=False, num_workers=max(1, args.num_workers // 2), pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = DiceCELoss(
        sigmoid=True, squared_pred=False, lambda_dice=1.0, lambda_ce=args.ce_weight
    )
    history: list[dict[str, Any]] = []
    best_value = float("-inf")
    best_epoch = 0
    stale = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        train_dsc: list[float] = []
        train_base_dsc: list[float] = []
        residual_abs: list[float] = []
        for batch in fit_loader:
            image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            image, label = augment_batch(image, label, max_shift=args.max_shift)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(image)
                logits, base_logits, residual_logits = output_logits(output)
                loss = loss_fn(logits, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            train_dsc.extend(dice_per_case(logits.detach().float(), label).cpu().tolist())
            train_base_dsc.extend(
                dice_per_case(base_logits.detach().float(), label).cpu().tolist()
            )
            residual_abs.append(float(residual_logits.detach().float().abs().mean()))

        validation = validate(model, tune_loader, device)
        value = float(validation["scan_dsc_mean"])
        strict_new_high = value > best_value
        if strict_new_high:
            best_value = value
            best_epoch = epoch
            stale = 0
            checkpoint = {
                "schema": CHECKPOINT_SCHEMA,
                "schema_version": 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "epoch": epoch,
                "arm": args.arm,
                "validation": validation,
                "model_state_dict": model.state_dict(),
                "architecture": {
                    "class": model.__class__.__name__,
                    "init_filters": args.init_filters,
                    "dropout": args.dropout,
                    "residual_channels": args.residual_channels,
                    "intensity_kernel": args.intensity_kernel,
                    "use_intensity": args.arm == "intensity_residual",
                },
                "hyperparameters": serializable_args(args),
                "data_roles": {
                    "fit": case_view_metadata(args.fit_case_ids),
                    "inner_tune": case_view_metadata(args.inner_tune_case_ids),
                    "case_overlap": 0,
                    "subject_overlap": 0,
                },
                "fixed_logit_threshold": 0.0,
                "test_access": False,
            }
            torch.save(checkpoint, output_dir / "inner_tune_dice_best.pth")
        else:
            stale += 1
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_dsc": float(np.mean(train_dsc)),
            "train_base_dsc": float(np.mean(train_base_dsc)),
            "train_residual_abs_mean": float(np.mean(residual_abs)),
            "validation": validation,
            "strict_new_high": strict_new_high,
            "stale_epochs": stale,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        write_json_atomic(
            output_dir / "progress.json",
            {"history": history, "best_epoch": best_epoch, "best_dsc": best_value},
        )
        print(
            f"arm={args.arm} epoch={epoch} loss={record['train_loss']:.5f} "
            f"train={record['train_dsc']:.5f} val={value:.6f} "
            f"subject={validation['subject_dsc_mean']:.6f} "
            f"best={best_value:.6f}@{best_epoch}", flush=True,
        )
        if epoch >= args.min_epochs and stale >= args.patience:
            break

    best_path = output_dir / "inner_tune_dice_best.pth"
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "confirmation_access": False,
        "outer_selector_access": False,
        "arm": args.arm,
        "selection": {
            "metric": "inner-tune scan-macro DSC at fixed raw-logit threshold 0",
            "epoch": best_epoch, "value": best_value,
        },
        "best_checkpoint": {"path": str(best_path), "sha256": sha256_file(best_path)},
        "data_roles": {
            "fit": case_view_metadata(args.fit_case_ids),
            "inner_tune": case_view_metadata(args.inner_tune_case_ids),
            "fit_cases": len(fit_data), "inner_tune_cases": len(tune_data),
            "case_overlap": 0, "subject_overlap": 0,
        },
        "cache": {"path": str(args.cache.resolve()), "index_sha256": sha256_file(args.cache / "index.json")},
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "hyperparameters": serializable_args(args),
        "history": history,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
    }
    write_json_atomic(output_dir / "TRAIN_DONE.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--fit-case-ids", type=Path, required=True)
    parser.add_argument("--inner-tune-case-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--ce-weight", type=float, default=0.1)
    parser.add_argument("--max-shift", type=int, default=8)
    parser.add_argument("--init-filters", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--residual-channels", type=int, default=8)
    parser.add_argument("--intensity-kernel", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-fit-cases", type=int, default=None)
    parser.add_argument("--limit-tune-cases", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
