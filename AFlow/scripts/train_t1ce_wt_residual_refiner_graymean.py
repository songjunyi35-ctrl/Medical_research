#!/usr/bin/env python3
"""Paired continuation of the GT-crop residual refiner with gray-mean loss.

This development-only runner starts from a frozen residual-refiner checkpoint.
It preserves the original DiceCE objective and optionally adds a differentiable
version of the MSCL gray-threshold loss.  There is deliberately no test path.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from monai.losses import DiceCELoss

from scripts.train_sammed3d_t1ce_wt_residual_refiner import (
    RefinerDataset,
    ResidualRefiner,
    augment_batch,
    dice_per_case,
    sha256_file,
    write_json_atomic,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_residual_refiner_graymean_training"
SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_residual_refiner_graymean_checkpoint"
SOURCE_CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_residual_refiner_checkpoint"


def soft_gray_mean_mse(
    logits: torch.Tensor,
    image: torch.Tensor,
    label: torch.Tensor,
    *,
    temperature: float = 1.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """MSE between GT and differentiable predicted-region mean intensity.

    All reductions are float32 even under autocast.  The GT-centered training
    cache guarantees a non-empty target, but the denominator is still guarded.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if logits.shape != image.shape or logits.shape != label.shape:
        raise ValueError("logits, image, and label must have identical shapes")
    logits_f = logits.float()
    image_f = image.float()
    target = (label > 0.5).float()
    soft_mask = torch.sigmoid(logits_f / temperature)
    reduce_dims = tuple(range(1, logits.ndim))
    predicted_mean = (soft_mask * image_f).sum(dim=reduce_dims) / soft_mask.sum(
        dim=reduce_dims
    ).clamp_min(epsilon)
    target_mean = (target * image_f).sum(dim=reduce_dims) / target.sum(
        dim=reduce_dims
    ).clamp_min(epsilon)
    return torch.mean(torch.square(predicted_mean - target_mean))


def gray_mean_error_per_case(
    logits: torch.Tensor,
    image: torch.Tensor,
    label: torch.Tensor,
    *,
    temperature: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return soft/hard squared errors and hard-mask non-empty indicators."""
    logits_f = logits.float()
    image_f = image.float()
    target = (label > 0.5).float()
    soft_mask = torch.sigmoid(logits_f / temperature)
    hard_mask = (logits_f > 0).float()
    reduce_dims = tuple(range(1, logits.ndim))
    target_mean = (target * image_f).sum(dim=reduce_dims) / target.sum(
        dim=reduce_dims
    ).clamp_min(epsilon)
    soft_mean = (soft_mask * image_f).sum(dim=reduce_dims) / soft_mask.sum(
        dim=reduce_dims
    ).clamp_min(epsilon)
    hard_count = hard_mask.sum(dim=reduce_dims)
    hard_mean = (hard_mask * image_f).sum(dim=reduce_dims) / hard_count.clamp_min(
        epsilon
    )
    return (
        torch.square(soft_mean - target_mean),
        torch.square(hard_mean - target_mean),
        hard_count > 0,
    )


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    temperature: float,
    epsilon: float,
) -> dict[str, Any]:
    model.eval()
    refined: list[float] = []
    coarse_values: list[float] = []
    soft_gray_mse: list[float] = []
    hard_gray_mse: list[float] = []
    hard_nonempty: list[bool] = []
    case_ids: list[str] = []
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        coarse = batch["coarse"].to(device=device, dtype=torch.float32, non_blocking=True)
        label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(image, coarse)
        logits_f = logits.float()
        refined.extend(dice_per_case(logits_f, label).cpu().tolist())
        coarse_values.extend(dice_per_case(coarse, label).cpu().tolist())
        soft_error, hard_error, nonempty = gray_mean_error_per_case(
            logits_f,
            image,
            label,
            temperature=temperature,
            epsilon=epsilon,
        )
        soft_gray_mse.extend(soft_error.cpu().tolist())
        hard_gray_mse.extend(hard_error.cpu().tolist())
        hard_nonempty.extend(nonempty.cpu().tolist())
        case_ids.extend(list(batch["case_id"]))
    refined_array = np.asarray(refined)
    coarse_array = np.asarray(coarse_values)
    return {
        "cases": len(refined),
        "coarse_dsc_mean": float(np.mean(coarse_array)),
        "refined_dsc_mean": float(np.mean(refined_array)),
        "delta": float(np.mean(refined_array - coarse_array)),
        "improved_cases": int(np.sum(refined_array > coarse_array)),
        "worsened_cases": int(np.sum(refined_array < coarse_array)),
        "soft_gray_mse_mean": float(np.mean(soft_gray_mse)),
        "hard_gray_mse_mean": float(np.mean(hard_gray_mse)),
        "hard_prediction_nonempty_cases": int(np.sum(hard_nonempty)),
        "per_case": [
            {
                "case_id": case_id,
                "coarse_dsc": coarse_dsc,
                "refined_dsc": refined_dsc,
                "soft_gray_mse": soft_error,
                "hard_gray_mse": hard_error,
                "hard_prediction_nonempty": nonempty,
            }
            for case_id, coarse_dsc, refined_dsc, soft_error, hard_error, nonempty in zip(
                case_ids,
                coarse_values,
                refined,
                soft_gray_mse,
                hard_gray_mse,
                hard_nonempty,
                strict=True,
            )
        ],
    }


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    continuation_epoch: int,
    source_checkpoint: Path,
    source_checkpoint_sha256: str,
    source_epoch: int,
    validation: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "continuation_epoch": continuation_epoch,
        "source_epoch": source_epoch,
        "validation": validation,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "initialization": {
            "path": str(source_checkpoint),
            "sha256": source_checkpoint_sha256,
            "optimizer_state_restored": False,
        },
        "objective": {
            "segmentation": "MONAI DiceCELoss(sigmoid=True, lambda_dice=1, lambda_ce=1)",
            "gray_mean": "MSE(soft-predicted-region T1ce mean, GT-region T1ce mean)",
            "gray_weight": args.gray_weight,
            "soft_mask_temperature": args.gray_temperature,
            "epsilon": args.gray_epsilon,
        },
        "architecture": {
            "class": "ResidualRefiner",
            "backbone": "MONAI SegResNet",
            "init_filters": args.init_filters,
            "dropout": args.dropout,
            "input_channels": ["T1ce", "tanh(stage2_click3_raw_logit/5)"],
            "output": "stage2_click3_raw_logit + learned_residual",
        },
    }


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.gray_weight < 0:
        raise ValueError("gray_weight must be non-negative")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("refiner continuation requires CUDA")
    torch.cuda.set_device(device)

    train_data = RefinerDataset(args.train_cache)
    validation_data = RefinerDataset(args.validation_cache)
    if train_data.index["split"] != "train" or validation_data.index["split"] != "validation":
        raise ValueError("train/validation refiner cache split mismatch")
    if train_data.index["model"]["checkpoint_sha256"] != validation_data.index["model"]["checkpoint_sha256"]:
        raise ValueError("train/validation coarse checkpoint mismatch")

    source_checkpoint = args.init_checkpoint.resolve()
    source_sha256 = sha256_file(source_checkpoint)
    if args.expected_init_sha256 and source_sha256 != args.expected_init_sha256:
        raise RuntimeError("initial residual-refiner checkpoint hash mismatch")
    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if source.get("schema") != SOURCE_CHECKPOINT_SCHEMA:
        raise ValueError("unexpected source residual-refiner checkpoint schema")
    architecture = source.get("architecture", {})
    if int(architecture.get("init_filters", -1)) != args.init_filters:
        raise ValueError("init_filters differs from source checkpoint")
    if float(architecture.get("dropout", -1)) != args.dropout:
        raise ValueError("dropout differs from source checkpoint")
    source_epoch = int(source["epoch"])

    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        train_data.sampling_weights(),
        num_samples=len(train_data),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    validation_workers = max(1, args.num_workers // 2)
    validation_loader = DataLoader(
        validation_data,
        batch_size=1,
        shuffle=False,
        num_workers=validation_workers,
        pin_memory=True,
        persistent_workers=validation_workers > 0,
    )
    model = ResidualRefiner(args.init_filters, args.dropout).to(device)
    model.load_state_dict(source["model_state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    segmentation_loss_fn = DiceCELoss(
        sigmoid=True,
        squared_pred=False,
        lambda_dice=1.0,
        lambda_ce=1.0,
    )

    history: list[dict[str, Any]] = []
    started = time.time()
    baseline = validate(
        model,
        validation_loader,
        device,
        temperature=args.gray_temperature,
        epsilon=args.gray_epsilon,
    )
    if args.expected_baseline_dsc is not None and abs(
        baseline["refined_dsc_mean"] - args.expected_baseline_dsc
    ) > args.baseline_tolerance:
        raise RuntimeError(
            "reloaded validation DSC differs from preregistered baseline: "
            f"{baseline['refined_dsc_mean']:.12f} vs {args.expected_baseline_dsc:.12f}"
        )
    best_value = float(baseline["refined_dsc_mean"])
    best_epoch = 0
    stale = 0
    best_path = output_dir / "refiner_val_dice_best.pth"
    torch.save(
        checkpoint_payload(
            model=model,
            optimizer=optimizer,
            continuation_epoch=0,
            source_checkpoint=source_checkpoint,
            source_checkpoint_sha256=source_sha256,
            source_epoch=source_epoch,
            validation=baseline,
            args=args,
        ),
        best_path,
    )
    print(
        f"arm={args.arm_name} continuation_epoch=0 coarse={baseline['coarse_dsc_mean']:.6f} "
        f"refined={baseline['refined_dsc_mean']:.6f} soft_gray_mse={baseline['soft_gray_mse_mean']:.6f}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_losses: list[float] = []
        segmentation_losses: list[float] = []
        gray_losses: list[float] = []
        train_dice: list[float] = []
        for batch in train_loader:
            image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            coarse = batch["coarse"].to(device=device, dtype=torch.float32, non_blocking=True)
            label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            image, coarse, label = augment_batch(image, coarse, label, max_shift=args.max_shift)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(image, coarse)
                segmentation_loss = segmentation_loss_fn(logits, label)
            gray_loss = soft_gray_mean_mse(
                logits,
                image,
                label,
                temperature=args.gray_temperature,
                epsilon=args.gray_epsilon,
            )
            loss = segmentation_loss.float() + args.gray_weight * gray_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_losses.append(float(loss.detach()))
            segmentation_losses.append(float(segmentation_loss.detach()))
            gray_losses.append(float(gray_loss.detach()))
            train_dice.extend(dice_per_case(logits.detach().float(), label).cpu().tolist())

        validation = validate(
            model,
            validation_loader,
            device,
            temperature=args.gray_temperature,
            epsilon=args.gray_epsilon,
        )
        value = float(validation["refined_dsc_mean"])
        strict_new_high = value > best_value
        if strict_new_high:
            best_value = value
            best_epoch = epoch
            stale = 0
            torch.save(
                checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    continuation_epoch=epoch,
                    source_checkpoint=source_checkpoint,
                    source_checkpoint_sha256=source_sha256,
                    source_epoch=source_epoch,
                    validation=validation,
                    args=args,
                ),
                best_path,
            )
        else:
            stale += 1
        record = {
            "continuation_epoch": epoch,
            "source_plus_continuation_epoch": source_epoch + epoch,
            "train_total_loss": float(np.mean(total_losses)),
            "train_segmentation_loss": float(np.mean(segmentation_losses)),
            "train_gray_mean_loss": float(np.mean(gray_losses)),
            "train_dsc": float(np.mean(train_dice)),
            "validation": validation,
            "strict_new_high": strict_new_high,
            "stale_epochs": stale,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        write_json_atomic(
            output_dir / "progress.json",
            {
                "schema": SCHEMA + ".progress",
                "arm_name": args.arm_name,
                "gray_weight": args.gray_weight,
                "baseline_dsc": baseline["refined_dsc_mean"],
                "history": history,
                "best_continuation_epoch": best_epoch,
                "best_dsc": best_value,
                "test_access": False,
            },
        )
        print(
            f"arm={args.arm_name} continuation_epoch={epoch} total={record['train_total_loss']:.5f} "
            f"seg={record['train_segmentation_loss']:.5f} gray={record['train_gray_mean_loss']:.5f} "
            f"train={record['train_dsc']:.5f} val={value:.6f} "
            f"soft_gray_mse={validation['soft_gray_mse_mean']:.6f} best={best_value:.6f}",
            flush=True,
        )
        if stale >= args.patience:
            break

    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "arm_name": args.arm_name,
        "hypothesis": "soft gray-mean consistency improves GT-centered residual refinement",
        "paper_alignment": {
            "paper_term": "L_th = MSE(mean G(R_i), mean G(R_o))",
            "local_interpretation": "R_o is sigmoid(logit / temperature), enabling gradients",
            "known_ambiguity": "paper does not disclose hard/soft mask, coefficient, intensity domain, or empty-mask handling",
        },
        "task": {
            "input": "T1ce single channel plus frozen SAM click-3 raw logits",
            "target": "WT={1,2,3}; raw label 4 excluded",
            "split": "972 train / 108 validation, subject-clean",
            "geometry": "GT-centered N4 128^3 stage-2 cache; development-only",
        },
        "initialization": {
            "path": str(source_checkpoint),
            "sha256": source_sha256,
            "source_epoch": source_epoch,
            "optimizer_state_restored": False,
        },
        "baseline_validation": baseline,
        "selection": {
            "metric": "validation scan-macro DSC",
            "continuation_epoch": best_epoch,
            "source_plus_continuation_epoch": source_epoch + best_epoch,
            "value": best_value,
            "delta_from_reloaded_initialization": best_value - baseline["refined_dsc_mean"],
        },
        "best_checkpoint": {"path": str(best_path), "sha256": sha256_file(best_path)},
        "coarse_checkpoint_sha256": train_data.index["model"]["checkpoint_sha256"],
        "train_cache": {
            "path": str(args.train_cache.resolve()),
            "index_sha256": sha256_file(args.train_cache / "index.json"),
        },
        "validation_cache": {
            "path": str(args.validation_cache.resolve()),
            "index_sha256": sha256_file(args.validation_cache / "index.json"),
        },
        "hyperparameters": serializable_args(args),
        "history": history,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": time.time() - started,
    }
    write_json_atomic(output_dir / "TRAIN_DONE.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-init-sha256", default="")
    parser.add_argument("--expected-baseline-dsc", type=float)
    parser.add_argument("--baseline-tolerance", type=float, default=1e-7)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm-name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--init-filters", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-shift", type=int, default=12)
    parser.add_argument("--gray-weight", type=float, required=True)
    parser.add_argument("--gray-temperature", type=float, default=1.0)
    parser.add_argument("--gray-epsilon", type=float, default=1e-6)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
