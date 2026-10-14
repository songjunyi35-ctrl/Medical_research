#!/usr/bin/env python3
"""Matched full-data control/boundary training for the frozen-base safe adapter."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from monai.losses import DiceCELoss, DiceLoss

from sammed3d.t1ce_wt_boundary_aux import boundary_dice_per_case, boundary_target
from sammed3d.t1ce_wt_safe_adapter import FrozenBaseSafeAdapter
from scripts.train_sammed3d_t1ce_wt_residual_refiner import (
    RefinerDataset,
    ResidualRefiner,
    augment_batch,
    dice_per_case,
    sha256_file,
    write_json_atomic,
)
from scripts.train_t1ce_wt_residual_refiner_graymean_robust import corrupt_coarse_batch
from scripts.train_t1ce_wt_safe_adapter import (
    SOURCE_CHECKPOINT_SCHEMA,
    serializable_args,
    subject_macro,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_boundary_safe_adapter_training"
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_boundary_safe_adapter_checkpoint"
SCHEMA_VERSION = 1


@torch.inference_mode()
def validate(
    model: FrozenBaseSafeAdapter,
    loader: DataLoader,
    device: torch.device,
    *,
    boundary_radius: int,
) -> dict[str, Any]:
    model.eval()
    final_values: list[float] = []
    base_values: list[float] = []
    coarse_values: list[float] = []
    boundary_values: list[float] = []
    boundary_fractions: list[float] = []
    case_ids: list[str] = []
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        coarse = batch["coarse"].to(device=device, dtype=torch.float32, non_blocking=True)
        label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model.forward_with_aux(image, coarse)
        target = boundary_target(label, radius=boundary_radius)
        final_values.extend(dice_per_case(output.logits.float(), label).cpu().tolist())
        base_values.extend(dice_per_case(output.base_logits.float(), label).cpu().tolist())
        coarse_values.extend(dice_per_case(coarse, label).cpu().tolist())
        boundary_values.extend(
            boundary_dice_per_case(output.gate_logits.float(), target).cpu().tolist()
        )
        boundary_fractions.extend(target.flatten(1).mean(1).cpu().tolist())
        case_ids.extend(list(batch["case_id"]))
    final_array = np.asarray(final_values)
    base_array = np.asarray(base_values)
    return {
        "cases": len(case_ids),
        "subjects": len({case_id.rsplit("-", 1)[0] for case_id in case_ids}),
        "coarse_dsc_mean": float(np.mean(coarse_values)),
        "base_dsc_mean": float(np.mean(base_array)),
        "base_subject_dsc_mean": subject_macro(case_ids, base_values),
        "final_dsc_mean": float(np.mean(final_array)),
        "final_subject_dsc_mean": subject_macro(case_ids, final_values),
        "delta_from_base": float(np.mean(final_array - base_array)),
        "improved_vs_base_cases": int(np.sum(final_array > base_array)),
        "worsened_vs_base_cases": int(np.sum(final_array < base_array)),
        "boundary_head_dsc_mean": float(np.mean(boundary_values)),
        "boundary_target_fraction_mean": float(np.mean(boundary_fractions)),
        "per_case": [
            {
                "case_id": case_id,
                "coarse_dsc": coarse,
                "base_dsc": base,
                "final_dsc": final,
                "delta_from_base": final - base,
                "boundary_head_dsc": boundary_dsc,
            }
            for case_id, coarse, base, final, boundary_dsc in zip(
                case_ids,
                coarse_values,
                base_values,
                final_values,
                boundary_values,
                strict=True,
            )
        ],
    }


def checkpoint_payload(
    *,
    model: FrozenBaseSafeAdapter,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation: dict[str, Any],
    args: argparse.Namespace,
    source_checkpoint: Path,
    source_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "epoch": epoch,
        "validation": validation,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "initialization": {
            "path": str(source_checkpoint),
            "sha256": source_sha256,
            "base_frozen": True,
            "adapter_zero_initialized": True,
        },
        "architecture": {
            "name": "SAGE-3D frozen-base safe adapter with optional boundary supervision",
            "base": "frozen robust ResidualRefiner",
            "adapter_mode": "plain",
            "adapter_init_filters": args.adapter_init_filters,
            "adapter_dropout": args.adapter_dropout,
            "heads": ["adapter_residual", "boundary"],
            "output": "base_logit + adapter_residual",
            "boundary_head_used_at_inference": False,
        },
        "objective": {
            "segmentation": "MONAI DiceCELoss(sigmoid=True)",
            "boundary": "MONAI DiceLoss(sigmoid=True) on symmetric 3D morphological band",
            "boundary_weight": args.boundary_weight,
            "boundary_radius": args.boundary_radius,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.boundary_weight < 0:
        raise ValueError("boundary_weight must be non-negative")
    if args.boundary_radius < 1:
        raise ValueError("boundary_radius must be at least one")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("boundary safe-adapter training requires CUDA")
    torch.cuda.set_device(device)

    train_source = RefinerDataset(args.train_cache)
    validation_source = RefinerDataset(args.validation_cache)
    if train_source.index["split"] != "train" or validation_source.index["split"] != "validation":
        raise ValueError("train/validation cache split mismatch")
    if train_source.index.get("test_access") is not False or validation_source.index.get("test_access") is not False:
        raise ValueError("test-accessible cache is forbidden")
    if train_source.index["model"]["checkpoint_sha256"] != validation_source.index["model"]["checkpoint_sha256"]:
        raise ValueError("train/validation coarse checkpoint mismatch")

    source_checkpoint = args.init_checkpoint.resolve()
    source_sha256 = sha256_file(source_checkpoint)
    if args.expected_init_sha256 and source_sha256 != args.expected_init_sha256:
        raise RuntimeError("robust base checkpoint hash mismatch")
    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if source.get("schema") != SOURCE_CHECKPOINT_SCHEMA:
        raise ValueError("unexpected robust base checkpoint schema")

    train_limit = args.limit_train_cases or len(train_source)
    validation_limit = args.limit_validation_cases or len(validation_source)
    if not 0 < train_limit <= len(train_source) or not 0 < validation_limit <= len(validation_source):
        raise ValueError("invalid case limit")
    train_data = train_source if train_limit == len(train_source) else Subset(train_source, range(train_limit))
    validation_data = validation_source if validation_limit == len(validation_source) else Subset(validation_source, range(validation_limit))
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        train_source.sampling_weights()[:train_limit],
        num_samples=train_limit,
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
    validation_workers = max(0, args.num_workers // 2)
    validation_loader = DataLoader(
        validation_data,
        batch_size=1,
        shuffle=False,
        num_workers=validation_workers,
        pin_memory=True,
        persistent_workers=validation_workers > 0,
    )

    base = ResidualRefiner(args.base_init_filters, args.base_dropout)
    base.load_state_dict(source["model_state_dict"], strict=True)
    model = FrozenBaseSafeAdapter(
        base,
        mode="plain",
        init_filters=args.adapter_init_filters,
        dropout=args.adapter_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    segmentation_loss_fn = DiceCELoss(
        sigmoid=True,
        squared_pred=False,
        lambda_dice=1.0,
        lambda_ce=1.0,
    )
    boundary_loss_fn = DiceLoss(sigmoid=True, squared_pred=False, reduction="mean")

    started = time.time()
    baseline = validate(
        model,
        validation_loader,
        device,
        boundary_radius=args.boundary_radius,
    )
    if abs(baseline["final_dsc_mean"] - baseline["base_dsc_mean"]) > 1e-9:
        raise RuntimeError("zero-initialized adapter does not reproduce frozen base")
    if args.expected_baseline_dsc is not None and abs(
        baseline["base_dsc_mean"] - args.expected_baseline_dsc
    ) > args.baseline_tolerance:
        raise RuntimeError("frozen base DSC differs from preregistered anchor")
    best_value = float(baseline["final_dsc_mean"])
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    best_path = output_dir / "boundary_safe_adapter_val_dice_best.pth"
    torch.save(
        checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=0,
            validation=baseline,
            args=args,
            source_checkpoint=source_checkpoint,
            source_sha256=source_sha256,
        ),
        best_path,
    )
    print(
        f"arm={args.arm_name} epoch=0 base={baseline['base_dsc_mean']:.6f} "
        f"final={baseline['final_dsc_mean']:.6f} subject={baseline['final_subject_dsc_mean']:.6f}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_losses: list[float] = []
        segmentation_losses: list[float] = []
        boundary_losses: list[float] = []
        train_dice: list[float] = []
        for batch in train_loader:
            image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            coarse = batch["coarse"].to(device=device, dtype=torch.float32, non_blocking=True)
            label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            image, coarse, label = augment_batch(image, coarse, label, max_shift=args.max_shift)
            coarse = corrupt_coarse_batch(
                coarse,
                max_shift=args.coarse_shift_max,
                probability=args.coarse_shift_probability,
            )
            target = boundary_target(label, radius=args.boundary_radius)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model.forward_with_aux(image, coarse)
                segmentation_loss = segmentation_loss_fn(output.logits, label)
                boundary_loss = boundary_loss_fn(output.gate_logits, target)
                loss = segmentation_loss + args.boundary_weight * boundary_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=5.0)
            optimizer.step()
            total_losses.append(float(loss.detach()))
            segmentation_losses.append(float(segmentation_loss.detach()))
            boundary_losses.append(float(boundary_loss.detach()))
            train_dice.extend(dice_per_case(output.logits.detach().float(), label).cpu().tolist())

        validation = validate(
            model,
            validation_loader,
            device,
            boundary_radius=args.boundary_radius,
        )
        value = float(validation["final_dsc_mean"])
        strict_new_high = value > best_value
        if strict_new_high:
            best_value = value
            best_epoch = epoch
            stale = 0
            torch.save(
                checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    validation=validation,
                    args=args,
                    source_checkpoint=source_checkpoint,
                    source_sha256=source_sha256,
                ),
                best_path,
            )
        else:
            stale += 1
        record = {
            "epoch": epoch,
            "train_total_loss": float(np.mean(total_losses)),
            "train_segmentation_loss": float(np.mean(segmentation_losses)),
            "train_boundary_loss": float(np.mean(boundary_losses)),
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
                "schema_version": SCHEMA_VERSION,
                "arm_name": args.arm_name,
                "boundary_weight": args.boundary_weight,
                "history": history,
                "best_epoch": best_epoch,
                "best_dsc": best_value,
                "test_access": False,
            },
        )
        print(
            f"arm={args.arm_name} epoch={epoch} total={record['train_total_loss']:.5f} "
            f"seg={record['train_segmentation_loss']:.5f} boundary={record['train_boundary_loss']:.5f} "
            f"train={record['train_dsc']:.5f} val={value:.6f} "
            f"subject={validation['final_subject_dsc_mean']:.6f} "
            f"boundary_dsc={validation['boundary_head_dsc_mean']:.6f} best={best_value:.6f}",
            flush=True,
        )
        if stale >= args.patience:
            break

    selected = baseline if best_epoch == 0 else history[best_epoch - 1]["validation"]
    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "arm_name": args.arm_name,
        "hypothesis": "sparse Dice-only 3D boundary supervision improves a safe adapter without the background suppression of error-map DiceCE",
        "task": {
            "input": "T1ce plus frozen SAM click-3 raw logits",
            "target": "WT={1,2,3}; raw label 4 excluded",
            "split": "972 train / 108 validation, subject-clean",
            "geometry": "GT-centered N4 128^3 refiner cache; development-only",
        },
        "initialization": {
            "path": str(source_checkpoint),
            "sha256": source_sha256,
            "base_frozen": True,
        },
        "baseline_validation": baseline,
        "selection": {
            "metric": "validation scan-macro DSC",
            "epoch": best_epoch,
            "value": best_value,
            "subject_macro_value": selected["final_subject_dsc_mean"],
            "boundary_head_dsc": selected["boundary_head_dsc_mean"],
            "delta_from_frozen_base": best_value - baseline["base_dsc_mean"],
        },
        "best_checkpoint": {"path": str(best_path), "sha256": sha256_file(best_path)},
        "train_cache": {
            "path": str(args.train_cache.resolve()),
            "index_sha256": sha256_file(args.train_cache / "index.json"),
            "cases_used": train_limit,
        },
        "validation_cache": {
            "path": str(args.validation_cache.resolve()),
            "index_sha256": sha256_file(args.validation_cache / "index.json"),
            "cases_used": validation_limit,
        },
        "hyperparameters": serializable_args(args),
        "history": history,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "boundary_module_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / "sammed3d/t1ce_wt_boundary_aux.py"
        ),
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
    parser.add_argument("--baseline-tolerance", type=float, default=1e-5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm-name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-init-filters", type=int, default=16)
    parser.add_argument("--base-dropout", type=float, default=0.1)
    parser.add_argument("--adapter-init-filters", type=int, default=8)
    parser.add_argument("--adapter-dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-shift", type=int, default=12)
    parser.add_argument("--coarse-shift-max", type=int, default=4)
    parser.add_argument("--coarse-shift-probability", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, required=True)
    parser.add_argument("--boundary-radius", type=int, default=1)
    parser.add_argument("--limit-train-cases", type=int, default=0)
    parser.add_argument("--limit-validation-cases", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
