#!/usr/bin/env python3
"""Train the error-localized prompt-gated T1ce/WT 3D residual refiner.

This development-only runner accepts only train/validation refiner caches whose
indices explicitly state ``test_access: false``.  It bridges a frozen legacy
residual-refiner checkpoint with exact epoch-zero prediction parity, then trains
with DiceCE plus an auxiliary coarse-error gate target.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from monai.losses import DiceCELoss

from sammed3d.t1ce_wt_error_gated_refiner import (
    ErrorGatedResidualRefiner,
    bridge_residual_refiner_state_dict,
    correction_target,
)
from scripts.train_sammed3d_t1ce_wt_residual_refiner import (
    RefinerDataset,
    augment_batch,
    dice_per_case,
    sha256_file,
    write_json_atomic,
)
from scripts.train_t1ce_wt_residual_refiner_graymean_robust import (
    corrupt_coarse_batch,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_error_gated_refiner_training"
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_error_gated_refiner_checkpoint"
SOURCE_CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_residual_refiner_checkpoint"
SCHEMA_VERSION = 1


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def subject_id(case_id: str) -> str:
    fields = case_id.rsplit("-", 1)
    if len(fields) != 2 or not fields[1].isdigit():
        raise ValueError(f"unexpected BraTS case id: {case_id}")
    return fields[0]


def subject_macro(case_ids: list[str], values: list[float]) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for case_id, value in zip(case_ids, values, strict=True):
        grouped[subject_id(case_id)].append(value)
    return float(np.mean([np.mean(group) for group in grouped.values()]))


def limited_dataset(dataset: RefinerDataset, limit: int) -> RefinerDataset | Subset:
    if limit < 0:
        raise ValueError("case limit must be non-negative")
    if not limit:
        return dataset
    if limit > len(dataset):
        raise ValueError("case limit exceeds dataset")
    return Subset(dataset, range(limit))


@torch.inference_mode()
def validate(
    model: ErrorGatedResidualRefiner,
    loader: DataLoader,
    device: torch.device,
    *,
    gate_dilation_radius: int,
) -> dict[str, Any]:
    model.eval()
    refined: list[float] = []
    coarse_values: list[float] = []
    gate_values: list[float] = []
    gate_target_fractions: list[float] = []
    residual_scales: list[float] = []
    case_ids: list[str] = []
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        coarse = batch["coarse"].to(device=device, dtype=torch.float32, non_blocking=True)
        label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model.forward_with_aux(image, coarse)
        gate_target = correction_target(
            coarse,
            label,
            dilation_radius=gate_dilation_radius,
        )
        logits = output.logits.float()
        refined.extend(dice_per_case(logits, label).cpu().tolist())
        coarse_values.extend(dice_per_case(coarse, label).cpu().tolist())
        gate_values.extend(dice_per_case(output.gate_logits.float(), gate_target).cpu().tolist())
        gate_target_fractions.extend(gate_target.flatten(1).mean(1).cpu().tolist())
        residual_scales.extend(output.residual_scale.float().flatten(1).mean(1).cpu().tolist())
        case_ids.extend(list(batch["case_id"]))
    refined_array = np.asarray(refined)
    coarse_array = np.asarray(coarse_values)
    return {
        "cases": len(refined),
        "subjects": len({subject_id(case_id) for case_id in case_ids}),
        "coarse_dsc_mean": float(np.mean(coarse_array)),
        "refined_dsc_mean": float(np.mean(refined_array)),
        "refined_subject_dsc_mean": subject_macro(case_ids, refined),
        "delta": float(np.mean(refined_array - coarse_array)),
        "improved_cases": int(np.sum(refined_array > coarse_array)),
        "worsened_cases": int(np.sum(refined_array < coarse_array)),
        "gate_dsc_mean": float(np.mean(gate_values)),
        "gate_target_fraction_mean": float(np.mean(gate_target_fractions)),
        "residual_scale_mean": float(np.mean(residual_scales)),
        "per_case": [
            {
                "case_id": case_id,
                "coarse_dsc": coarse_dsc,
                "refined_dsc": refined_dsc,
                "gate_dsc": gate_dsc,
                "gate_target_fraction": gate_fraction,
                "residual_scale_mean": residual_scale,
            }
            for case_id, coarse_dsc, refined_dsc, gate_dsc, gate_fraction, residual_scale in zip(
                case_ids,
                coarse_values,
                refined,
                gate_values,
                gate_target_fractions,
                residual_scales,
                strict=True,
            )
        ],
    }


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation: dict[str, Any],
    args: argparse.Namespace,
    source_checkpoint: Path,
    source_sha256: str,
    source_epoch: int,
    bridge: dict[str, object],
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
            "source_epoch": source_epoch,
            "optimizer_state_restored": False,
            "bridge": bridge,
        },
        "architecture": {
            "name": "Error-Localized Prompt-Gated 3D Residual Refiner (ELPGR-3D A2)",
            "backbone": "MONAI SegResNet",
            "init_filters": args.init_filters,
            "dropout": args.dropout,
            "input_channels": [
                "T1ce",
                "tanh(frozen_SAM_logit/5)",
                "4*p*(1-p) coarse uncertainty",
                "3x3x3 coarse-probability morphological gradient",
            ],
            "heads": ["residual", "correction_gate"],
            "output": "coarse_logit + (1 + tanh(gate_logit)) * residual",
            "inference_label_input": False,
        },
        "objective": {
            "segmentation": "MONAI DiceCELoss(sigmoid=True, lambda_dice=1, lambda_ce=1)",
            "gate": "MONAI DiceCELoss against dilated XOR(coarse mask, GT)",
            "gate_weight": args.gate_weight,
            "gate_dilation_radius": args.gate_dilation_radius,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.gate_weight < 0:
        raise ValueError("gate_weight must be non-negative")
    if args.gate_dilation_radius < 0:
        raise ValueError("gate_dilation_radius must be non-negative")
    if args.coarse_shift_max < 0 or not 0 <= args.coarse_shift_probability <= 1:
        raise ValueError("invalid coarse corruption")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("error-gated refiner training requires CUDA")
    torch.cuda.set_device(device)

    train_source = RefinerDataset(args.train_cache)
    validation_source = RefinerDataset(args.validation_cache)
    if train_source.index["split"] != "train" or validation_source.index["split"] != "validation":
        raise ValueError("train/validation refiner cache split mismatch")
    if train_source.index.get("test_access") is not False or validation_source.index.get("test_access") is not False:
        raise ValueError("test-accessible cache is forbidden")
    if train_source.index["model"]["checkpoint_sha256"] != validation_source.index["model"]["checkpoint_sha256"]:
        raise ValueError("train/validation coarse checkpoint mismatch")
    train_data = limited_dataset(train_source, args.limit_train_cases)
    validation_data = limited_dataset(validation_source, args.limit_validation_cases)

    source_checkpoint = args.init_checkpoint.resolve()
    source_sha256 = sha256_file(source_checkpoint)
    if args.expected_init_sha256 and source_sha256 != args.expected_init_sha256:
        raise RuntimeError("initial residual-refiner checkpoint hash mismatch")
    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if source.get("schema") != SOURCE_CHECKPOINT_SCHEMA:
        raise ValueError("unexpected source residual-refiner checkpoint schema")
    source_architecture = source.get("architecture", {})
    if int(source_architecture.get("init_filters", -1)) != args.init_filters:
        raise ValueError("init_filters differs from source checkpoint")
    if float(source_architecture.get("dropout", -1)) != args.dropout:
        raise ValueError("dropout differs from source checkpoint")
    source_epoch = int(source["epoch"])

    generator = torch.Generator().manual_seed(args.seed)
    weights = train_source.sampling_weights()
    if args.limit_train_cases:
        weights = weights[: args.limit_train_cases]
    sampler = WeightedRandomSampler(
        weights,
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
    validation_workers = max(0, args.num_workers // 2)
    validation_loader = DataLoader(
        validation_data,
        batch_size=1,
        shuffle=False,
        num_workers=validation_workers,
        pin_memory=True,
        persistent_workers=validation_workers > 0,
    )

    model = ErrorGatedResidualRefiner(args.init_filters, args.dropout)
    bridge = bridge_residual_refiner_state_dict(model, source["model_state_dict"])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    segmentation_loss_fn = DiceCELoss(
        sigmoid=True,
        squared_pred=False,
        lambda_dice=1.0,
        lambda_ce=1.0,
    )
    gate_loss_fn = DiceCELoss(
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
        gate_dilation_radius=args.gate_dilation_radius,
    )
    if args.expected_baseline_dsc is not None and abs(
        baseline["refined_dsc_mean"] - args.expected_baseline_dsc
    ) > args.baseline_tolerance:
        raise RuntimeError(
            "bridged validation DSC differs from preregistered baseline: "
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
            epoch=0,
            validation=baseline,
            args=args,
            source_checkpoint=source_checkpoint,
            source_sha256=source_sha256,
            source_epoch=source_epoch,
            bridge=bridge,
        ),
        best_path,
    )
    print(
        f"arm={args.arm_name} epoch=0 coarse={baseline['coarse_dsc_mean']:.6f} "
        f"refined={baseline['refined_dsc_mean']:.6f} "
        f"subject={baseline['refined_subject_dsc_mean']:.6f}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_losses: list[float] = []
        segmentation_losses: list[float] = []
        gate_losses: list[float] = []
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
            gate_target = correction_target(
                coarse,
                label,
                dilation_radius=args.gate_dilation_radius,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model.forward_with_aux(image, coarse)
                segmentation_loss = segmentation_loss_fn(output.logits, label)
                gate_loss = gate_loss_fn(output.gate_logits, gate_target)
                loss = segmentation_loss + args.gate_weight * gate_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_losses.append(float(loss.detach()))
            segmentation_losses.append(float(segmentation_loss.detach()))
            gate_losses.append(float(gate_loss.detach()))
            train_dice.extend(dice_per_case(output.logits.detach().float(), label).cpu().tolist())

        validation = validate(
            model,
            validation_loader,
            device,
            gate_dilation_radius=args.gate_dilation_radius,
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
                    epoch=epoch,
                    validation=validation,
                    args=args,
                    source_checkpoint=source_checkpoint,
                    source_sha256=source_sha256,
                    source_epoch=source_epoch,
                    bridge=bridge,
                ),
                best_path,
            )
        else:
            stale += 1
        record = {
            "epoch": epoch,
            "source_plus_continuation_epoch": source_epoch + epoch,
            "train_total_loss": float(np.mean(total_losses)),
            "train_segmentation_loss": float(np.mean(segmentation_losses)),
            "train_gate_loss": float(np.mean(gate_losses)),
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
                "baseline_dsc": baseline["refined_dsc_mean"],
                "history": history,
                "best_epoch": best_epoch,
                "best_dsc": best_value,
                "test_access": False,
            },
        )
        print(
            f"arm={args.arm_name} epoch={epoch} total={record['train_total_loss']:.5f} "
            f"seg={record['train_segmentation_loss']:.5f} gate={record['train_gate_loss']:.5f} "
            f"train={record['train_dsc']:.5f} val={value:.6f} "
            f"subject={validation['refined_subject_dsc_mean']:.6f} best={best_value:.6f}",
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
        "hypothesis": (
            "label-free coarse uncertainty/boundary prompts plus an error-supervised "
            "correction gate reduce harmful global residual rewriting"
        ),
        "task": {
            "input": "T1ce plus frozen SAM click-3 raw logits",
            "target": "WT={1,2,3}; raw label 4 excluded",
            "split": "972 train / 108 validation, subject-clean",
            "geometry": "GT-centered N4 128^3 refiner cache; development-only",
            "known_limitation": "the 108-case validation has prior development exposure",
        },
        "initialization": {
            "path": str(source_checkpoint),
            "sha256": source_sha256,
            "source_epoch": source_epoch,
            "bridge": bridge,
        },
        "baseline_validation": baseline,
        "selection": {
            "metric": "validation scan-macro DSC",
            "epoch": best_epoch,
            "source_plus_continuation_epoch": source_epoch + best_epoch,
            "value": best_value,
            "subject_macro_value": (
                baseline["refined_subject_dsc_mean"]
                if best_epoch == 0
                else history[best_epoch - 1]["validation"]["refined_subject_dsc_mean"]
            ),
            "delta_from_bridged_initialization": best_value - baseline["refined_dsc_mean"],
        },
        "best_checkpoint": {"path": str(best_path), "sha256": sha256_file(best_path)},
        "coarse_checkpoint_sha256": train_source.index["model"]["checkpoint_sha256"],
        "train_cache": {
            "path": str(args.train_cache.resolve()),
            "index_sha256": sha256_file(args.train_cache / "index.json"),
            "cases_used": len(train_data),
        },
        "validation_cache": {
            "path": str(args.validation_cache.resolve()),
            "index_sha256": sha256_file(args.validation_cache / "index.json"),
            "cases_used": len(validation_data),
        },
        "hyperparameters": serializable_args(args),
        "history": history,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "architecture_module_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / "sammed3d/t1ce_wt_error_gated_refiner.py"
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
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--init-filters", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-shift", type=int, default=12)
    parser.add_argument("--coarse-shift-max", type=int, default=4)
    parser.add_argument("--coarse-shift-probability", type=float, default=1.0)
    parser.add_argument("--gate-weight", type=float, default=0.1)
    parser.add_argument("--gate-dilation-radius", type=int, default=2)
    parser.add_argument("--limit-train-cases", type=int, default=0)
    parser.add_argument("--limit-validation-cases", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
