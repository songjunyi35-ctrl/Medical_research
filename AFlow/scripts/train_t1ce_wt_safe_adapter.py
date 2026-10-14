#!/usr/bin/env python3
"""Train a zero-initialized adapter on a frozen robust T1ce/WT refiner."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from monai.losses import DiceCELoss

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


SCHEMA = "aflow.sammed3d.t1ce_wt_safe_adapter_training"
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_safe_adapter_checkpoint"
SOURCE_CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_residual_refiner_graymean_checkpoint"
SCHEMA_VERSION = 1


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


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


@torch.inference_mode()
def validate(
    model: FrozenBaseSafeAdapter,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    final_values: list[float] = []
    base_values: list[float] = []
    coarse_values: list[float] = []
    scales: list[float] = []
    residual_magnitudes: list[float] = []
    case_ids: list[str] = []
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        coarse = batch["coarse"].to(device=device, dtype=torch.float32, non_blocking=True)
        label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model.forward_with_aux(image, coarse)
        final_values.extend(dice_per_case(output.logits.float(), label).cpu().tolist())
        base_values.extend(dice_per_case(output.base_logits.float(), label).cpu().tolist())
        coarse_values.extend(dice_per_case(coarse, label).cpu().tolist())
        scales.extend(output.residual_scale.float().flatten(1).mean(1).cpu().tolist())
        residual_magnitudes.extend(output.residual.float().abs().flatten(1).mean(1).cpu().tolist())
        case_ids.extend(list(batch["case_id"]))
    final_array = np.asarray(final_values)
    base_array = np.asarray(base_values)
    return {
        "cases": len(case_ids),
        "subjects": len({subject_id(case_id) for case_id in case_ids}),
        "coarse_dsc_mean": float(np.mean(coarse_values)),
        "base_dsc_mean": float(np.mean(base_array)),
        "base_subject_dsc_mean": subject_macro(case_ids, base_values),
        "final_dsc_mean": float(np.mean(final_array)),
        "final_subject_dsc_mean": subject_macro(case_ids, final_values),
        "delta_from_base": float(np.mean(final_array - base_array)),
        "improved_vs_base_cases": int(np.sum(final_array > base_array)),
        "worsened_vs_base_cases": int(np.sum(final_array < base_array)),
        "residual_scale_mean": float(np.mean(scales)),
        "adapter_residual_abs_mean": float(np.mean(residual_magnitudes)),
        "per_case": [
            {
                "case_id": case_id,
                "coarse_dsc": coarse,
                "base_dsc": base,
                "final_dsc": final,
                "delta_from_base": final - base,
            }
            for case_id, coarse, base, final in zip(
                case_ids, coarse_values, base_values, final_values, strict=True
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
            "name": "SAGE-3D frozen-base safe adapter",
            "mode": args.mode,
            "base": "frozen robust ResidualRefiner",
            "adapter": "MONAI SegResNet",
            "adapter_init_filters": args.adapter_init_filters,
            "adapter_dropout": args.adapter_dropout,
            "input_channels": [
                "T1ce",
                "tanh(coarse_logit/5)",
                "tanh(frozen_base_logit/5)",
                "frozen_base_uncertainty",
                "abs(base_probability-coarse_probability)",
            ],
            "heads": ["adapter_residual", "implicit_gate"],
            "plain_output": "base_logit + adapter_residual",
            "gated_output": "base_logit + 2*sigmoid(gate)*adapter_residual",
            "inference_label_input": False,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode not in FrozenBaseSafeAdapter.MODES:
        raise ValueError(f"mode must be one of {sorted(FrozenBaseSafeAdapter.MODES)}")
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
        raise RuntimeError("safe-adapter training requires CUDA")
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
    architecture = source.get("architecture", {})
    if int(architecture.get("init_filters", -1)) != args.base_init_filters:
        raise ValueError("base init_filters mismatch")
    if float(architecture.get("dropout", -1)) != args.base_dropout:
        raise ValueError("base dropout mismatch")

    train_limit = args.limit_train_cases or len(train_source)
    validation_limit = args.limit_validation_cases or len(validation_source)
    if not 0 < train_limit <= len(train_source) or not 0 < validation_limit <= len(validation_source):
        raise ValueError("invalid case limit")
    train_data = train_source if train_limit == len(train_source) else Subset(train_source, range(train_limit))
    validation_data = validation_source if validation_limit == len(validation_source) else Subset(validation_source, range(validation_limit))
    weights = train_source.sampling_weights()[:train_limit]
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        weights,
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
        mode=args.mode,
        init_filters=args.adapter_init_filters,
        dropout=args.adapter_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = DiceCELoss(
        sigmoid=True,
        squared_pred=False,
        lambda_dice=1.0,
        lambda_ce=1.0,
    )

    started = time.time()
    baseline = validate(model, validation_loader, device)
    if abs(baseline["final_dsc_mean"] - baseline["base_dsc_mean"]) > 1e-9:
        raise RuntimeError("zero-initialized adapter does not reproduce frozen base")
    if args.expected_baseline_dsc is not None and abs(
        baseline["base_dsc_mean"] - args.expected_baseline_dsc
    ) > args.baseline_tolerance:
        raise RuntimeError(
            "frozen base DSC differs from preregistered baseline: "
            f"{baseline['base_dsc_mean']:.12f} vs {args.expected_baseline_dsc:.12f}"
        )
    best_value = float(baseline["final_dsc_mean"])
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    best_path = output_dir / "safe_adapter_val_dice_best.pth"
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
        f"arm={args.arm_name} mode={args.mode} epoch=0 base={baseline['base_dsc_mean']:.6f} "
        f"final={baseline['final_dsc_mean']:.6f} subject={baseline['final_subject_dsc_mean']:.6f}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
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
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model.forward_with_aux(image, coarse)
                loss = loss_fn(output.logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            train_dice.extend(dice_per_case(output.logits.detach().float(), label).cpu().tolist())

        validation = validate(model, validation_loader, device)
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
            "train_loss": float(np.mean(losses)),
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
                "mode": args.mode,
                "history": history,
                "best_epoch": best_epoch,
                "best_dsc": best_value,
                "test_access": False,
            },
        )
        print(
            f"arm={args.arm_name} mode={args.mode} epoch={epoch} loss={record['train_loss']:.5f} "
            f"train={record['train_dsc']:.5f} val={value:.6f} "
            f"subject={validation['final_subject_dsc_mean']:.6f} "
            f"delta={validation['delta_from_base']:+.6f} best={best_value:.6f}",
            flush=True,
        )
        if stale >= args.patience:
            break

    selected_validation = baseline if best_epoch == 0 else history[best_epoch - 1]["validation"]
    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "arm_name": args.arm_name,
        "mode": args.mode,
        "hypothesis": (
            "a frozen strong base plus a zero-initialized uncertainty-conditioned adapter "
            "can add corrections without catastrophically rewriting the base"
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
            "base_frozen": True,
            "optimizer_state_restored": False,
        },
        "baseline_validation": baseline,
        "selection": {
            "metric": "validation scan-macro DSC",
            "epoch": best_epoch,
            "value": best_value,
            "subject_macro_value": selected_validation["final_subject_dsc_mean"],
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
        "parameter_counts": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        },
        "hyperparameters": serializable_args(args),
        "history": history,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "architecture_module_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / "sammed3d/t1ce_wt_safe_adapter.py"
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
    parser.add_argument("--mode", choices=sorted(FrozenBaseSafeAdapter.MODES), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260816)
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
    parser.add_argument("--limit-train-cases", type=int, default=0)
    parser.add_argument("--limit-validation-cases", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
