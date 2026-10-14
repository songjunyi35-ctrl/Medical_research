#!/usr/bin/env python3
"""Train one plain locator trajectory with frozen pre/post-100 LR windows."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from monai.losses import DiceCELoss
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator  # noqa: E402
from scripts.train_t1ce_wt_capg3d import (  # noqa: E402
    BrainCropDataset,
    augment_batch,
    case_view_metadata,
    dice_per_case,
    read_case_ids,
    sha256_file,
    subject_id,
    validate,
    write_json_atomic,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_long_schedule_training"
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_long_schedule_checkpoint"
FORBIDDEN_ROLE_TOKENS = ("selector", "confirmation", "sealed", "test270")


def assert_inner_only_path(path: Path) -> None:
    lowered = str(path.resolve()).lower()
    if any(token in lowered for token in FORBIDDEN_ROLE_TOKENS):
        raise ValueError(f"forbidden data role in path: {path}")


def learning_rate_for_epoch(
    epoch: int,
    *,
    base_lr: float,
    min_lr: float,
    constant_epochs: int,
    total_epochs: int,
) -> float:
    if not 1 <= epoch <= total_epochs:
        raise ValueError("epoch outside frozen schedule")
    if not 1 <= constant_epochs < total_epochs:
        raise ValueError("constant_epochs must be inside total_epochs")
    if not 0.0 < min_lr <= base_lr:
        raise ValueError("expected 0 < min_lr <= base_lr")
    if epoch <= constant_epochs:
        return float(base_lr)
    denominator = total_epochs - constant_epochs - 1
    progress = 1.0 if denominator <= 0 else (epoch - constant_epochs - 1) / denominator
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr + (base_lr - min_lr) * cosine)


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.smoke and (args.epochs != 200 or args.constant_epochs != 100):
        raise ValueError("formal protocol requires 200 total / 100 constant epochs")
    assert_inner_only_path(args.fit_case_ids)
    assert_inner_only_path(args.fresh_tune_case_ids)
    if sha256_file(args.fit_case_ids) != args.expected_fit_sha256:
        raise ValueError("fit manifest SHA-256 mismatch")
    if sha256_file(args.fresh_tune_case_ids) != args.expected_fresh_tune_sha256:
        raise ValueError("fresh-tune manifest SHA-256 mismatch")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal long-schedule training requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    fit_data = BrainCropDataset(
        args.cache,
        expected_split="train",
        limit=args.limit_fit_cases,
        case_ids=read_case_ids(args.fit_case_ids),
    )
    tune_data = BrainCropDataset(
        args.cache,
        expected_split="train",
        limit=args.limit_fresh_tune_cases,
        case_ids=read_case_ids(args.fresh_tune_case_ids),
    )
    if len(fit_data) != args.expected_fit_cases:
        raise ValueError("unexpected fit case count")
    if len(tune_data) != args.expected_fresh_tune_cases:
        raise ValueError("unexpected fresh-tune case count")
    if set(fit_data.case_ids) & set(tune_data.case_ids):
        raise ValueError("fit/fresh-tune case leakage")
    fit_subjects = {subject_id(case_id) for case_id in fit_data.case_ids}
    tune_subjects = {subject_id(case_id) for case_id in tune_data.case_ids}
    if fit_subjects & tune_subjects:
        raise ValueError("fit/fresh-tune subject leakage")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    data_roles = {
        "fit": case_view_metadata(args.fit_case_ids),
        "fresh_tune": case_view_metadata(args.fresh_tune_case_ids),
        "fit_cases": len(fit_data),
        "fit_subjects": len(fit_subjects),
        "fresh_tune_cases": len(tune_data),
        "fresh_tune_subjects": len(tune_subjects),
        "case_overlap": 0,
        "subject_overlap": 0,
    }
    write_json_atomic(
        output_dir / "STARTED.json",
        {
            "schema": SCHEMA,
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "started_before_optimizer_step",
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "data_roles": data_roles,
            "hyperparameters": serializable_args(args),
            "outer_selector_access": False,
            "confirmation_access": False,
            "test_access": False,
        },
    )

    generator = torch.Generator().manual_seed(args.seed)
    fit_loader = DataLoader(
        fit_data,
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            fit_data.sampling_weights(),
            num_samples=len(fit_data),
            replacement=True,
            generator=generator,
        ),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    tune_loader = DataLoader(
        tune_data,
        batch_size=args.validation_batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    if len(fit_loader) == 0:
        raise RuntimeError("fit loader has zero optimizer steps")

    model = AutomaticPromptLocator(
        center_aware=False,
        init_filters=args.init_filters,
        dropout=args.dropout,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != args.expected_parameter_count:
        raise ValueError(f"parameter count {parameter_count} does not match protocol")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay
    )
    loss_function = DiceCELoss(
        sigmoid=True,
        squared_pred=False,
        lambda_dice=1.0,
        lambda_ce=args.ce_weight,
    )

    history: list[dict[str, Any]] = []
    best: dict[str, dict[str, Any]] = {
        "pre100": {"value": float("-inf"), "epoch": 0, "path": output_dir / "pre100_best.pth"},
        "post100": {"value": float("-inf"), "epoch": 0, "path": output_dir / "post100_best.pth"},
    }
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        lr = learning_rate_for_epoch(
            epoch,
            base_lr=args.base_lr,
            min_lr=args.min_lr,
            constant_epochs=args.constant_epochs,
            total_epochs=args.epochs,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        losses: list[float] = []
        train_dsc: list[float] = []
        for batch in fit_loader:
            image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            image, label = augment_batch(image, label, max_shift=args.max_shift)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(image)
                loss = loss_function(output.segmentation_logits, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            train_dsc.extend(
                dice_per_case(output.segmentation_logits.detach().float(), label).cpu().tolist()
            )

        validation = validate(model, tune_loader, device)
        value = float(validation["locator_dsc_mean"])
        window = "pre100" if epoch <= args.constant_epochs else "post100"
        strict_new_high = value > float(best[window]["value"])
        if strict_new_high:
            best[window]["value"] = value
            best[window]["epoch"] = epoch
            checkpoint = {
                "schema": CHECKPOINT_SCHEMA,
                "schema_version": 1,
                "epoch": epoch,
                "window": window,
                "arm": "plain",
                "learning_rate": lr,
                "validation": validation,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "architecture": {
                    "class": "AutomaticPromptLocator",
                    "backbone": "MONAI SegResNet",
                    "center_aware": False,
                    "init_filters": args.init_filters,
                    "dropout": args.dropout,
                    "input_channels": ["T1ce"],
                    "outputs": ["WT segmentation logit"],
                },
                "hyperparameters": serializable_args(args),
                "data_roles": data_roles,
                "outer_selector_access": False,
                "confirmation_access": False,
                "test_access": False,
            }
            torch.save(checkpoint, best[window]["path"])

        record = {
            "epoch": epoch,
            "window": window,
            "learning_rate": lr,
            "train_loss": float(np.mean(losses)),
            "train_dsc": float(np.mean(train_dsc)),
            "validation": validation,
            "strict_new_high_in_window": strict_new_high,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        write_json_atomic(
            output_dir / "progress.json",
            {
                "history": history,
                "best": {
                    name: {"epoch": item["epoch"], "value": item["value"]}
                    for name, item in best.items()
                },
            },
        )
        print(
            f"epoch={epoch} window={window} lr={lr:.8g} "
            f"loss={record['train_loss']:.5f} train={record['train_dsc']:.5f} "
            f"tune={value:.6f} subject={validation['locator_subject_dsc_mean']:.6f} "
            f"point={validation['automatic_point_hit_rate']:.4f} "
            f"best={best[window]['value']:.6f}@{best[window]['epoch']}",
            flush=True,
        )

    selections: dict[str, dict[str, Any]] = {}
    for window, item in best.items():
        path = Path(item["path"])
        selections[window] = {
            "epoch": int(item["epoch"]),
            "value": float(item["value"]),
            "checkpoint": {"path": str(path), "sha256": sha256_file(path)},
        }
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
        "fixed_logit_threshold": 0.0,
        "schedule": {
            "epochs": args.epochs,
            "constant_epochs": args.constant_epochs,
            "base_lr": args.base_lr,
            "min_lr": args.min_lr,
            "post100": "cosine inclusive endpoints",
        },
        "selections": selections,
        "data_roles": data_roles,
        "parameter_count": parameter_count,
        "hyperparameters": serializable_args(args),
        "history": history,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
    }
    write_json_atomic(output_dir / "TRAIN_DONE.json", report)
    print(
        f"TRAIN_DONE pre={selections['pre100']['value']:.9f}@{selections['pre100']['epoch']} "
        f"post={selections['post100']['value']:.9f}@{selections['post100']['epoch']}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--fit-case-ids", type=Path, required=True)
    parser.add_argument("--fresh-tune-case-ids", type=Path, required=True)
    parser.add_argument("--expected-fit-sha256", required=True)
    parser.add_argument("--expected-fresh-tune-sha256", required=True)
    parser.add_argument("--expected-fit-cases", type=int, default=540)
    parser.add_argument("--expected-fresh-tune-cases", type=int, default=108)
    parser.add_argument("--expected-parameter-count", type=int, default=4_700_897)
    parser.add_argument("--limit-fit-cases", type=int, default=None)
    parser.add_argument("--limit-fresh-tune-cases", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--constant-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--init-filters", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--base-lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-shift", type=int, default=8)
    parser.add_argument("--ce-weight", type=float, default=0.1)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
