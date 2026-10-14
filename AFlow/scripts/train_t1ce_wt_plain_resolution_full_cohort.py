#!/usr/bin/env python3
"""Train the 160^3 plain locator on all 756 train cases for 200 epochs.

This benchmark-oriented runner intentionally has no validation or evaluation
argument.  It saves fixed epoch168 and epoch200 checkpoints without reading an
outer selector, confirmation cohort, or test set.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
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
from sammed3d.t1ce_wt_variable_resolution import (  # noqa: E402
    VariableResolutionBrainCropDataset,
)
from scripts.train_t1ce_wt_capg3d import (  # noqa: E402
    augment_batch,
    case_view_metadata,
    dice_per_case,
    read_case_ids,
    sha256_file,
    subject_id,
    write_json_atomic,
)
from scripts.train_t1ce_wt_plain_long_schedule import learning_rate_for_epoch  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_resolution_full_cohort_training"
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_resolution_full_cohort_checkpoint"


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: AutomaticPromptLocator,
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    args: argparse.Namespace,
    data_role: dict[str, Any],
    sampling_contract: dict[str, Any],
) -> dict[str, Any]:
    value = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "epoch": epoch,
        "resolution": args.resolution,
        "learning_rate": learning_rate,
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
        "data_role": data_role,
        "sampling_contract": sampling_contract,
        "selection": "fixed checkpoint saved without validation access",
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    torch.save(value, path)
    return {"path": str(path), "sha256": sha256_file(path), "epoch": epoch}


@torch.inference_mode()
def reload_max_abs_difference(
    left: AutomaticPromptLocator,
    right: AutomaticPromptLocator,
    loader: DataLoader,
    device: torch.device,
) -> float:
    left.eval()
    right.eval()
    maximum = 0.0
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            left_logits = left(image).segmentation_logits
            right_logits = right(image).segmentation_logits
        maximum = max(
            maximum, float((left_logits.float() - right_logits.float()).abs().max())
        )
    return maximum


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.resolution != 160
        or args.epochs != 200
        or args.schedule_total_epochs != 200
        or args.constant_epochs != 100
        or args.batch_size != 8
    ):
        raise ValueError(
            "benchmark override requires resolution160, epoch200, schedule200/100, batch8"
        )
    output = args.output_dir.resolve()
    if any(
        token in str(output).lower()
        for token in ("selector", "confirmation", "sealed", "test270")
    ):
        raise ValueError("output path contains a forbidden role token")
    if sha256_file(args.train_case_ids) != args.expected_train_sha256:
        raise ValueError("756-case train manifest SHA-256 mismatch")
    case_ids = read_case_ids(args.train_case_ids)
    assert case_ids is not None
    train_data = VariableResolutionBrainCropDataset(
        args.cache,
        expected_resolution=args.resolution,
        expected_manifest_sha256=args.expected_train_sha256,
        case_ids=case_ids,
    )
    if len(train_data) != args.expected_train_cases:
        raise ValueError(
            f"full train view has {len(train_data)} cases, expected {args.expected_train_cases}"
        )
    if train_data.index.get("smoke") is not False:
        raise ValueError("full training cannot consume a smoke cache")
    if train_data.index["source_positive_voxels_sha256"] != (
        args.expected_source_positive_voxels_sha256
    ):
        raise ValueError("native WT volume hash mismatch")
    if any(
        train_data.index.get(key) is not False
        for key in ("outer_selector_access", "confirmation_access", "test_access")
    ):
        raise ValueError("cache declares forbidden data access")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full-cohort resolution training requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    output.mkdir(parents=True, exist_ok=False)
    subjects = {subject_id(case_id) for case_id in train_data.case_ids}
    data_role = {
        "train": case_view_metadata(args.train_case_ids),
        "cases": len(train_data),
        "subjects": len(subjects),
        "role": "historical_train_cohort_only",
    }
    sampling_weights = train_data.sampling_weights()
    sampling_contract = {
        "basis": "native-grid WT positive voxels before resolution resize",
        "weights_sha256": tensor_sha256(sampling_weights),
        "source_positive_voxels_sha256": train_data.index[
            "source_positive_voxels_sha256"
        ],
        "sampler": "WeightedRandomSampler(replacement=True,num_samples=756)",
        "generator_seed": args.seed,
    }
    write_json_atomic(
        output / "STARTED.json",
        {
            "schema": SCHEMA,
            "schema_version": 1,
            "status": "started_before_optimizer_step",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "goal": "benchmark-oriented full-train 160^3 epoch200 override",
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "cache": train_data.provenance(),
            "data_role": data_role,
            "sampling_contract": sampling_contract,
            "hyperparameters": serializable_args(args),
            "validation_access": False,
            "outer_selector_access": False,
            "confirmation_access": False,
            "test_access": False,
        },
    )

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            sampling_weights,
            num_samples=len(train_data),
            replacement=True,
            generator=generator,
        ),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    reload_loader = DataLoader(
        train_data,
        batch_size=args.reload_cases,
        shuffle=False,
        num_workers=0,
    )
    model = AutomaticPromptLocator(
        center_aware=False, init_filters=args.init_filters, dropout=args.dropout
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != args.expected_parameter_count:
        raise ValueError("unexpected model parameter count")
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
    checkpoints: dict[str, dict[str, Any]] = {}
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        lr = learning_rate_for_epoch(
            epoch,
            base_lr=args.base_lr,
            min_lr=args.min_lr,
            constant_epochs=args.constant_epochs,
            total_epochs=args.schedule_total_epochs,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        losses: list[float] = []
        train_dsc: list[float] = []
        for batch in loader:
            image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            image, label = augment_batch(image, label, max_shift=args.max_shift)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output_value = model(image)
                loss = loss_function(output_value.segmentation_logits, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            train_dsc.extend(
                dice_per_case(output_value.segmentation_logits.detach().float(), label)
                .cpu()
                .tolist()
            )
        record = {
            "epoch": epoch,
            "learning_rate": lr,
            "train_loss": float(np.mean(losses)),
            "train_dsc": float(np.mean(train_dsc)),
            "optimizer_steps": len(loader),
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        if epoch in (168, 200):
            checkpoint_path = output / f"plain_160_full756_fixed_epoch{epoch}.pth"
            checkpoints[str(epoch)] = save_checkpoint(
                checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                learning_rate=lr,
                args=args,
                data_role=data_role,
                sampling_contract=sampling_contract,
            )
        write_json_atomic(
            output / "progress.json",
            {"history": history, "checkpoints": checkpoints},
        )
        print(
            f"resolution=160 full756 epoch={epoch}/{args.epochs} lr={lr:.8g} "
            f"loss={record['train_loss']:.6f} train={record['train_dsc']:.6f}",
            flush=True,
        )

    final_path = Path(checkpoints["200"]["path"])
    reloaded = AutomaticPromptLocator(
        center_aware=False, init_filters=args.init_filters, dropout=args.dropout
    ).to(device)
    final_value = torch.load(final_path, map_location=device, weights_only=False)
    reloaded.load_state_dict(final_value["model_state_dict"], strict=True)
    reload_difference = reload_max_abs_difference(
        model, reloaded, reload_loader, device
    )
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "goal": "benchmark-oriented full-train 160^3 epoch200 override",
        "resolution": 160,
        "fixed_final_epoch": 200,
        "checkpoints": checkpoints,
        "final_reload_max_abs_difference": reload_difference,
        "cache": train_data.provenance(),
        "data_role": data_role,
        "sampling_contract": sampling_contract,
        "parameter_count": parameter_count,
        "history": history,
        "hyperparameters": serializable_args(args),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    write_json_atomic(output / "TRAIN_DONE.json", report)
    print(
        f"TRAIN_DONE epoch200={checkpoints['200']['sha256']} "
        f"epoch168={checkpoints['168']['sha256']} reload_max={reload_difference:.9g}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=160)
    parser.add_argument("--train-case-ids", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-source-positive-voxels-sha256", required=True)
    parser.add_argument("--expected-train-cases", type=int, default=756)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--schedule-total-epochs", type=int, default=200)
    parser.add_argument("--constant-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--reload-cases", type=int, default=8)
    parser.add_argument("--init-filters", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--base-lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-shift", type=int, default=8)
    parser.add_argument("--ce-weight", type=float, default=0.1)
    parser.add_argument("--expected-parameter-count", type=int, default=4_700_897)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
