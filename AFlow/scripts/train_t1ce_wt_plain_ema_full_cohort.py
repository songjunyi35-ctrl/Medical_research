#!/usr/bin/env python3
"""Train full-756 true-128^3 plain locator with fixed late-cosine EMA."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from monai.losses import DiceCELoss
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler


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


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_ema_full_cohort_training"
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_ema_checkpoint"


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


@torch.no_grad()
def update_ema(ema: nn.Module, raw: nn.Module, decay: float) -> None:
    if not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must satisfy 0 <= decay < 1")
    ema_parameters = dict(ema.named_parameters())
    raw_parameters = dict(raw.named_parameters())
    if ema_parameters.keys() != raw_parameters.keys():
        raise ValueError("EMA/raw parameter names differ")
    for name, ema_value in ema_parameters.items():
        raw_value = raw_parameters[name].detach()
        ema_value.mul_(decay).add_(raw_value, alpha=1.0 - decay)
    ema_buffers = dict(ema.named_buffers())
    raw_buffers = dict(raw.named_buffers())
    if ema_buffers.keys() != raw_buffers.keys():
        raise ValueError("EMA/raw buffer names differ")
    for name, ema_value in ema_buffers.items():
        raw_value = raw_buffers[name].detach()
        if ema_value.is_floating_point():
            ema_value.mul_(decay).add_(raw_value, alpha=1.0 - decay)
        else:
            ema_value.copy_(raw_value)


def save_checkpoint(
    path: Path,
    *,
    model: AutomaticPromptLocator,
    kind: str,
    epoch: int,
    learning_rate: float,
    args: argparse.Namespace,
    data_role: dict[str, Any],
    sampling_contract: dict[str, Any],
    ema_updates: int,
) -> dict[str, Any]:
    if kind not in {"raw", "ema"}:
        raise ValueError("checkpoint kind must be raw or ema")
    value = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "kind": kind,
        "epoch": epoch,
        "resolution": 128,
        "learning_rate": learning_rate,
        "model_state_dict": model.state_dict(),
        "architecture": {
            "class": "AutomaticPromptLocator",
            "backbone": "MONAI SegResNet",
            "center_aware": False,
            "init_filters": args.init_filters,
            "dropout": args.dropout,
            "input_channels": ["T1ce"],
            "outputs": ["WT segmentation logit"]
        },
        "ema": {
            "enabled": kind == "ema",
            "decay": args.ema_decay,
            "start_after_epoch": args.ema_start_after_epoch,
            "updates": ema_updates
        },
        "hyperparameters": serializable_args(args),
        "data_role": data_role,
        "sampling_contract": sampling_contract,
        "selection": "fixed checkpoint saved without validation access",
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False
    }
    torch.save(value, path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "kind": kind,
        "epoch": epoch,
        "ema_updates": ema_updates
    }


@torch.inference_mode()
def max_abs_difference(
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
    frozen = {
        "resolution": 128,
        "epochs": 200,
        "schedule_total_epochs": 200,
        "constant_epochs": 100,
        "batch_size": 8,
        "ema_start_after_epoch": 100,
        "ema_decay": 0.999,
        "base_lr": 2e-4,
        "min_lr": 1e-5,
        "weight_decay": 1e-5,
        "max_shift": 8,
        "ce_weight": 0.1
    }
    for name, expected in frozen.items():
        if getattr(args, name) != expected:
            raise ValueError(f"formal EMA protocol requires {name}={expected}")
    if sha256_file(args.train_case_ids) != args.expected_train_sha256:
        raise ValueError("full train manifest SHA-256 mismatch")
    case_ids = read_case_ids(args.train_case_ids)
    assert case_ids is not None
    train_data = VariableResolutionBrainCropDataset(
        args.cache,
        expected_resolution=128,
        expected_manifest_sha256=args.expected_train_sha256,
        case_ids=case_ids
    )
    if len(train_data) != args.expected_train_cases:
        raise ValueError("unexpected full train case count")
    if train_data.index.get("smoke") is not False:
        raise ValueError("formal EMA training cannot consume a smoke cache")
    if train_data.index["source_positive_voxels_sha256"] != (
        args.expected_source_positive_voxels_sha256
    ):
        raise ValueError("native WT volume hash mismatch")
    if any(
        train_data.index.get(key) is not False
        for key in ("outer_selector_access", "confirmation_access", "test_access")
    ):
        raise ValueError("train cache declares forbidden access")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full-cohort EMA training requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    output = args.output_dir.resolve()
    if any(
        token in str(output).lower()
        for token in ("selector", "confirmation", "sealed", "test270")
    ):
        raise ValueError("output path contains a forbidden role token")
    output.mkdir(parents=True, exist_ok=False)
    subjects = {subject_id(case_id) for case_id in train_data.case_ids}
    data_role = {
        "train": case_view_metadata(args.train_case_ids),
        "cases": len(train_data),
        "subjects": len(subjects),
        "role": "historical_train_cohort_only"
    }
    sampling_weights = train_data.sampling_weights()
    sampling_contract = {
        "basis": "native-grid WT positive voxels before resolution resize",
        "weights_sha256": tensor_sha256(sampling_weights),
        "source_positive_voxels_sha256": train_data.index[
            "source_positive_voxels_sha256"
        ],
        "sampler": "WeightedRandomSampler(replacement=True,num_samples=756)",
        "generator_seed": args.seed
    }
    write_json_atomic(output / "STARTED.json", {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "started_before_optimizer_step",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "cache": train_data.provenance(),
        "data_role": data_role,
        "sampling_contract": sampling_contract,
        "ema_protocol": {
            "start_after_epoch": args.ema_start_after_epoch,
            "decay_per_optimizer_step": args.ema_decay,
            "hyperparameter_search": False
        },
        "hyperparameters": serializable_args(args),
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False
    })

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            sampling_weights,
            num_samples=len(train_data),
            replacement=True,
            generator=generator
        ),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True
    )
    reload_loader = DataLoader(
        Subset(train_data, list(range(args.reload_cases))),
        batch_size=args.reload_cases,
        shuffle=False,
        num_workers=0
    )
    model = AutomaticPromptLocator(
        center_aware=False, init_filters=args.init_filters, dropout=args.dropout
    ).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != args.expected_parameter_count:
        raise ValueError("unexpected model parameter count")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay
    )
    loss_function = DiceCELoss(
        sigmoid=True,
        squared_pred=False,
        lambda_dice=1.0,
        lambda_ce=args.ce_weight
    )

    ema_model: AutomaticPromptLocator | None = None
    ema_updates = 0
    history: list[dict[str, Any]] = []
    checkpoints: dict[str, dict[str, Any]] = {}
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        lr = learning_rate_for_epoch(
            epoch,
            base_lr=args.base_lr,
            min_lr=args.min_lr,
            constant_epochs=args.constant_epochs,
            total_epochs=args.schedule_total_epochs
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
            if epoch > args.ema_start_after_epoch:
                if ema_model is None:
                    raise RuntimeError("EMA model was not initialized after epoch100")
                update_ema(ema_model, model, args.ema_decay)
                ema_updates += 1
            losses.append(float(loss.detach()))
            train_dsc.extend(
                dice_per_case(output_value.segmentation_logits.detach().float(), label)
                .cpu()
                .tolist()
            )
        if epoch == args.ema_start_after_epoch:
            ema_model = copy.deepcopy(model).requires_grad_(False)
            ema_model.eval()
        record = {
            "epoch": epoch,
            "learning_rate": lr,
            "train_loss": float(np.mean(losses)),
            "train_dsc": float(np.mean(train_dsc)),
            "optimizer_steps": len(loader),
            "ema_updates": ema_updates,
            "elapsed_seconds": time.time() - started
        }
        history.append(record)
        if epoch in (168, 200):
            if ema_model is None:
                raise RuntimeError("EMA model missing at a fixed checkpoint epoch")
            for kind, checkpoint_model in (("raw", model), ("ema", ema_model)):
                path = output / f"plain_128_full756_{kind}_fixed_epoch{epoch}.pth"
                checkpoints[f"{kind}_{epoch}"] = save_checkpoint(
                    path,
                    model=checkpoint_model,
                    kind=kind,
                    epoch=epoch,
                    learning_rate=lr,
                    args=args,
                    data_role=data_role,
                    sampling_contract=sampling_contract,
                    ema_updates=ema_updates
                )
        write_json_atomic(output / "progress.json", {
            "history": history,
            "checkpoints": checkpoints,
            "ema_updates": ema_updates
        })
        print(
            f"resolution=128 full756 ema={args.ema_decay} epoch={epoch}/{args.epochs} "
            f"lr={lr:.8g} loss={record['train_loss']:.6f} "
            f"train={record['train_dsc']:.6f} ema_updates={ema_updates}",
            flush=True
        )

    assert ema_model is not None
    reload_differences: dict[str, float] = {}
    for kind, original in (("raw", model), ("ema", ema_model)):
        path = Path(checkpoints[f"{kind}_200"]["path"])
        value = torch.load(path, map_location=device, weights_only=False)
        reloaded = AutomaticPromptLocator(
            center_aware=False, init_filters=args.init_filters, dropout=args.dropout
        ).to(device)
        reloaded.load_state_dict(value["model_state_dict"], strict=True)
        reload_differences[kind] = max_abs_difference(
            original, reloaded, reload_loader, device
        )
        del reloaded
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "resolution": 128,
        "epochs": 200,
        "checkpoints": checkpoints,
        "reload_max_abs_difference": reload_differences,
        "ema_protocol": {
            "start_after_epoch": args.ema_start_after_epoch,
            "decay_per_optimizer_step": args.ema_decay,
            "updates": ema_updates,
            "expected_updates": (args.epochs - args.ema_start_after_epoch) * len(loader)
        },
        "cache": train_data.provenance(),
        "data_role": data_role,
        "sampling_contract": sampling_contract,
        "parameter_count": args.expected_parameter_count,
        "history": history,
        "hyperparameters": serializable_args(args),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False
    }
    if ema_updates != report["ema_protocol"]["expected_updates"]:
        raise RuntimeError("EMA update count differs from the frozen protocol")
    if any(value != 0.0 for value in reload_differences.values()):
        raise RuntimeError("raw or EMA checkpoint reload is not exact")
    write_json_atomic(output / "TRAIN_DONE.json", report)
    print(
        f"TRAIN_DONE raw200={checkpoints['raw_200']['sha256']} "
        f"ema200={checkpoints['ema_200']['sha256']} updates={ema_updates} "
        f"reload_raw={reload_differences['raw']} reload_ema={reload_differences['ema']}",
        flush=True
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--train-case-ids", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-source-positive-voxels-sha256", required=True)
    parser.add_argument("--expected-train-cases", type=int, default=756)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260833)
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
    parser.add_argument("--ema-start-after-epoch", type=int, default=100)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--expected-parameter-count", type=int, default=4_700_897)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
