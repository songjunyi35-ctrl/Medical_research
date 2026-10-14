#!/usr/bin/env python3
"""Train one fixed-epoch plain locator arm on a declared cubic resolution.

Formal runs are locked to epoch 168 on the historical 200-epoch schedule
(epochs 1-100 at 2e-4, then the same cosine trajectory toward 1e-5).  Inner
tune data are read once, after the final optimizer step; no epoch or threshold
is selected.  This runner has no outer, confirmation, or test mode.
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
    FORMAL_RESOLUTIONS,
    VariableResolutionBrainCropDataset,
)
from scripts.train_t1ce_wt_capg3d import (  # noqa: E402
    augment_batch,
    case_view_metadata,
    dice_per_case,
    read_case_ids,
    sha256_file,
    subject_id,
    validate,
    write_json_atomic,
)
from scripts.train_t1ce_wt_plain_long_schedule import (  # noqa: E402
    learning_rate_for_epoch,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_fixed_resolution_training"
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_fixed_resolution_checkpoint"
FORBIDDEN_ROLE_TOKENS = ("selector", "confirmation", "sealed", "test270")


def assert_development_path(path: Path) -> None:
    lowered = str(path.resolve()).lower()
    if any(token in lowered for token in FORBIDDEN_ROLE_TOKENS):
        raise ValueError(f"forbidden data role in path: {path}")


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


@torch.inference_mode()
def reload_max_abs_difference(
    original: AutomaticPromptLocator,
    reloaded: AutomaticPromptLocator,
    loader: DataLoader,
    device: torch.device,
) -> float:
    original.eval()
    reloaded.eval()
    maximum = 0.0
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            left = original(image).segmentation_logits
            right = reloaded(image).segmentation_logits
        maximum = max(maximum, float((left.float() - right.float()).abs().max()))
    return maximum


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.cache, args.fit_case_ids, args.inner_tune_case_ids, args.output_dir):
        assert_development_path(path)
    if args.resolution not in FORMAL_RESOLUTIONS:
        raise ValueError("resolution must be 128 or 160")
    if not args.smoke:
        frozen = {
            "epochs": 168,
            "schedule_total_epochs": 200,
            "constant_epochs": 100,
            "batch_size": 8,
            "base_lr": 2e-4,
            "min_lr": 1e-5,
            "weight_decay": 1e-5,
            "max_shift": 8,
            "ce_weight": 0.1,
        }
        for name, expected in frozen.items():
            if getattr(args, name) != expected:
                raise ValueError(f"formal protocol requires {name}={expected}")

    if sha256_file(args.fit_case_ids) != args.expected_fit_sha256:
        raise ValueError("fit case-view SHA-256 mismatch")
    if sha256_file(args.inner_tune_case_ids) != args.expected_inner_tune_sha256:
        raise ValueError("inner-tune case-view SHA-256 mismatch")
    fit_case_ids = read_case_ids(args.fit_case_ids)
    tune_case_ids = read_case_ids(args.inner_tune_case_ids)
    assert fit_case_ids is not None and tune_case_ids is not None
    fit_data = VariableResolutionBrainCropDataset(
        args.cache,
        expected_resolution=args.resolution,
        expected_manifest_sha256=args.expected_cohort_sha256,
        case_ids=fit_case_ids,
    )
    tune_data = VariableResolutionBrainCropDataset(
        args.cache,
        expected_resolution=args.resolution,
        expected_manifest_sha256=args.expected_cohort_sha256,
        case_ids=tune_case_ids,
    )
    if len(fit_data) != args.expected_fit_cases:
        raise ValueError(f"fit view has {len(fit_data)} cases, expected {args.expected_fit_cases}")
    if len(tune_data) != args.expected_inner_tune_cases:
        raise ValueError(
            f"inner-tune view has {len(tune_data)} cases, "
            f"expected {args.expected_inner_tune_cases}"
        )
    if not args.smoke and fit_data.index.get("smoke") is not False:
        raise ValueError("formal training cannot consume a smoke cache")
    if fit_data.index["source_positive_voxels_sha256"] != (
        args.expected_source_positive_voxels_sha256
    ):
        raise ValueError("resolution-independent source WT volume hash mismatch")
    if fit_data.index["source_positive_voxels_sha256"] != (
        tune_data.index["source_positive_voxels_sha256"]
    ):
        raise ValueError("fit/tune cache provenance mismatch")

    fit_cases = set(fit_data.case_ids)
    tune_cases = set(tune_data.case_ids)
    fit_subjects = {subject_id(case_id) for case_id in fit_cases}
    tune_subjects = {subject_id(case_id) for case_id in tune_cases}
    if fit_cases & tune_cases or fit_subjects & tune_subjects:
        raise ValueError("fit/inner-tune case or subject leakage")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("fixed-resolution training requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    data_roles = {
        "fit": case_view_metadata(args.fit_case_ids),
        "inner_tune": case_view_metadata(args.inner_tune_case_ids),
        "fit_cases": len(fit_data),
        "fit_subjects": len(fit_subjects),
        "inner_tune_cases": len(tune_data),
        "inner_tune_subjects": len(tune_subjects),
        "case_overlap": 0,
        "subject_overlap": 0,
    }
    sampling_weights = fit_data.sampling_weights()
    sampling_contract = {
        "basis": "native-grid WT positive voxels before resolution-specific resize",
        "weights_sha256": tensor_sha256(sampling_weights),
        "source_positive_voxels_sha256": fit_data.index[
            "source_positive_voxels_sha256"
        ],
        "sampler": "WeightedRandomSampler(replacement=True,num_samples=fit_cases)",
        "generator_seed": args.seed,
    }
    started_record = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started_before_optimizer_step",
        "resolution": args.resolution,
        "fixed_epoch": args.epochs,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "cache": fit_data.provenance(),
        "data_roles": data_roles,
        "sampling_contract": sampling_contract,
        "hyperparameters": serializable_args(args),
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    write_json_atomic(output / "STARTED.json", started_record)

    generator = torch.Generator().manual_seed(args.seed)
    fit_loader = DataLoader(
        fit_data,
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            sampling_weights,
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
        center_aware=False, init_filters=args.init_filters, dropout=args.dropout
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
        for batch in fit_loader:
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
            "optimizer_steps": len(fit_loader),
            "elapsed_seconds": time.time() - started,
            "inner_tune_accessed": False,
        }
        history.append(record)
        write_json_atomic(output / "progress.json", {"history": history})
        print(
            f"resolution={args.resolution} epoch={epoch}/{args.epochs} lr={lr:.8g} "
            f"loss={record['train_loss']:.6f} train={record['train_dsc']:.6f}",
            flush=True,
        )

    validation = validate(model, tune_loader, device)
    training_peak_allocated = int(torch.cuda.max_memory_allocated(device))
    training_peak_reserved = int(torch.cuda.max_memory_reserved(device))
    checkpoint_path = output / f"plain_fixed_epoch{args.epochs}.pth"
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "epoch": args.epochs,
        "resolution": args.resolution,
        "arm": "plain",
        "fixed_logit_threshold": 0.0,
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
        "cache": fit_data.provenance(),
        "data_roles": data_roles,
        "sampling_contract": sampling_contract,
        "hyperparameters": serializable_args(args),
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    torch.save(checkpoint, checkpoint_path)

    reloaded = AutomaticPromptLocator(
        center_aware=False, init_filters=args.init_filters, dropout=args.dropout
    ).to(device)
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    reloaded.load_state_dict(loaded["model_state_dict"], strict=True)
    reload_difference = reload_max_abs_difference(model, reloaded, tune_loader, device)
    reload_validation = validate(reloaded, tune_loader, device)
    if reload_validation != validation:
        raise RuntimeError("fresh checkpoint reload changed fixed-epoch validation summary")

    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "resolution": args.resolution,
        "arm": "128_control" if args.resolution == 128 else "160_candidate",
        "fixed_epoch": args.epochs,
        "fixed_logit_threshold": 0.0,
        "inner_tune_first_access": "after_final_optimizer_step",
        "validation": validation,
        "reload_max_abs_difference": reload_difference,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "cache": fit_data.provenance(),
        "data_roles": data_roles,
        "sampling_contract": sampling_contract,
        "parameter_count": parameter_count,
        "schedule": {
            "trained_epochs": args.epochs,
            "definition_total_epochs": args.schedule_total_epochs,
            "constant_epochs": args.constant_epochs,
            "base_lr": args.base_lr,
            "min_lr": args.min_lr,
            "epoch168_lr": learning_rate_for_epoch(
                168,
                base_lr=args.base_lr,
                min_lr=args.min_lr,
                constant_epochs=args.constant_epochs,
                total_epochs=args.schedule_total_epochs,
            ),
        },
        "hyperparameters": serializable_args(args),
        "history": history,
        "training_peak_gpu_memory_bytes": training_peak_allocated,
        "training_peak_gpu_reserved_bytes": training_peak_reserved,
        "end_to_end_peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": time.time() - started,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    write_json_atomic(output / "TRAIN_DONE.json", report)
    print(
        f"TRAIN_DONE resolution={args.resolution} epoch={args.epochs} "
        f"scan={validation['locator_dsc_mean']:.9f} "
        f"subject={validation['locator_subject_dsc_mean']:.9f} "
        f"point={validation['automatic_point_hit_rate']:.9f} "
        f"reload_max={reload_difference:.9g}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--resolution", type=int, choices=FORMAL_RESOLUTIONS, required=True)
    parser.add_argument("--fit-case-ids", type=Path, required=True)
    parser.add_argument("--inner-tune-case-ids", type=Path, required=True)
    parser.add_argument("--expected-fit-sha256", required=True)
    parser.add_argument("--expected-inner-tune-sha256", required=True)
    parser.add_argument("--expected-cohort-sha256", required=True)
    parser.add_argument("--expected-source-positive-voxels-sha256", required=True)
    parser.add_argument("--expected-fit-cases", type=int, default=648)
    parser.add_argument("--expected-inner-tune-cases", type=int, default=108)
    parser.add_argument("--expected-parameter-count", type=int, default=4_700_897)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--epochs", type=int, default=168)
    parser.add_argument("--schedule-total-epochs", type=int, default=200)
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
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
