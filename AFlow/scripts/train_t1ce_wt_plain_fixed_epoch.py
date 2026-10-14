#!/usr/bin/env python3
"""Train the inner-authorized plain schedule on all 756 train cases at fixed epoch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
    write_json_atomic,
)
from scripts.train_t1ce_wt_plain_long_schedule import (  # noqa: E402
    learning_rate_for_epoch,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_fixed_epoch_training"
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_fixed_epoch_checkpoint"


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.stop_epoch != 168 or args.schedule_total_epochs != 200:
        raise ValueError("authorized final protocol requires stop=168 / schedule total=200")
    lowered = str(args.train_case_ids.resolve()).lower()
    if any(token in lowered for token in ("selector", "confirmation", "sealed", "test270")):
        raise ValueError("final training manifest contains a forbidden role token")
    if sha256_file(args.train_case_ids) != args.expected_train_sha256:
        raise ValueError("756-case train manifest SHA-256 mismatch")
    decision = __import__("json").loads(args.inner_decision.read_text())
    if sha256_file(args.inner_decision) != args.expected_inner_decision_sha256:
        raise ValueError("inner decision SHA-256 mismatch")
    if decision.get("all_inner_gates_passed") is not True:
        raise ValueError("inner gates did not authorize final training")
    if int(decision["post100"]["epoch"]) != args.stop_epoch:
        raise ValueError("fixed epoch differs from inner-authorized epoch")
    if any(decision.get(key) is not False for key in (
        "outer_selector_access", "confirmation_access", "test_access"
    )):
        raise ValueError("decision access flags are not sealed")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("fixed-epoch final training requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    train_data = BrainCropDataset(
        args.cache,
        expected_split="train",
        case_ids=read_case_ids(args.train_case_ids),
    )
    if len(train_data) != args.expected_train_cases:
        raise ValueError("unexpected final train case count")
    train_subjects = {subject_id(case_id) for case_id in train_data.case_ids}
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    data_role = {
        "train": case_view_metadata(args.train_case_ids),
        "cases": len(train_data),
        "subjects": len(train_subjects),
    }
    write_json_atomic(
        output_dir / "STARTED.json",
        {
            "schema": SCHEMA,
            "schema_version": 1,
            "status": "started_before_optimizer_step",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "inner_decision": {
                "path": str(args.inner_decision.resolve()),
                "sha256": args.expected_inner_decision_sha256,
            },
            "data_role": data_role,
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
            train_data.sampling_weights(),
            num_samples=len(train_data),
            replacement=True,
            generator=generator,
        ),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    model = AutomaticPromptLocator(
        center_aware=False,
        init_filters=args.init_filters,
        dropout=args.dropout,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != args.expected_parameter_count:
        raise ValueError("unexpected final model parameter count")
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
    for epoch in range(1, args.stop_epoch + 1):
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
                output = model(image)
                loss = loss_function(output.segmentation_logits, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            train_dsc.extend(
                dice_per_case(output.segmentation_logits.detach().float(), label).cpu().tolist()
            )
        record = {
            "epoch": epoch,
            "learning_rate": lr,
            "train_loss": float(np.mean(losses)),
            "train_dsc": float(np.mean(train_dsc)),
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        write_json_atomic(output_dir / "progress.json", {"history": history})
        print(
            f"epoch={epoch}/{args.stop_epoch} lr={lr:.8g} "
            f"loss={record['train_loss']:.5f} train={record['train_dsc']:.5f}",
            flush=True,
        )

    checkpoint_path = output_dir / "plain_fixed_epoch168.pth"
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "epoch": args.stop_epoch,
        "learning_rate": history[-1]["learning_rate"],
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
        "selection": "fixed epoch authorized by independent fresh-inner decision",
        "inner_decision_sha256": args.expected_inner_decision_sha256,
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    torch.save(checkpoint, checkpoint_path)
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "fixed_epoch": args.stop_epoch,
        "schedule_total_epochs": args.schedule_total_epochs,
        "data_role": data_role,
        "parameter_count": parameter_count,
        "history": history,
        "hyperparameters": serializable_args(args),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    write_json_atomic(output_dir / "TRAIN_DONE.json", report)
    print(
        f"TRAIN_DONE fixed_epoch={args.stop_epoch} checkpoint={checkpoint_path} "
        f"sha256={report['checkpoint']['sha256']}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--train-case-ids", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-train-cases", type=int, default=756)
    parser.add_argument("--inner-decision", type=Path, required=True)
    parser.add_argument("--expected-inner-decision-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--stop-epoch", type=int, default=168)
    parser.add_argument("--schedule-total-epochs", type=int, default=200)
    parser.add_argument("--constant-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
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
