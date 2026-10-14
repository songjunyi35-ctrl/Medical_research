#!/usr/bin/env python3
"""Matched 32-epoch supervised/KD continuation from historical epoch168."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator  # noqa: E402
from sammed3d.t1ce_wt_variable_resolution import VariableResolutionBrainCropDataset  # noqa: E402
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


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_voxel_kd_continuation"
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_voxel_kd_checkpoint"
HISTORICAL_SHA256 = "f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1"
EMA168_SHA256 = "4ea97dacb9a538993e1c16593bc56b03ce6dc3df2f457c1f0426439c6252197e"
TEACHER_EMA_WEIGHT = 0.25
TEMPERATURE = 2.0
KD_WEIGHT = 0.3
SAVE_GLOBAL_EPOCHS = (176, 184, 192, 200)


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def binary_voxel_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student/teacher logit shapes differ")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    student_binary = torch.cat(
        (torch.zeros_like(student_logits), student_logits), dim=1
    ) / temperature
    teacher_binary = torch.cat(
        (torch.zeros_like(teacher_logits), teacher_logits), dim=1
    ) / temperature
    student_log_probability = F.log_softmax(student_binary, dim=1)
    teacher_probability = F.softmax(teacher_binary, dim=1)
    per_voxel = F.kl_div(
        student_log_probability, teacher_probability, reduction="none"
    ).sum(dim=1)
    return per_voxel.mean() * temperature * temperature


def _model(checkpoint: dict[str, Any], device: torch.device):
    architecture = checkpoint["architecture"]
    model = AutomaticPromptLocator(
        center_aware=False,
        init_filters=int(architecture["init_filters"]),
        dropout=float(architecture["dropout"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


@torch.inference_mode()
def reload_difference(
    original: AutomaticPromptLocator,
    reloaded: AutomaticPromptLocator,
    loader: DataLoader,
    device: torch.device,
) -> float:
    original.eval()
    reloaded.eval()
    maximum = 0.0
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            left = original(image).segmentation_logits
            right = reloaded(image).segmentation_logits
        maximum = max(maximum, float((left.float() - right.float()).abs().max()))
    return maximum


def save_checkpoint(
    path: Path,
    *,
    model: AutomaticPromptLocator,
    args: argparse.Namespace,
    global_epoch: int,
    learning_rate: float,
    data_role: dict[str, Any],
    sampling_contract: dict[str, Any],
) -> dict[str, Any]:
    value = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "arm": args.arm,
        "continuation_epoch": global_epoch - 168,
        "global_epoch": global_epoch,
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
        "distillation": {
            "enabled": args.arm == "kd",
            "teacher_formula": "0.75 historical168 logit + 0.25 EMA168 logit",
            "teacher_ema_weight": TEACHER_EMA_WEIGHT,
            "temperature": TEMPERATURE,
            "kd_weight": KD_WEIGHT if args.arm == "kd" else 0.0,
            "historical_teacher_sha256": HISTORICAL_SHA256,
            "ema_teacher_sha256": EMA168_SHA256
        },
        "data_role": data_role,
        "sampling_contract": sampling_contract,
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False
    }
    torch.save(value, path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "arm": args.arm,
        "global_epoch": global_epoch
    }


def run(args: argparse.Namespace):
    if args.arm not in {"control", "kd"}:
        raise ValueError("arm must be control or kd")
    if args.continuation_epochs != 32 or args.start_global_epoch != 168:
        raise ValueError("frozen continuation is global epoch169-200")
    expected_kd = KD_WEIGHT if args.arm == "kd" else 0.0
    if args.kd_weight != expected_kd or args.temperature != TEMPERATURE:
        raise ValueError("arm KD hyperparameters differ from frozen protocol")
    if args.batch_size != 8 or args.seed != 20260846:
        raise ValueError("frozen continuation requires batch8 seed20260846")
    if sha256_file(args.initial_checkpoint) != HISTORICAL_SHA256:
        raise ValueError("initial historical checkpoint SHA-256 mismatch")
    if sha256_file(args.ema_teacher_checkpoint) != EMA168_SHA256:
        raise ValueError("EMA teacher checkpoint SHA-256 mismatch")
    if sha256_file(args.train_case_ids) != args.expected_train_sha256:
        raise ValueError("full train manifest SHA-256 mismatch")
    historical_checkpoint = torch.load(
        args.initial_checkpoint, map_location="cpu", weights_only=False
    )
    ema_checkpoint = torch.load(
        args.ema_teacher_checkpoint, map_location="cpu", weights_only=False
    )
    case_ids = read_case_ids(args.train_case_ids)
    assert case_ids is not None
    data = VariableResolutionBrainCropDataset(
        args.cache,
        expected_resolution=128,
        expected_manifest_sha256=args.expected_train_sha256,
        case_ids=case_ids
    )
    if len(data) != 756 or data.index.get("test_access") is not False:
        raise ValueError("unexpected full train cache")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("voxel KD continuation requires CUDA")
    torch.cuda.set_device(device)
    student = _model(historical_checkpoint, device)
    historical_teacher = None
    ema_teacher = None
    if args.arm == "kd":
        historical_teacher = _model(historical_checkpoint, device).eval().requires_grad_(False)
        ema_teacher = _model(ema_checkpoint, device).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.base_lr, weight_decay=args.weight_decay
    )
    supervised_loss = DiceCELoss(
        sigmoid=True,
        squared_pred=False,
        lambda_dice=1.0,
        lambda_ce=0.1
    )
    sampling_weights = data.sampling_weights()
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            sampling_weights,
            num_samples=len(data),
            replacement=True,
            generator=generator
        ),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True
    )
    reload_loader = DataLoader(
        Subset(data, list(range(8))), batch_size=8, shuffle=False, num_workers=0
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    data_role = {
        "train": case_view_metadata(args.train_case_ids),
        "cases": len(data),
        "subjects": len({subject_id(case) for case in data.case_ids})
    }
    sampling_contract = {
        "weights_sha256": tensor_sha256(sampling_weights),
        "generator_seed": args.seed,
        "num_samples": len(data),
        "replacement": True
    }
    write_json_atomic(output / "STARTED.json", {
        "schema": SCHEMA,
        "status": "started_before_optimizer_step",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "data_role": data_role,
        "sampling_contract": sampling_contract,
        "distillation": {
            "temperature": args.temperature,
            "kd_weight": args.kd_weight,
            "teacher_ema_weight": TEACHER_EMA_WEIGHT
        },
        "validation_access": False,
        "test_access": False
    })
    # Reset RNG after all model construction so control/KD student dropout streams match.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    history = []
    checkpoints = {}
    started = time.time()
    for continuation_epoch in range(1, args.continuation_epochs + 1):
        global_epoch = args.start_global_epoch + continuation_epoch
        lr = learning_rate_for_epoch(
            global_epoch,
            base_lr=args.base_lr,
            min_lr=args.min_lr,
            constant_epochs=100,
            total_epochs=200
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        student.train()
        total_values = []
        supervised_values = []
        kd_values = []
        train_dsc = []
        for batch in loader:
            image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            image, label = augment_batch(image, label, max_shift=8)
            teacher_logits = None
            if args.arm == "kd":
                assert historical_teacher is not None and ema_teacher is not None
                with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    history_logits = historical_teacher(image).segmentation_logits.float()
                    ema_logits = ema_teacher(image).segmentation_logits.float()
                    teacher_logits = (
                        history_logits * (1.0 - TEACHER_EMA_WEIGHT)
                        + ema_logits * TEACHER_EMA_WEIGHT
                    )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                student_output = student(image)
                sup = supervised_loss(student_output.segmentation_logits, label)
            if teacher_logits is None:
                kd = torch.zeros((), device=device)
            else:
                kd = binary_voxel_kl(
                    student_output.segmentation_logits.float(),
                    teacher_logits,
                    temperature=args.temperature
                )
            total = sup + args.kd_weight * kd
            total.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()
            total_values.append(float(total.detach()))
            supervised_values.append(float(sup.detach()))
            kd_values.append(float(kd.detach()))
            train_dsc.extend(
                dice_per_case(
                    student_output.segmentation_logits.detach().float(), label
                ).cpu().tolist()
            )
        record = {
            "continuation_epoch": continuation_epoch,
            "global_epoch": global_epoch,
            "learning_rate": lr,
            "train_total_loss": float(np.mean(total_values)),
            "train_supervised_loss": float(np.mean(supervised_values)),
            "train_kd_loss": float(np.mean(kd_values)),
            "train_dsc": float(np.mean(train_dsc)),
            "optimizer_steps": len(loader),
            "elapsed_seconds": time.time() - started
        }
        history.append(record)
        if global_epoch in SAVE_GLOBAL_EPOCHS:
            path = output / f"plain_128_{args.arm}_global_epoch{global_epoch}.pth"
            checkpoints[str(global_epoch)] = save_checkpoint(
                path,
                model=student,
                args=args,
                global_epoch=global_epoch,
                learning_rate=lr,
                data_role=data_role,
                sampling_contract=sampling_contract
            )
        write_json_atomic(output / "progress.json", {
            "history": history, "checkpoints": checkpoints
        })
        print(
            f"arm={args.arm} global_epoch={global_epoch}/200 "
            f"loss={record['train_total_loss']:.6f} "
            f"sup={record['train_supervised_loss']:.6f} "
            f"kd={record['train_kd_loss']:.6f} "
            f"dsc={record['train_dsc']:.6f}",
            flush=True
        )
    final_path = Path(checkpoints["200"]["path"])
    final_value = torch.load(final_path, map_location=device, weights_only=False)
    reloaded = _model(final_value, device)
    difference = reload_difference(student, reloaded, reload_loader, device)
    if difference != 0.0:
        raise RuntimeError("final continuation reload is not exact")
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "arm": args.arm,
        "checkpoints": checkpoints,
        "history": history,
        "reload_max_abs_difference": difference,
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False
    }
    write_json_atomic(output / "TRAIN_DONE.json", report)
    print(json.dumps({
        "arm": args.arm,
        "final_checkpoint": checkpoints["200"],
        "reload_max_abs_difference": difference,
        "elapsed_seconds": report["elapsed_seconds"]
    }, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("control", "kd"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--train-case-ids", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--ema-teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260846)
    parser.add_argument("--start-global-epoch", type=int, default=168)
    parser.add_argument("--continuation-epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--kd-weight", type=float, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
