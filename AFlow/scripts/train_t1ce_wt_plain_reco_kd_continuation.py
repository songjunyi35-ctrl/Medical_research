#!/usr/bin/env python3
"""2026 ReCo-inspired region-normalized KD continuation for WT segmentation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
import scripts.train_t1ce_wt_plain_voxel_kd_continuation as base  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_reco_kd_continuation"
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_reco_kd_checkpoint"
KL_WEIGHT = 0.3
REGION_WEIGHT = 0.1
BOUNDARY_MULTIPLIER = 2.0


def _masked_case_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    flattened_value = value.flatten(1)
    flattened_mask = mask.flatten(1).to(value.dtype)
    count = flattened_mask.sum(dim=1)
    mean = (flattened_value * flattened_mask).sum(dim=1) / count.clamp_min(1.0)
    valid = count > 0
    if not bool(valid.any()):
        return value.new_zeros(())
    return mean[valid].mean()


def region_normalized_logit_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    label: torch.Tensor,
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape or student_logits.shape != label.shape:
        raise ValueError("student/teacher/label shapes differ")
    error = F.smooth_l1_loss(
        student_logits.float(), teacher_logits.float(), reduction="none"
    )[:, 0]
    foreground = label[:, 0] > 0.5
    dilated = F.max_pool3d(
        foreground.float().unsqueeze(1), kernel_size=5, stride=1, padding=2
    )[:, 0] > 0.5
    eroded = ~(
        F.max_pool3d(
            (~foreground).float().unsqueeze(1), kernel_size=5, stride=1, padding=2
        )[:, 0] > 0.5
    )
    boundary = dilated & ~eroded
    background = ~foreground & ~boundary
    foreground_loss = _masked_case_mean(error, foreground)
    boundary_loss = _masked_case_mean(error, boundary)
    background_loss = _masked_case_mean(error, background)
    return (
        foreground_loss + BOUNDARY_MULTIPLIER * boundary_loss + background_loss
    ) / (2.0 + BOUNDARY_MULTIPLIER)


def save_checkpoint(path, model, args, global_epoch, lr, data_role, sampling_contract):
    value = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "global_epoch": global_epoch,
        "continuation_epoch": global_epoch - 168,
        "resolution": 128,
        "learning_rate": lr,
        "model_state_dict": model.state_dict(),
        "architecture": {
            "class": "AutomaticPromptLocator",
            "backbone": "MONAI SegResNet",
            "center_aware": False,
            "init_filters": 16,
            "dropout": 0.1,
            "input_channels": ["T1ce"],
            "outputs": ["WT segmentation logit"]
        },
        "distillation": {
            "teacher_formula": "0.75 historical168 logit + 0.25 EMA168 logit",
            "temperature": 2.0,
            "voxel_kl_weight": KL_WEIGHT,
            "region_weight": REGION_WEIGHT,
            "region_definition": "GT foreground + 2-voxel morphological boundary + nonboundary background",
            "boundary_multiplier": BOUNDARY_MULTIPLIER,
            "historical_teacher_sha256": base.HISTORICAL_SHA256,
            "ema_teacher_sha256": base.EMA168_SHA256
        },
        "data_role": data_role,
        "sampling_contract": sampling_contract,
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False
    }
    torch.save(value, path)
    return {"path": str(path), "sha256": sha256_file(path), "global_epoch": global_epoch}


def run(args: argparse.Namespace):
    if args.seed != 20260846 or args.batch_size != 8 or args.continuation_epochs != 32:
        raise ValueError("frozen ReCo protocol requires seed20260846 batch8 32 epochs")
    if args.kl_weight != KL_WEIGHT or args.region_weight != REGION_WEIGHT:
        raise ValueError("ReCo KD weights differ from frozen protocol")
    if sha256_file(args.initial_checkpoint) != base.HISTORICAL_SHA256:
        raise ValueError("historical checkpoint SHA-256 mismatch")
    if sha256_file(args.ema_teacher_checkpoint) != base.EMA168_SHA256:
        raise ValueError("EMA checkpoint SHA-256 mismatch")
    if sha256_file(args.train_case_ids) != args.expected_train_sha256:
        raise ValueError("train manifest SHA-256 mismatch")
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ReCo KD requires CUDA")
    torch.cuda.set_device(device)
    student = base._model(historical_checkpoint, device)
    historical_teacher = base._model(historical_checkpoint, device).eval().requires_grad_(False)
    ema_teacher = base._model(ema_checkpoint, device).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=2e-4, weight_decay=1e-5)
    supervised_loss = DiceCELoss(
        sigmoid=True, squared_pred=False, lambda_dice=1.0, lambda_ce=0.1
    )
    sampling_weights = data.sampling_weights()
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        data,
        batch_size=8,
        sampler=WeightedRandomSampler(
            sampling_weights, num_samples=len(data), replacement=True, generator=generator
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
        "weights_sha256": base.tensor_sha256(sampling_weights),
        "generator_seed": args.seed,
        "num_samples": len(data),
        "replacement": True
    }
    write_json_atomic(output / "STARTED.json", {
        "schema": SCHEMA,
        "status": "started_before_optimizer_step",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "data_role": data_role,
        "sampling_contract": sampling_contract,
        "voxel_kl_weight": KL_WEIGHT,
        "region_weight": REGION_WEIGHT,
        "test_access": False
    })
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    history = []
    checkpoints = {}
    started = time.time()
    for continuation_epoch in range(1, 33):
        global_epoch = 168 + continuation_epoch
        lr = learning_rate_for_epoch(
            global_epoch, base_lr=2e-4, min_lr=1e-5,
            constant_epochs=100, total_epochs=200
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        student.train()
        total_values, sup_values, kl_values, region_values, train_dsc = [], [], [], [], []
        for batch in loader:
            image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            image, label = augment_batch(image, label, max_shift=8)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                history_logits = historical_teacher(image).segmentation_logits.float()
                ema_logits = ema_teacher(image).segmentation_logits.float()
                teacher_logits = 0.75 * history_logits + 0.25 * ema_logits
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                student_output = student(image)
                sup = supervised_loss(student_output.segmentation_logits, label)
            student_logits = student_output.segmentation_logits.float()
            kl = base.binary_voxel_kl(
                student_logits, teacher_logits, temperature=2.0
            )
            region = region_normalized_logit_kd(student_logits, teacher_logits, label)
            total = sup + KL_WEIGHT * kl + REGION_WEIGHT * region
            total.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()
            total_values.append(float(total.detach()))
            sup_values.append(float(sup.detach()))
            kl_values.append(float(kl.detach()))
            region_values.append(float(region.detach()))
            train_dsc.extend(
                dice_per_case(student_logits.detach(), label).cpu().tolist()
            )
        record = {
            "continuation_epoch": continuation_epoch,
            "global_epoch": global_epoch,
            "learning_rate": lr,
            "train_total_loss": float(np.mean(total_values)),
            "train_supervised_loss": float(np.mean(sup_values)),
            "train_kd_loss": float(np.mean(kl_values)),
            "train_region_loss": float(np.mean(region_values)),
            "train_dsc": float(np.mean(train_dsc)),
            "optimizer_steps": len(loader),
            "elapsed_seconds": time.time() - started
        }
        history.append(record)
        if global_epoch in base.SAVE_GLOBAL_EPOCHS:
            path = output / f"plain_128_reco_kd_global_epoch{global_epoch}.pth"
            checkpoints[str(global_epoch)] = save_checkpoint(
                path, student, args, global_epoch, lr, data_role, sampling_contract
            )
        write_json_atomic(output / "progress.json", {
            "history": history, "checkpoints": checkpoints
        })
        print(
            f"arm=reco_kd global_epoch={global_epoch}/200 "
            f"loss={record['train_total_loss']:.6f} sup={record['train_supervised_loss']:.6f} "
            f"kl={record['train_kd_loss']:.6f} region={record['train_region_loss']:.6f} "
            f"dsc={record['train_dsc']:.6f}",
            flush=True
        )
    final_path = Path(checkpoints["200"]["path"])
    final_value = torch.load(final_path, map_location=device, weights_only=False)
    reloaded = base._model(final_value, device)
    difference = base.reload_difference(student, reloaded, reload_loader, device)
    if difference != 0.0:
        raise RuntimeError("ReCo final reload is not exact")
    report = {
        "schema": SCHEMA,
        "status": "complete",
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
        "final_checkpoint": checkpoints["200"],
        "reload_max_abs_difference": difference,
        "elapsed_seconds": report["elapsed_seconds"]
    }, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--train-case-ids", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--ema-teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260846)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--continuation-epochs", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--kl-weight", type=float, default=KL_WEIGHT)
    parser.add_argument("--region-weight", type=float, default=REGION_WEIGHT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
