#!/usr/bin/env python3
"""Train one preregistered fit864 seed teacher for consensus distillation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from scripts.build_t1ce_wt_native_volume_artifact import native_sampling_weights  # noqa: E402
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
from scripts.train_t1ce_wt_plain_long_schedule import learning_rate_for_epoch  # noqa: E402
from scripts.train_t1ce_wt_plain_voxel_kd_continuation import (  # noqa: E402
    EMA168_SHA256,
    HISTORICAL_SHA256,
    KD_WEIGHT,
    TEACHER_EMA_WEIGHT,
    TEMPERATURE,
    _model,
    binary_voxel_kl,
)
from scripts.train_t1ce_wt_voxel_kd_972_draw_budget import (  # noqa: E402
    state_dict_sha256,
    tensor_sha256,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_consensus_seed_teacher"
CHECKPOINT_SCHEMA = SCHEMA + ".checkpoint"
ARMS = ("teacher_seed20261031", "teacher_seed20261032", "teacher_seed20261033")
ARM_SEED = {
    "teacher_seed20261031": 20261031,
    "teacher_seed20261032": 20261032,
    "teacher_seed20261033": 20261033,
}
FIT_SHA256 = "f46a2158fb9cbc7a797321cee13bdd97bf372750c4146b4a0dbe89e8f0e7edb8"
SPLIT_SHA256 = "b88fe33fd353035d271afbea0fb8e3c349f78b3206a60fcdacae2654c7cc6dae"
IDEA_SHA256 = "97b048fd71c6b2ad7d4266497ce6a074f27cedd98c22e60bb926fad8ce1fc0d4"
PREREG_SHA256 = "120d1f00899d1d0d51910b063a4e9301b26b7e0f584fc1c78938c0a0546bdb73"
NATIVE_ARTIFACT_SHA256 = "32be09856d71e78f512792c8bcf8f5f9ee137e6ba1e4358862ff2a61ba828c31"
CACHE_INDEX_SHA256 = "7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd"
FIT_CASES = 864
FIT_SUBJECTS = 401
PARAMETER_COUNT = 4_700_897
STOP_EPOCH = 168
SCHEDULE_TOTAL_EPOCHS = 200
SAMPLER_DRAWS = 972
CONSUMED_DRAWS = 968
STEPS_PER_EPOCH = 121
MAX_PEAK_BYTES = 24 * 1024**3
MAX_FORMAL_SECONDS = 6 * 60 * 60


class NativeFitDataset(BrainCropDataset):
    def __init__(self, *args: Any, native_volumes: dict[str, int], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.native_volumes = native_volumes

    def sampling_weights(self) -> torch.Tensor:
        return native_sampling_weights(self.case_ids, self.native_volumes)


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.arm not in ARMS:
        raise ValueError("unexpected consensus teacher arm")
    expected_seed = ARM_SEED[args.arm]
    if args.seed is None:
        args.seed = expected_seed
    if args.seed != expected_seed or args.batch_size != 8:
        raise ValueError("teacher seed/batch differ from preregistration")
    if args.stop_epoch != STOP_EPOCH or args.schedule_total_epochs != SCHEDULE_TOTAL_EPOCHS:
        raise ValueError("teacher epoch/schedule differ from preregistration")
    frozen = {
        args.fit_case_ids: FIT_SHA256,
        args.split_manifest: SPLIT_SHA256,
        args.idea: IDEA_SHA256,
        args.prereg: PREREG_SHA256,
        args.native_artifact: NATIVE_ARTIFACT_SHA256,
    }
    for path, expected in frozen.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen input SHA mismatch: {path}")
    if sha256_file(args.cache / "index.json") != CACHE_INDEX_SHA256:
        raise ValueError("train cache index SHA mismatch")
    if sha256_file(args.historical_teacher_checkpoint) != HISTORICAL_SHA256:
        raise ValueError("historical teacher SHA mismatch")
    if sha256_file(args.ema_teacher_checkpoint) != EMA168_SHA256:
        raise ValueError("EMA teacher SHA mismatch")
    artifact = json.loads(args.native_artifact.read_text())
    if artifact.get("status") != "passed" or artifact.get("audit", {}).get("all_gates_passed") is not True:
        raise ValueError("native-volume artifact did not pass")
    native_volumes = {str(key): int(value) for key, value in artifact["native_wt_voxels"].items()}

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("consensus teacher training requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    limit = args.limit_fit_cases if args.smoke else None
    train_data = NativeFitDataset(
        args.cache,
        expected_split="train",
        case_ids=read_case_ids(args.fit_case_ids),
        limit=limit,
        native_volumes=native_volumes,
    )
    train_subjects = {subject_id(case_id) for case_id in train_data.case_ids}
    if not args.smoke and (len(train_data) != FIT_CASES or len(train_subjects) != FIT_SUBJECTS):
        raise ValueError("formal fit case/subject count mismatch")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    data_role = {
        "fit": case_view_metadata(args.fit_case_ids),
        "cases": len(train_data),
        "subjects": len(train_subjects),
    }
    write_json_atomic(output_dir / "STARTED.json", {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "started_before_optimizer_step",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "data_role": data_role,
        "hyperparameters": serializable_args(args),
        "access": {"train_only_inner": False, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False},
    })

    draws = args.smoke_steps * args.batch_size if args.smoke else SAMPLER_DRAWS
    generator = torch.Generator().manual_seed(args.seed)
    weights = train_data.sampling_weights()
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(weights, num_samples=draws, replacement=True, generator=generator),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    expected_steps = args.smoke_steps if args.smoke else STEPS_PER_EPOCH
    if len(loader) != expected_steps:
        raise RuntimeError("teacher optimizer-step count mismatch")

    history_checkpoint = torch.load(args.historical_teacher_checkpoint, map_location="cpu", weights_only=False)
    ema_checkpoint = torch.load(args.ema_teacher_checkpoint, map_location="cpu", weights_only=False)
    historical_teacher = _model(history_checkpoint, device).eval().requires_grad_(False)
    ema_teacher = _model(ema_checkpoint, device).eval().requires_grad_(False)
    model = AutomaticPromptLocator(center_aware=False, init_filters=16, dropout=0.1).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise ValueError("student parameter-count mismatch")
    initial_state_sha = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay, foreach=None, fused=None)
    loss_function = DiceCELoss(sigmoid=True, squared_pred=False, lambda_dice=1.0, lambda_ce=0.1)

    history: list[dict[str, Any]] = []
    started = time.time()
    epochs_to_run = 1 if args.smoke else args.stop_epoch
    for epoch in range(1, epochs_to_run + 1):
        lr = learning_rate_for_epoch(
            epoch,
            base_lr=args.base_lr,
            min_lr=args.min_lr,
            constant_epochs=100,
            total_epochs=args.schedule_total_epochs,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        losses: list[float] = []
        supervised_losses: list[float] = []
        kd_losses: list[float] = []
        train_dice: list[float] = []
        for step, batch in enumerate(loader):
            image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            image, label = augment_batch(image, label, max_shift=8)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                history_logits = historical_teacher(image).segmentation_logits.float()
                ema_logits = ema_teacher(image).segmentation_logits.float()
                teacher_logits = history_logits * 0.75 + ema_logits * 0.25
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(image)
                supervised = loss_function(output.segmentation_logits, label)
            kd = binary_voxel_kl(output.segmentation_logits.float(), teacher_logits, temperature=TEMPERATURE)
            loss = supervised + KD_WEIGHT * kd
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite teacher-training loss")
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
            if float(grad_norm) <= 0:
                raise RuntimeError("non-positive teacher-training gradient")
            if any(
                parameter.grad is not None
                for teacher in (historical_teacher, ema_teacher)
                for parameter in teacher.parameters()
            ):
                raise RuntimeError("frozen base teacher received gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
            supervised_losses.append(float(supervised.detach()))
            kd_losses.append(float(kd.detach()))
            train_dice.extend(dice_per_case(output.segmentation_logits.detach().float(), label).cpu().tolist())
        row = {
            "epoch": epoch,
            "learning_rate": lr,
            "train_loss": float(np.mean(losses)),
            "train_supervised_loss": float(np.mean(supervised_losses)),
            "train_kd_loss": float(np.mean(kd_losses)),
            "train_dice": float(np.mean(train_dice)),
            "optimizer_steps": len(losses),
            "consumed_draws": len(losses) * args.batch_size,
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        write_json_atomic(output_dir / "progress.json", {"history": history})
        print(
            f"arm={args.arm} epoch={epoch}/{args.stop_epoch} loss={row['train_loss']:.6f} "
            f"sup={row['train_supervised_loss']:.6f} kd={row['train_kd_loss']:.6f} "
            f"train={row['train_dice']:.6f}", flush=True,
        )

    completed_epoch = int(history[-1]["epoch"])
    checkpoint_path = output_dir / f"plain_{args.arm}_epoch{completed_epoch}.pth"
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "arm": args.arm,
        "seed": args.seed,
        "epoch": completed_epoch,
        "model_state_dict": model.state_dict(),
        "model_state_sha256": state_dict_sha256(model.state_dict()),
        "initial_state_sha256": initial_state_sha,
        "optimizer_state_dict": optimizer.state_dict(),
        "architecture": {"class": "AutomaticPromptLocator", "init_filters": 16, "dropout": 0.1, "input_channels": ["T1ce"], "outputs": ["WT segmentation logit"]},
        "data_role": data_role,
        "distillation": {"teacher_formula": "0.75 history168 + 0.25 EMA168 raw logits", "temperature": TEMPERATURE, "kd_weight": KD_WEIGHT},
        "sampling_contract": {"population_count": len(train_data), "requested_draws": draws, "consumed_draws": expected_steps * args.batch_size, "optimizer_steps": expected_steps, "native_weight_sha256": tensor_sha256(weights), "replacement": True},
        "documents": {"idea_sha256": IDEA_SHA256, "prereg_sha256": PREREG_SHA256, "split_sha256": SPLIT_SHA256},
        "selection": "fixed epoch168; no inner access during training",
        "access": {"train_only_inner": False, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False},
    }
    torch.save(checkpoint, checkpoint_path)
    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reload_exact = reloaded["model_state_sha256"] == state_dict_sha256(reloaded["model_state_dict"])
    elapsed = time.time() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    gates = {
        "epoch_count_exact": len(history) == (1 if args.smoke else STOP_EPOCH),
        "all_steps_exact": all(row["optimizer_steps"] == expected_steps for row in history),
        "all_losses_finite": all(np.isfinite(row[key]) for row in history for key in ("train_loss", "train_supervised_loss", "train_kd_loss", "train_dice")),
        "checkpoint_reload_exact": reload_exact,
        "peak_at_most_24gib": peak <= MAX_PEAK_BYTES,
        "elapsed_within_6h": elapsed <= MAX_FORMAL_SECONDS,
    }
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete" if all(gates.values()) else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm,
        "seed": args.seed,
        "all_gates_passed": all(gates.values()),
        "gates": gates,
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
        "model_state_sha256": checkpoint["model_state_sha256"],
        "initial_state_sha256": initial_state_sha,
        "data_role": data_role,
        "sampling_contract": checkpoint["sampling_contract"],
        "documents": checkpoint["documents"],
        "history": history,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": peak,
        "access": checkpoint["access"],
    }
    write_json_atomic(output_dir / "TRAIN_DONE.json", report)
    print(f"TRAIN_DONE arm={args.arm} checkpoint={checkpoint_path} sha256={report['checkpoint']['sha256']}", flush=True)
    if not report["all_gates_passed"]:
        raise RuntimeError("consensus seed teacher engineering gates failed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    record = ROOT / "archive/research_records/2026-09-01_fresh_multiseed_consensus_distillation"
    split = ROOT / "data/datasets/brats2024_t1ce_consensus_distill_inner_seed20261030_v2"
    prior = ROOT / "archive/research_records/2026-08-21_voxel_kd_data972_native_sampling"
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--fit-case-ids", type=Path, default=split / "fit_cases.jsonl")
    parser.add_argument("--split-manifest", type=Path, default=split / "split_manifest.json")
    parser.add_argument("--idea", type=Path, default=record / "IDEA.md")
    parser.add_argument("--prereg", type=Path, default=record / "PREREG.md")
    parser.add_argument("--native-artifact", type=Path, default=prior / "NATIVE_WT_VOLUMES.json")
    parser.add_argument("--historical-teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--ema-teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stop-epoch", type=int, default=STOP_EPOCH)
    parser.add_argument("--schedule-total-epochs", type=int, default=SCHEDULE_TOTAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--limit-fit-cases", type=int, default=8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
