#!/usr/bin/env python3
"""Train one frozen fit864 trajectory and emit raw168 plus LAWA6 checkpoints."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
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
from scripts.build_t1ce_wt_native_volume_artifact import native_sampling_weights  # noqa: E402
from scripts.lawa6_utils import LawaAccumulator, capture_epochs, cpu_clone_state  # noqa: E402
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
    TEMPERATURE,
    _model,
    binary_voxel_kl,
)
from scripts.train_t1ce_wt_voxel_kd_972_draw_budget import (  # noqa: E402
    state_dict_sha256,
    tensor_sha256,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_lawa6_stage_a"
CHECKPOINT_SCHEMA = SCHEMA + ".checkpoint"
ARMS = ("stagea_seed20261061", "stagea_seed20261062")
ARM_SEED = {"stagea_seed20261061": 20261061, "stagea_seed20261062": 20261062}
FIT_SHA256 = "bf741b71121a9523047926fab6b6e3e7b880f7696adeec349cb6fbf01dbf1a7f"
SPLIT_SHA256 = "5e6f938ff40c32682748e249a7f355a90ad75ecb1d7a3a77038b0614fc781cdc"
IDEA_SHA256 = "b2cfc6b6b01624d0efa564c8f019b15f65b6892bf96fd76c1ec6331d8dc3092d"
PREREG_SHA256 = "cd72ab82bdadf5cd574a40f353cdf10218d4eace38ebe97ada3edc3f9a66cfc9"
NATIVE_ARTIFACT_SHA256 = "32be09856d71e78f512792c8bcf8f5f9ee137e6ba1e4358862ff2a61ba828c31"
CACHE_INDEX_SHA256 = "7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd"
FIT_CASES = 864
FIT_SUBJECTS = 393
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


def relative_l2(candidate: dict[str, torch.Tensor], raw: dict[str, torch.Tensor]) -> float:
    numerator = 0.0
    denominator = 0.0
    for key, raw_value in raw.items():
        if raw_value.is_floating_point() or raw_value.is_complex():
            delta = candidate[key].to(torch.float64) - raw_value.to(torch.float64)
            numerator += float(torch.sum(delta.square()))
            denominator += float(torch.sum(raw_value.to(torch.float64).square()))
    return math.sqrt(numerator / denominator) if denominator > 0 else 0.0


def checkpoint_payload(
    *,
    args: argparse.Namespace,
    kind: str,
    state: dict[str, torch.Tensor],
    initial_state_sha: str,
    data_role: dict[str, Any],
    sampling_contract: dict[str, Any],
    captured_epochs: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "kind": kind,
        "arm": args.arm,
        "seed": args.seed,
        "epoch": captured_epochs[-1],
        "model_state_dict": state,
        "model_state_sha256": state_dict_sha256(state),
        "initial_state_sha256": initial_state_sha,
        "architecture": {
            "class": "AutomaticPromptLocator",
            "backbone": "MONAI SegResNet",
            "init_filters": 16,
            "dropout": 0.1,
            "parameter_count": PARAMETER_COUNT,
            "input_channels": ["T1ce"],
            "outputs": ["WT segmentation logit"],
        },
        "data_role": data_role,
        "distillation": {
            "teacher_formula": "0.75 history168 + 0.25 EMA168 raw logits",
            "historical_teacher_sha256": HISTORICAL_SHA256,
            "ema_teacher_sha256": EMA168_SHA256,
            "temperature": TEMPERATURE,
            "kd_weight": KD_WEIGHT,
        },
        "sampling_contract": sampling_contract,
        "lawa_contract": {
            "kind": kind,
            "captured_epochs": list(captured_epochs),
            "count": 6,
            "coefficients": [1.0 / 6.0] * 6,
            "floating_accumulator": "cpu_float64",
            "nonfloating_policy": "require_exact_and_copy",
            "feedback_into_training": False,
        },
        "documents": {
            "idea_sha256": IDEA_SHA256,
            "prereg_sha256": PREREG_SHA256,
            "split_sha256": SPLIT_SHA256,
        },
        "selection": "fixed raw epoch168 versus uniform epochs163-168 LAWA6; no inner access during training",
        "access": {
            "train_only_inner": False,
            "fixed_validation108": False,
            "outer": False,
            "confirmation": False,
            "test270": False,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.arm not in ARMS:
        raise ValueError("unexpected LAWA6 Stage A arm")
    expected_seed = ARM_SEED[args.arm]
    if args.seed is None:
        args.seed = expected_seed
    if args.seed != expected_seed or args.batch_size != 8:
        raise ValueError("seed/batch differ from preregistration")
    if not args.smoke and (args.stop_epoch != STOP_EPOCH or args.schedule_total_epochs != SCHEDULE_TOTAL_EPOCHS):
        raise ValueError("formal epoch/schedule differ from preregistration")
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
        raise RuntimeError("LAWA6 training requires CUDA")
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

    smoke_epochs = capture_epochs(True)
    epochs_to_run = smoke_epochs[-1] if args.smoke else args.stop_epoch
    draws = args.batch_size if args.smoke else SAMPLER_DRAWS
    expected_steps = 1 if args.smoke else STEPS_PER_EPOCH
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
    if len(loader) != expected_steps:
        raise RuntimeError("optimizer-step count mismatch")

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

    captures = capture_epochs(args.smoke)
    accumulator = LawaAccumulator()
    captured: list[int] = []
    history: list[dict[str, Any]] = []
    last_image: torch.Tensor | None = None
    started = time.time()
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
        for batch in loader:
            image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            image, label = augment_batch(image, label, max_shift=8)
            last_image = image[:1].detach().clone()
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
                raise FloatingPointError("non-finite LAWA6 training loss")
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
            if float(grad_norm) <= 0:
                raise RuntimeError("non-positive student gradient")
            if any(
                parameter.grad is not None
                for teacher in (historical_teacher, ema_teacher)
                for parameter in teacher.parameters()
            ):
                raise RuntimeError("frozen teacher received gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
            supervised_losses.append(float(supervised.detach()))
            kd_losses.append(float(kd.detach()))
            train_dice.extend(dice_per_case(output.segmentation_logits.detach().float(), label).cpu().tolist())
        if epoch in captures:
            accumulator.update(model.state_dict())
            captured.append(epoch)
        row = {
            "epoch": epoch,
            "learning_rate": lr,
            "train_loss": float(np.mean(losses)),
            "train_supervised_loss": float(np.mean(supervised_losses)),
            "train_kd_loss": float(np.mean(kd_losses)),
            "train_dice": float(np.mean(train_dice)),
            "optimizer_steps": len(losses),
            "consumed_draws": len(losses) * args.batch_size,
            "lawa_captured": epoch in captures,
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        write_json_atomic(output_dir / "progress.json", {"history": history, "captured_epochs": captured})
        print(
            f"arm={args.arm} epoch={epoch}/{epochs_to_run} loss={row['train_loss']:.6f} "
            f"sup={row['train_supervised_loss']:.6f} kd={row['train_kd_loss']:.6f} "
            f"train={row['train_dice']:.6f} captured={row['lawa_captured']}", flush=True,
        )

    if tuple(captured) != captures or last_image is None:
        raise RuntimeError("frozen LAWA capture schedule incomplete")
    raw_state = cpu_clone_state(model.state_dict())
    lawa_state = accumulator.average()
    raw_sha = state_dict_sha256(raw_state)
    if raw_sha != state_dict_sha256(model.state_dict()):
        raise RuntimeError("LAWA capture mutated raw final state")
    sampling_contract = {
        "population_count": len(train_data),
        "requested_draws": draws,
        "consumed_draws": expected_steps * args.batch_size,
        "optimizer_steps": expected_steps,
        "native_weight_sha256": tensor_sha256(weights),
        "replacement": True,
    }
    raw_payload = checkpoint_payload(
        args=args,
        kind="raw_epoch168" if not args.smoke else "raw_smoke_epoch6",
        state=raw_state,
        initial_state_sha=initial_state_sha,
        data_role=data_role,
        sampling_contract=sampling_contract,
        captured_epochs=captures,
    )
    lawa_payload = checkpoint_payload(
        args=args,
        kind="lawa6_epochs163_168" if not args.smoke else "lawa6_smoke_epochs1_6",
        state=lawa_state,
        initial_state_sha=initial_state_sha,
        data_role=data_role,
        sampling_contract=sampling_contract,
        captured_epochs=captures,
    )
    raw_path = output_dir / f"raw_{args.arm}_epoch{captures[-1]}.pth"
    lawa_path = output_dir / f"lawa6_{args.arm}_epoch{captures[-1]}.pth"
    torch.save(raw_payload, raw_path)
    torch.save(lawa_payload, lawa_path)
    raw_reload = torch.load(raw_path, map_location="cpu", weights_only=False)
    lawa_reload = torch.load(lawa_path, map_location="cpu", weights_only=False)
    reload_exact = {
        "raw": raw_reload["model_state_sha256"] == state_dict_sha256(raw_reload["model_state_dict"]),
        "lawa": lawa_reload["model_state_sha256"] == state_dict_sha256(lawa_reload["model_state_dict"]),
    }

    model.eval()
    with torch.no_grad():
        model.load_state_dict(raw_state, strict=True)
        raw_direct = model(last_image).segmentation_logits.float()
        model.load_state_dict(raw_reload["model_state_dict"], strict=True)
        raw_reloaded = model(last_image).segmentation_logits.float()
        model.load_state_dict(lawa_state, strict=True)
        lawa_direct = model(last_image).segmentation_logits.float()
        model.load_state_dict(lawa_reload["model_state_dict"], strict=True)
        lawa_reloaded = model(last_image).segmentation_logits.float()
    deploy_exact = {
        "raw": torch.equal(raw_direct, raw_reloaded),
        "lawa": torch.equal(lawa_direct, lawa_reloaded),
    }
    finite_logits = bool(torch.isfinite(raw_direct).all() and torch.isfinite(lawa_direct).all())
    delta = relative_l2(lawa_state, raw_state)
    elapsed = time.time() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    gates = {
        "epoch_count_exact": len(history) == epochs_to_run,
        "all_steps_exact": all(row["optimizer_steps"] == expected_steps for row in history),
        "all_losses_finite": all(np.isfinite(row[key]) for row in history for key in ("train_loss", "train_supervised_loss", "train_kd_loss", "train_dice")),
        "capture_epochs_exact": tuple(captured) == captures,
        "capture_count_six": accumulator.count == 6,
        "raw_checkpoint_reload_exact": reload_exact["raw"],
        "lawa_checkpoint_reload_exact": reload_exact["lawa"],
        "raw_deploy_logit_exact": deploy_exact["raw"],
        "lawa_deploy_logit_exact": deploy_exact["lawa"],
        "raw_and_lawa_logits_finite": finite_logits,
        "lawa_nonzero_parameter_delta": delta > 0.0,
        "peak_at_most_24gib": peak <= MAX_PEAK_BYTES,
        "elapsed_within_6h": args.smoke or elapsed <= MAX_FORMAL_SECONDS,
    }
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete" if all(gates.values()) else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm,
        "seed": args.seed,
        "smoke": args.smoke,
        "all_gates_passed": all(gates.values()),
        "gates": gates,
        "checkpoints": {
            "raw": {"path": str(raw_path), "sha256": sha256_file(raw_path), "model_state_sha256": raw_payload["model_state_sha256"]},
            "lawa6": {"path": str(lawa_path), "sha256": sha256_file(lawa_path), "model_state_sha256": lawa_payload["model_state_sha256"]},
        },
        "initial_state_sha256": initial_state_sha,
        "lawa": {"captured_epochs": captured, "count": accumulator.count, "relative_l2_to_raw": delta},
        "data_role": data_role,
        "sampling_contract": sampling_contract,
        "documents": raw_payload["documents"],
        "history": history,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": peak,
        "access": raw_payload["access"],
    }
    write_json_atomic(output_dir / "TRAIN_DONE.json", report)
    print(
        f"TRAIN_DONE arm={args.arm} raw={raw_path} lawa={lawa_path} "
        f"relative_l2={delta:.9g} peak={peak} elapsed={elapsed:.2f}", flush=True,
    )
    if not all(gates.values()):
        raise RuntimeError("LAWA6 Stage A engineering gates failed")
    return report


def main() -> None:
    record = ROOT / "archive/research_records/2026-09-02_full972_lawa6_stability"
    split = ROOT / "data/datasets/brats2024_t1ce_lawa6_inner_seed20261060_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--fit-case-ids", type=Path, default=split / "fit_cases.jsonl")
    parser.add_argument("--split-manifest", type=Path, default=split / "split_manifest.json")
    parser.add_argument("--idea", type=Path, default=record / "IDEA.md")
    parser.add_argument("--prereg", type=Path, default=record / "PREREG.md")
    parser.add_argument("--native-artifact", type=Path, default=ROOT / "archive/research_records/2026-08-21_voxel_kd_data972_native_sampling/NATIVE_WT_VOLUMES.json")
    parser.add_argument("--historical-teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--ema-teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--stop-epoch", type=int, default=STOP_EPOCH)
    parser.add_argument("--schedule-total-epochs", type=int, default=SCHEDULE_TOTAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--base-lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-fit-cases", type=int, default=32)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
