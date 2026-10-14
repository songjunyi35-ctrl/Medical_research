#!/usr/bin/env python3
"""Train full random-init voxel-KD trajectories on a common train-inner split."""
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
from scripts.build_t1ce_wt_native_volume_artifact import native_sampling_weights  # noqa: E402
from scripts.train_t1ce_wt_plain_voxel_kd_continuation import (  # noqa: E402
    EMA168_SHA256,
    HISTORICAL_SHA256,
    KD_WEIGHT,
    TEACHER_EMA_WEIGHT,
    TEMPERATURE,
    _model,
    binary_voxel_kl,
)
from scripts.train_t1ce_wt_voxel_kd_972_draw_budget import state_dict_sha256, tensor_sha256  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_full972_width_inner"
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_full972_width_inner_checkpoint"
FORBIDDEN_ROLE_TOKENS = ("selector", "confirmation", "sealed", "test270")
ARMS = ("width16_control", "width24_candidate")
FIT = {arm: (864, 395, "6533098a1b0b6e0e7ea37bf166de0b2b0744d5404ec6dac91840e9cfaf8a6f3b") for arm in ARMS}
WIDTH = {"width16_control": 16, "width24_candidate": 24}
PARAMETERS = {"width16_control": 4_700_897, "width24_candidate": 10_574_161}
TUNE_SHA256 = "87dd75450fb1258b2f5ff2c87ae6635cff3077bb5dfb8109b95cb51681da8bef"
SPLIT_SHA256 = "c4fe9066ecb35fd7e86fada4493b59aec03d4b1b21c95f9cbce5c2719532c60b"
CACHE_INDEX_SHA256 = "7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd"
NATIVE_ARTIFACT_SHA256 = "32be09856d71e78f512792c8bcf8f5f9ee137e6ba1e4358862ff2a61ba828c31"
IDEA_SHA256 = "648581269c576141806f459230b94b85a4bbeb4a141d79f9d814a1829c2edc8b"
PREREG_SHA256 = "7a443e1497c8a061df4de959b457373ff6279f927df432adcdd9cccddc247c95"
SAMPLER_DRAWS = 972
CONSUMED_DRAWS = 968
STEPS_PER_EPOCH = 121
MAX_PEAK_BYTES = 32 * 1024**3
MAX_FORMAL_SECONDS = 8 * 60 * 60


_NATIVE_VOLUMES: dict[str, int] | None = None


class NativeFitDataset(BrainCropDataset):
    def sampling_weights(self) -> torch.Tensor:
        if _NATIVE_VOLUMES is None:
            raise RuntimeError("native volumes not configured")
        return native_sampling_weights(self.case_ids, _NATIVE_VOLUMES)


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
    global _NATIVE_VOLUMES
    if args.arm not in ARMS:
        raise ValueError("unexpected arm")
    if not args.smoke and (args.epochs != 200 or args.constant_epochs != 100):
        raise ValueError("formal protocol requires 200 total / 100 constant epochs")
    if args.seed != 20260960 or args.batch_size != 8:
        raise ValueError("frozen width inner protocol requires seed20260960 and batch8")
    if args.init_filters is None:
        args.init_filters = WIDTH[args.arm]
    if args.expected_parameter_count is None:
        args.expected_parameter_count = PARAMETERS[args.arm]
    if args.init_filters != WIDTH[args.arm] or args.expected_parameter_count != PARAMETERS[args.arm]:
        raise ValueError("arm width/parameter contract mismatch")
    assert_inner_only_path(args.fit_case_ids)
    assert_inner_only_path(args.fresh_tune_case_ids)
    expected_cases, expected_subjects, expected_fit_sha = FIT[args.arm]
    if sha256_file(args.fit_case_ids) != expected_fit_sha:
        raise ValueError("fit manifest SHA-256 mismatch")
    if sha256_file(args.fresh_tune_case_ids) != TUNE_SHA256:
        raise ValueError("fresh-tune manifest SHA-256 mismatch")
    frozen = {
        args.idea: IDEA_SHA256,
        args.prereg: PREREG_SHA256,
        args.split_manifest: SPLIT_SHA256,
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
    artifact = __import__("json").loads(args.native_artifact.read_text())
    if artifact.get("status") != "passed" or artifact.get("audit", {}).get("all_gates_passed") is not True:
        raise ValueError("native artifact gates failed")
    _NATIVE_VOLUMES = {str(key): int(value) for key, value in artifact["native_wt_voxels"].items()}

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

    fit_data = NativeFitDataset(
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
    if len(fit_data) != expected_cases:
        raise ValueError("unexpected fit case count")
    if len(tune_data) != 108:
        raise ValueError("unexpected fresh-tune case count")
    if set(fit_data.case_ids) & set(tune_data.case_ids):
        raise ValueError("fit/fresh-tune case leakage")
    fit_subjects = {subject_id(case_id) for case_id in fit_data.case_ids}
    tune_subjects = {subject_id(case_id) for case_id in tune_data.case_ids}
    if fit_subjects & tune_subjects:
        raise ValueError("fit/fresh-tune subject leakage")
    if len(fit_subjects) != expected_subjects or len(tune_subjects) != 49:
        raise ValueError("subject count mismatch")

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
            "validation_access": False,
            "fixed_validation108_access": False,
            "inner_tune_access": True,
        },
    )

    generator = torch.Generator().manual_seed(args.seed)
    fit_loader = DataLoader(
        fit_data,
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            fit_data.sampling_weights(),
            num_samples=SAMPLER_DRAWS,
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
    if len(fit_loader) != STEPS_PER_EPOCH:
        raise RuntimeError("inner loader must expose121 steps")

    historical_checkpoint = torch.load(
        args.historical_teacher_checkpoint, map_location="cpu", weights_only=False
    )
    ema_checkpoint = torch.load(
        args.ema_teacher_checkpoint, map_location="cpu", weights_only=False
    )
    historical_teacher = _model(historical_checkpoint, device).eval().requires_grad_(False)
    ema_teacher = _model(ema_checkpoint, device).eval().requires_grad_(False)
    model = AutomaticPromptLocator(
        center_aware=False,
        init_filters=args.init_filters,
        dropout=args.dropout,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != args.expected_parameter_count:
        raise ValueError(f"parameter count {parameter_count} does not match protocol")
    initial_state_sha256 = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay,
        foreach=None, fused=None,
    )
    loss_function = DiceCELoss(
        sigmoid=True,
        squared_pred=False,
        lambda_dice=1.0,
        lambda_ce=args.ce_weight,
    )
    sampling_weights = fit_data.sampling_weights()
    sampling_weight_sha256 = tensor_sha256(sampling_weights)

    history: list[dict[str, Any]] = []
    best: dict[str, dict[str, Any]] = {
        "pre100": {"value": float("-inf"), "subject": float("-inf"), "epoch": 0, "path": output_dir / "pre100_best.pth"},
        "post100": {"value": float("-inf"), "subject": float("-inf"), "epoch": 0, "path": output_dir / "post100_best.pth"},
    }
    started = time.time()
    epochs_to_run = 1 if args.smoke else args.epochs
    for epoch in range(1, epochs_to_run + 1):
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
        supervised_losses: list[float] = []
        kd_losses: list[float] = []
        train_dsc: list[float] = []
        for step_index, batch in enumerate(fit_loader):
            image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            image, label = augment_batch(image, label, max_shift=args.max_shift)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                history_logits = historical_teacher(image).segmentation_logits.float()
                ema_logits = ema_teacher(image).segmentation_logits.float()
                teacher_logits = (
                    history_logits * (1.0 - TEACHER_EMA_WEIGHT)
                    + ema_logits * TEACHER_EMA_WEIGHT
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(image)
                supervised = loss_function(output.segmentation_logits, label)
            kd = binary_voxel_kl(
                output.segmentation_logits.float(), teacher_logits, temperature=TEMPERATURE
            )
            loss = supervised + KD_WEIGHT * kd
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite inner loss")
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0, error_if_nonfinite=True
            )
            if float(grad_norm) <= 0:
                raise RuntimeError("non-positive student gradient")
            if any(
                parameter.grad is not None
                for teacher in (historical_teacher, ema_teacher)
                for parameter in teacher.parameters()
            ):
                raise RuntimeError("teacher received gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
            supervised_losses.append(float(supervised.detach()))
            kd_losses.append(float(kd.detach()))
            train_dsc.extend(
                dice_per_case(output.segmentation_logits.detach().float(), label).cpu().tolist()
            )
            if args.smoke and step_index + 1 >= args.smoke_steps:
                break

        validation = validate(model, tune_loader, device)
        value = float(validation["locator_dsc_mean"])
        subject_value = float(validation["locator_subject_dsc_mean"])
        window = "pre100" if epoch <= args.constant_epochs else "post100"
        strict_new_high = (value, subject_value, -epoch) > (
            float(best[window]["value"]), float(best[window]["subject"]), -int(best[window]["epoch"])
        )
        if strict_new_high:
            best[window]["value"] = value
            best[window]["subject"] = subject_value
            best[window]["epoch"] = epoch
            checkpoint = {
                "schema": CHECKPOINT_SCHEMA,
                "schema_version": 1,
                "epoch": epoch,
                "window": window,
                "arm": args.arm,
                "learning_rate": lr,
                "validation": validation,
                "model_state_dict": model.state_dict(),
                "model_state_sha256": state_dict_sha256(model.state_dict()),
                "initial_state_sha256": initial_state_sha256,
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
                "distillation": {
                    "teacher_formula": "0.75 history168 + 0.25 EMA168 raw logits",
                    "historical_teacher_sha256": HISTORICAL_SHA256,
                    "ema_teacher_sha256": EMA168_SHA256,
                    "temperature": TEMPERATURE,
                    "kd_weight": KD_WEIGHT,
                },
                "sampling_contract": {
                    "population_count": len(fit_data),
                    "requested_draws": SAMPLER_DRAWS,
                    "consumed_draws": CONSUMED_DRAWS,
                    "optimizer_steps": STEPS_PER_EPOCH,
                    "native_weight_sha256": sampling_weight_sha256,
                    "replacement": True,
                },
                "method_documents": {
                    "idea_sha256": IDEA_SHA256,
                    "prereg_sha256": PREREG_SHA256,
                    "split_sha256": SPLIT_SHA256,
                    "native_artifact_sha256": NATIVE_ARTIFACT_SHA256,
                },
                "fixed_validation108_access": False,
                "inner_tune_access": True,
                "outer_selector_access": False,
                "confirmation_access": False,
                "test_access": False,
                "validation_access": False,
            }
            torch.save(checkpoint, best[window]["path"])

        record = {
            "epoch": epoch,
            "window": window,
            "learning_rate": lr,
            "train_loss": float(np.mean(losses)),
            "train_supervised_loss": float(np.mean(supervised_losses)),
            "train_kd_loss": float(np.mean(kd_losses)),
            "train_dsc": float(np.mean(train_dsc)),
            "optimizer_steps": len(losses),
            "consumed_draws": len(losses) * args.batch_size,
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
                    name: {"epoch": item["epoch"], "value": item["value"], "subject": item["subject"]}
                    for name, item in best.items()
                },
            },
        )
        print(
            f"epoch={epoch} window={window} lr={lr:.8g} "
            f"loss={record['train_loss']:.5f} sup={record['train_supervised_loss']:.5f} "
            f"kd={record['train_kd_loss']:.5f} train={record['train_dsc']:.5f} "
            f"tune={value:.6f} subject={validation['locator_subject_dsc_mean']:.6f} "
            f"point={validation['automatic_point_hit_rate']:.4f} "
            f"best={best[window]['value']:.6f}@{best[window]['epoch']}",
            flush=True,
        )

    selections: dict[str, dict[str, Any]] = {}
    for window, item in best.items():
        if int(item["epoch"]) == 0:
            continue
        path = Path(item["path"])
        selections[window] = {
            "epoch": int(item["epoch"]),
            "value": float(item["value"]),
            "subject": float(item["subject"]),
            "checkpoint": {"path": str(path), "sha256": sha256_file(path)},
        }
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_state_sha256") != state_dict_sha256(checkpoint["model_state_dict"]):
            raise RuntimeError("best checkpoint model-state hash mismatch")
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    elapsed = time.time() - started
    gates = {
        "epoch_count_exact": len(history) == (1 if args.smoke else 200),
        "all_step_counts_exact": all(
            row["optimizer_steps"] == (args.smoke_steps if args.smoke else STEPS_PER_EPOCH)
            for row in history
        ),
        "all_losses_finite": all(
            np.isfinite(row[key])
            for row in history
            for key in ("train_loss", "train_supervised_loss", "train_kd_loss", "train_dsc")
        ),
        "both_window_checkpoints_saved": args.smoke or set(selections) == {"pre100", "post100"},
        "peak_allocated_at_most_32gib": peak_memory <= MAX_PEAK_BYTES,
        "elapsed_within_8h": elapsed <= MAX_FORMAL_SECONDS,
    }
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "arm": args.arm,
        "all_gates_passed": all(gates.values()),
        "gates": gates,
        "initial_state_sha256": initial_state_sha256,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
        "validation_access": False,
        "fixed_validation108_access": False,
        "inner_tune_access": True,
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
        "sampling_contract": {
            "requested_draws": SAMPLER_DRAWS,
            "consumed_draws": CONSUMED_DRAWS,
            "optimizer_steps": STEPS_PER_EPOCH,
            "population_count": len(fit_data),
            "native_weight_sha256": sampling_weight_sha256,
        },
        "method_documents": {
            "idea_sha256": IDEA_SHA256,
            "prereg_sha256": PREREG_SHA256,
            "split_sha256": SPLIT_SHA256,
            "native_artifact_sha256": NATIVE_ARTIFACT_SHA256,
        },
        "hyperparameters": serializable_args(args),
        "history": history,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": peak_memory,
    }
    write_json_atomic(output_dir / "TRAIN_DONE.json", report)
    print("TRAIN_DONE " + " ".join(
        f"{name}={item['value']:.9f}@{item['epoch']}"
        for name, item in selections.items()
    ), flush=True)
    if not report["all_gates_passed"]:
        raise RuntimeError("inner training gates failed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--fit-case-ids", type=Path, required=True)
    split_root = ROOT / "data/datasets/brats2024_t1ce_fulltrajectory_common_inner_seed20260930_v1"
    record = ROOT / "archive/research_records/2026-08-26_full972_width24_fulltrajectory"
    prior = ROOT / "archive/research_records/2026-08-21_voxel_kd_data972_native_sampling"
    parser.add_argument("--fresh-tune-case-ids", type=Path, default=split_root / "common_inner_tune_cases.jsonl")
    parser.add_argument("--historical-teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--ema-teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--idea", type=Path, default=record / "IDEA.md")
    parser.add_argument("--prereg", type=Path, default=record / "PREREG.md")
    parser.add_argument("--split-manifest", type=Path, default=split_root / "split_manifest.json")
    parser.add_argument("--native-artifact", type=Path, default=prior / "NATIVE_WT_VOLUMES.json")
    parser.add_argument("--expected-parameter-count", type=int, default=None)
    parser.add_argument("--limit-fit-cases", type=int, default=None)
    parser.add_argument("--limit-fresh-tune-cases", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260960)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--constant-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--init-filters", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--base-lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-shift", type=int, default=8)
    parser.add_argument("--ce-weight", type=float, default=0.1)
    parser.add_argument("--smoke-steps", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
