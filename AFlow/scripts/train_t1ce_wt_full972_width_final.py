#!/usr/bin/env python3
"""Retrain full756/full972 random-init voxel-KD to the inner-selected epoch."""
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
from scripts.build_t1ce_wt_native_volume_artifact import native_sampling_weights  # noqa: E402
from scripts.train_t1ce_wt_plain_voxel_kd_continuation import (  # noqa: E402
    EMA168_SHA256, HISTORICAL_SHA256, KD_WEIGHT, TEACHER_EMA_WEIGHT,
    TEMPERATURE, _model, binary_voxel_kl,
)
from scripts.train_t1ce_wt_voxel_kd_972_draw_budget import state_dict_sha256, tensor_sha256  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_full972_width_final"
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_full972_width_final_checkpoint"
ARMS = ("full972_width16", "full972_width24")
ARM = {arm: (972, 444, "6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65") for arm in ARMS}
WIDTH = {"full972_width16": 16, "full972_width24": 24}
PARAMETERS = {"full972_width16": 4_700_897, "full972_width24": 10_574_161}
INNER_SELECTION_SHA256 = "a2d899b6c94e76fe60c9565cdcd32e0e7e22c40a34deb44fbce83b68b354a95e"
IDEA_SHA256 = "648581269c576141806f459230b94b85a4bbeb4a141d79f9d814a1829c2edc8b"
PREREG_SHA256 = "7a443e1497c8a061df4de959b457373ff6279f927df432adcdd9cccddc247c95"
NATIVE_ARTIFACT_SHA256 = "32be09856d71e78f512792c8bcf8f5f9ee137e6ba1e4358862ff2a61ba828c31"
CACHE_INDEX_SHA256 = "7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd"
SAMPLER_DRAWS = 972
CONSUMED_DRAWS = 968
STEPS_PER_EPOCH = 121
MAX_PEAK_BYTES = 32 * 1024**3
MAX_FORMAL_SECONDS = 8 * 60 * 60


_NATIVE_VOLUMES: dict[str, int] | None = None


class NativeFullDataset(BrainCropDataset):
    def sampling_weights(self) -> torch.Tensor:
        if _NATIVE_VOLUMES is None:
            raise RuntimeError("native volumes not configured")
        return native_sampling_weights(self.case_ids, _NATIVE_VOLUMES)


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    global _NATIVE_VOLUMES
    if args.arm not in ARMS:
        raise ValueError("unexpected final arm")
    if args.stop_epoch != 166 or args.schedule_total_epochs != 200:
        raise ValueError("authorized width final protocol requires stop=166 / schedule total=200")
    if args.seed != 20260961 or args.batch_size != 8:
        raise ValueError("width final requires seed20260961 and batch8")
    if args.init_filters is None:
        args.init_filters = WIDTH[args.arm]
    if args.expected_parameter_count is None:
        args.expected_parameter_count = PARAMETERS[args.arm]
    if args.init_filters != WIDTH[args.arm] or args.expected_parameter_count != PARAMETERS[args.arm]:
        raise ValueError("width final arm contract mismatch")
    expected_cases, expected_subjects, manifest_sha = ARM[args.arm]
    if sha256_file(args.train_case_ids) != manifest_sha:
        raise ValueError("final arm manifest SHA mismatch")
    decision = __import__("json").loads(args.inner_selection.read_text())
    if sha256_file(args.inner_selection) != INNER_SELECTION_SHA256:
        raise ValueError("inner decision SHA-256 mismatch")
    if decision.get("final_retraining_authorized") is not True:
        raise ValueError("inner gates did not authorize final training")
    if int(decision["selected_epoch"]) != args.stop_epoch:
        raise ValueError("fixed epoch differs from inner-authorized epoch")
    if any(decision.get(key) is not False for key in (
        "fixed_validation108_access", "outer_selector_access", "confirmation_access", "test_access"
    )):
        raise ValueError("decision access flags are not sealed")
    frozen = {
        args.idea: IDEA_SHA256,
        args.prereg: PREREG_SHA256,
        args.native_artifact: NATIVE_ARTIFACT_SHA256,
    }
    for path, expected in frozen.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen final input SHA mismatch: {path}")
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
        raise RuntimeError("fixed-epoch final training requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    train_data = NativeFullDataset(
        args.cache,
        expected_split="train",
        case_ids=read_case_ids(args.train_case_ids),
    )
    if len(train_data) != expected_cases:
        raise ValueError("unexpected final train case count")
    train_subjects = {subject_id(case_id) for case_id in train_data.case_ids}
    if len(train_subjects) != expected_subjects:
        raise ValueError("unexpected final subject count")
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
                "path": str(args.inner_selection.resolve()),
                "sha256": INNER_SELECTION_SHA256,
            },
            "data_role": data_role,
            "hyperparameters": serializable_args(args),
            "validation_access": False,
            "outer_selector_access": False,
            "confirmation_access": False,
            "test_access": False,
            "fixed_validation108_access": False,
            "validation_access": False,
        },
    )

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            train_data.sampling_weights(),
            num_samples=SAMPLER_DRAWS,
            replacement=True,
            generator=generator,
        ),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    if len(loader) != STEPS_PER_EPOCH:
        raise RuntimeError("final loader must expose121 steps")
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
        raise ValueError("unexpected final model parameter count")
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
    sampling_weights = train_data.sampling_weights()
    sampling_weight_sha256 = tensor_sha256(sampling_weights)

    history: list[dict[str, Any]] = []
    started = time.time()
    epochs_to_run = 1 if args.smoke else args.stop_epoch
    for epoch in range(1, epochs_to_run + 1):
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
        supervised_losses: list[float] = []
        kd_losses: list[float] = []
        train_dsc: list[float] = []
        for step_index, batch in enumerate(loader):
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
                raise FloatingPointError("non-finite final loss")
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0, error_if_nonfinite=True
            )
            if float(grad_norm) <= 0:
                raise RuntimeError("non-positive final gradient")
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
        record = {
            "epoch": epoch,
            "learning_rate": lr,
            "train_loss": float(np.mean(losses)),
            "train_supervised_loss": float(np.mean(supervised_losses)),
            "train_kd_loss": float(np.mean(kd_losses)),
            "train_dsc": float(np.mean(train_dsc)),
            "optimizer_steps": len(losses),
            "consumed_draws": len(losses) * args.batch_size,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        write_json_atomic(output_dir / "progress.json", {"history": history})
        print(
            f"epoch={epoch}/{args.stop_epoch} lr={lr:.8g} "
            f"loss={record['train_loss']:.5f} sup={record['train_supervised_loss']:.5f} "
            f"kd={record['train_kd_loss']:.5f} train={record['train_dsc']:.5f}",
            flush=True,
        )

    completed_epoch = history[-1]["epoch"]
    checkpoint_path = output_dir / f"plain_fulltrajectory_{args.arm}_epoch{completed_epoch}.pth"
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "epoch": completed_epoch,
        "learning_rate": history[-1]["learning_rate"],
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
        "data_role": data_role,
        "arm": args.arm,
        "distillation": {
            "teacher_formula": "0.75 history168 + 0.25 EMA168 raw logits",
            "historical_teacher_sha256": HISTORICAL_SHA256,
            "ema_teacher_sha256": EMA168_SHA256,
            "temperature": TEMPERATURE,
            "kd_weight": KD_WEIGHT,
        },
        "sampling_contract": {
            "population_count": len(train_data),
            "requested_draws": SAMPLER_DRAWS,
            "consumed_draws": CONSUMED_DRAWS,
            "optimizer_steps": STEPS_PER_EPOCH,
            "native_weight_sha256": sampling_weight_sha256,
            "replacement": True,
        },
        "method_documents": {
            "idea_sha256": IDEA_SHA256,
            "prereg_sha256": PREREG_SHA256,
            "inner_selection_sha256": INNER_SELECTION_SHA256,
            "native_artifact_sha256": NATIVE_ARTIFACT_SHA256,
        },
        "selection": "fixed epoch authorized by independent fresh-inner decision",
        "inner_decision_sha256": INNER_SELECTION_SHA256,
        "validation_access": False,
        "fixed_validation108_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    torch.save(checkpoint, checkpoint_path)
    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reload_exact = reloaded["model_state_sha256"] == state_dict_sha256(reloaded["model_state_dict"])
    elapsed = time.time() - started
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    gates = {
        "epoch_count_exact": len(history) == (1 if args.smoke else args.stop_epoch),
        "all_steps_match_mode": all(
            row["optimizer_steps"] == (args.smoke_steps if args.smoke else STEPS_PER_EPOCH)
            for row in history
        ),
        "all_draws_match_mode": all(
            row["consumed_draws"] == (args.smoke_steps * args.batch_size if args.smoke else CONSUMED_DRAWS)
            for row in history
        ),
        "all_losses_finite": all(
            np.isfinite(row[key]) for row in history
            for key in ("train_loss", "train_supervised_loss", "train_kd_loss", "train_dsc")
        ),
        "checkpoint_model_state_exact": reload_exact,
        "peak_allocated_at_most_32gib": peak_memory <= MAX_PEAK_BYTES,
        "elapsed_within_8h": elapsed <= MAX_FORMAL_SECONDS,
    }
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete" if all(gates.values()) else "failed",
        "arm": args.arm,
        "all_gates_passed": all(gates.values()),
        "gates": gates,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "fixed_epoch": completed_epoch,
        "authorized_final_epoch": args.stop_epoch,
        "schedule_total_epochs": args.schedule_total_epochs,
        "data_role": data_role,
        "parameter_count": parameter_count,
        "initial_state_sha256": initial_state_sha256,
        "sampling_contract": checkpoint["sampling_contract"],
        "method_documents": checkpoint["method_documents"],
        "history": history,
        "hyperparameters": serializable_args(args),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": peak_memory,
        "validation_access": False,
        "fixed_validation108_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    write_json_atomic(output_dir / "TRAIN_DONE.json", report)
    print(
        f"TRAIN_DONE fixed_epoch={completed_epoch} checkpoint={checkpoint_path} "
        f"sha256={report['checkpoint']['sha256']}",
        flush=True,
    )
    if not report["all_gates_passed"]:
        raise RuntimeError("final fulltrajectory gates failed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--train-case-ids", type=Path, required=True)
    record = ROOT / "archive/research_records/2026-08-26_full972_width24_fulltrajectory"
    prior = ROOT / "archive/research_records/2026-08-21_voxel_kd_data972_native_sampling"
    parser.add_argument("--inner-selection", type=Path, default=record / "INNER_SELECTION.json")
    parser.add_argument("--historical-teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--ema-teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--idea", type=Path, default=record / "IDEA.md")
    parser.add_argument("--prereg", type=Path, default=record / "PREREG.md")
    parser.add_argument("--native-artifact", type=Path, default=prior / "NATIVE_WT_VOLUMES.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260961)
    parser.add_argument("--stop-epoch", type=int, default=166)
    parser.add_argument("--schedule-total-epochs", type=int, default=200)
    parser.add_argument("--constant-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--init-filters", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--base-lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-shift", type=int, default=8)
    parser.add_argument("--ce-weight", type=float, default=0.1)
    parser.add_argument("--expected-parameter-count", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
