#!/usr/bin/env python3
"""Fixed-compute cache-matched 756/972 data-pool voxel-KD experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
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
    SAVE_GLOBAL_EPOCHS,
    TEACHER_EMA_WEIGHT,
    TEMPERATURE,
    _model,
    binary_voxel_kl,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_voxel_kd_data972_train"
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_voxel_kd_data972_checkpoint"
ARMS = ("data756_control", "data972_candidate")
ARM_COUNTS = {"data756_control": (756, 346), "data972_candidate": (972, 444)}
ARM_MANIFEST_SHA256 = {
    "data756_control": "a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16",
    "data972_candidate": "6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65",
}
CACHE_INDEX_SHA256 = "7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd"
IDEA_SHA256 = "e98a2129a15c6532a96da400345337f3b7029d009ec3464c8a8bf1829a578ff9"
PREREG_SHA256 = "c52df229219e661c87e6a363e3aa01d0f6f0009957a1b5a4ca620729be9f5a8a"
AMENDMENT_SHA256 = "0c09945496e0f11e6d5a455b290d0a5e284a46b7fee712062a4d513eabbef3aa"
FEASIBILITY_SHA256 = "16c5b604c74b579e53decbdcbf7fc592782256be83ff21d85f5fa4f93164398e"
SAMPLER_DRAWS = 756
CONSUMED_DRAWS = 752
STEPS_PER_EPOCH = 94
MAX_PEAK_BYTES = 24 * 1024**3
MAX_FORMAL_SECONDS = 45 * 60


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def case_order_sha256(case_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(case_ids).encode("utf-8")).hexdigest()


def optimizer_contract(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    group = optimizer.param_groups[0]
    return {
        "class": "AdamW",
        "betas": list(group["betas"]),
        "eps": float(group["eps"]),
        "weight_decay": float(group["weight_decay"]),
        "amsgrad": bool(group["amsgrad"]),
        "maximize": bool(group["maximize"]),
        "foreach": group["foreach"],
        "fused": group["fused"],
        "capturable": bool(group["capturable"]),
        "parameter_tensors": sum(len(g["params"]) for g in optimizer.param_groups),
    }


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    args: argparse.Namespace,
    global_epoch: int,
    learning_rate: float,
    data_role: dict[str, Any],
    sampling_contract: dict[str, Any],
    initial_state_sha256: str,
    optimizer_spec: dict[str, Any],
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
        "model_state_sha256": state_dict_sha256(model.state_dict()),
        "initial_state_sha256": initial_state_sha256,
        "architecture": {
            "class": "AutomaticPromptLocator",
            "backbone": "MONAI SegResNet",
            "center_aware": False,
            "init_filters": 16,
            "dropout": 0.1,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "input_channels": ["T1ce"],
            "outputs": ["WT segmentation logit"],
        },
        "distillation": {
            "enabled": True,
            "teacher_formula": "0.75 historical168 logit + 0.25 EMA168 logit",
            "teacher_ema_weight": TEACHER_EMA_WEIGHT,
            "temperature": TEMPERATURE,
            "kd_weight": KD_WEIGHT,
            "historical_teacher_sha256": HISTORICAL_SHA256,
            "ema_teacher_sha256": EMA168_SHA256,
        },
        "data_role": data_role,
        "sampling_contract": sampling_contract,
        "optimizer_contract": optimizer_spec,
        "method_documents": {
            "idea_sha256": IDEA_SHA256,
            "prereg_sha256": PREREG_SHA256,
            "amendment_sha256": AMENDMENT_SHA256,
            "feasibility_sha256": FEASIBILITY_SHA256,
        },
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    torch.save(value, path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "model_state_sha256": value["model_state_sha256"],
        "arm": args.arm,
        "global_epoch": global_epoch,
    }


def strict_reload_difference(
    model: nn.Module,
    checkpoint_path: Path,
    loader: DataLoader,
    device: torch.device,
) -> float:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reloaded = _model(checkpoint, device).eval()
    model.eval()
    maximum = 0.0
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device=device, dtype=torch.float32)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                left = model(image).segmentation_logits
                right = reloaded(image).segmentation_logits
            maximum = max(maximum, float((left.float() - right.float()).abs().max()))
    return maximum


def validate_frozen_inputs(args: argparse.Namespace) -> tuple[list[str], str]:
    if args.arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if args.seed != 20260846 or args.batch_size != 8:
        raise ValueError("frozen protocol requires seed20260846 and batch8")
    if args.sampler_draws != SAMPLER_DRAWS:
        raise ValueError("frozen protocol requires exactly756 sampler draws")
    if args.start_global_epoch != 168 or args.continuation_epochs != 32:
        raise ValueError("frozen formal continuation is global169-200")
    if args.kd_weight != KD_WEIGHT or args.temperature != TEMPERATURE:
        raise ValueError("KD hyperparameters differ from frozen protocol")
    expected_manifest_sha = ARM_MANIFEST_SHA256[args.arm]
    if sha256_file(args.train_case_ids) != expected_manifest_sha:
        raise ValueError("arm manifest SHA-256 mismatch")
    if sha256_file(args.cache / "index.json") != CACHE_INDEX_SHA256:
        raise ValueError("972 cache index SHA-256 mismatch")
    if sha256_file(args.initial_checkpoint) != HISTORICAL_SHA256:
        raise ValueError("historical168 checkpoint SHA-256 mismatch")
    if sha256_file(args.ema_teacher_checkpoint) != EMA168_SHA256:
        raise ValueError("EMA168 checkpoint SHA-256 mismatch")
    docs = {
        args.idea: IDEA_SHA256,
        args.prereg: PREREG_SHA256,
        args.amendment: AMENDMENT_SHA256,
        args.feasibility: FEASIBILITY_SHA256,
    }
    for path, expected in docs.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen method document SHA mismatch: {path}")
    case_ids = read_case_ids(args.train_case_ids)
    if case_ids is None:
        raise ValueError("arm manifest has no case IDs")
    expected_count, expected_subjects = ARM_COUNTS[args.arm]
    if len(case_ids) != expected_count:
        raise ValueError("arm manifest case count mismatch")
    if len({subject_id(case) for case in case_ids}) != expected_subjects:
        raise ValueError("arm manifest subject count mismatch")
    return case_ids, expected_manifest_sha


def run(args: argparse.Namespace) -> dict[str, Any]:
    case_ids, expected_manifest_sha = validate_frozen_inputs(args)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {args.output_dir}")
    if args.mode == "formal" and (args.num_workers != 4 or args.smoke_steps != 4):
        raise ValueError("formal requires workers4 and frozen smoke_steps default")
    data = BrainCropDataset(
        args.cache, expected_split="train", case_ids=case_ids
    )
    expected_count, expected_subjects = ARM_COUNTS[args.arm]
    if len(data) != expected_count:
        raise ValueError("cache-restricted arm count mismatch")
    if data.index.get("test_access") is not False:
        raise ValueError("training cache declares test access")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("data972 voxel-KD requires CUDA")
    torch.cuda.set_device(device)
    historical_checkpoint = torch.load(
        args.initial_checkpoint, map_location="cpu", weights_only=False
    )
    ema_checkpoint = torch.load(
        args.ema_teacher_checkpoint, map_location="cpu", weights_only=False
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    student = _model(historical_checkpoint, device)
    historical_teacher = _model(historical_checkpoint, device).eval().requires_grad_(False)
    ema_teacher = _model(ema_checkpoint, device).eval().requires_grad_(False)
    initial_state_sha = state_dict_sha256(student.state_dict())
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
        foreach=False,
        fused=False,
    )
    optimizer_spec = optimizer_contract(optimizer)
    supervised_loss = DiceCELoss(
        sigmoid=True,
        squared_pred=False,
        lambda_dice=1.0,
        lambda_ce=0.1,
    )
    sampling_weights = data.sampling_weights()
    if sampling_weights.numel() != expected_count:
        raise RuntimeError("sampling-weight population size mismatch")
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            sampling_weights,
            num_samples=args.sampler_draws,
            replacement=True,
            generator=generator,
        ),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    if len(loader) != STEPS_PER_EPOCH:
        raise RuntimeError("DataLoader must expose exactly94 optimizer steps")
    reload_loader = DataLoader(
        Subset(data, list(range(8))), batch_size=8, shuffle=False, num_workers=0
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data_role = {
        "role": args.arm,
        "train": case_view_metadata(args.train_case_ids),
        "cache": str(args.cache.resolve()),
        "cache_index_sha256": CACHE_INDEX_SHA256,
        "cases": len(data),
        "subjects": expected_subjects,
        "retired_selector_confirmation_reclassified_as_train": args.arm
        == "data972_candidate",
    }
    sampling_contract = {
        "population_count": expected_count,
        "weights_sha256": tensor_sha256(sampling_weights),
        "weight_source": "selected cached 128^3 label positive voxels",
        "generator_seed": args.seed,
        "sampler_num_samples": SAMPLER_DRAWS,
        "consumed_draws_per_epoch": CONSUMED_DRAWS,
        "dropped_draws_per_epoch": 4,
        "optimizer_steps_per_epoch": STEPS_PER_EPOCH,
        "replacement": True,
    }
    started_payload = {
        "schema": SCHEMA + ".started",
        "schema_version": 1,
        "status": "started_before_optimizer_step",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "arm": args.arm,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "initial_state_sha256": initial_state_sha,
        "data_role": data_role,
        "sampling_contract": sampling_contract,
        "optimizer_contract": optimizer_spec,
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    write_json_atomic(args.output_dir / "STARTED.json", started_payload)

    # Construction is complete; reset the training RNG exactly as in the archived KD runner.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()

    def train_step(batch: dict[str, Any], lr: float) -> dict[str, Any]:
        for group in optimizer.param_groups:
            group["lr"] = lr
        student.train()
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
        image, label = augment_batch(image, label, max_shift=8)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            history_logits = historical_teacher(image).segmentation_logits.float()
            ema_logits = ema_teacher(image).segmentation_logits.float()
            teacher_logits = (
                history_logits * (1.0 - TEACHER_EMA_WEIGHT)
                + ema_logits * TEACHER_EMA_WEIGHT
            )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            student_output = student(image)
            supervised = supervised_loss(student_output.segmentation_logits, label)
        kd = binary_voxel_kl(
            student_output.segmentation_logits.float(),
            teacher_logits,
            temperature=TEMPERATURE,
        )
        total = supervised + KD_WEIGHT * kd
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("non-finite total training loss")
        total.backward()
        teacher_gradients = [
            p.grad for model in (historical_teacher, ema_teacher) for p in model.parameters()
        ]
        if any(grad is not None for grad in teacher_gradients):
            raise RuntimeError("frozen teacher unexpectedly received gradient")
        grad_norm = nn.utils.clip_grad_norm_(
            student.parameters(), max_norm=5.0, error_if_nonfinite=True
        )
        optimizer.step()
        return {
            "total": float(total.detach()),
            "supervised": float(supervised.detach()),
            "kd": float(kd.detach()),
            "grad_norm_before_clip": float(grad_norm.detach()),
            "dice": dice_per_case(
                student_output.segmentation_logits.detach().float(), label
            ).cpu().tolist(),
            "case_ids": [str(value) for value in batch["case_id"]],
        }

    if args.mode == "smoke":
        iterator = iter(loader)
        step_seconds: list[float] = []
        steps: list[dict[str, Any]] = []
        lr = learning_rate_for_epoch(
            169, base_lr=args.base_lr, min_lr=args.min_lr,
            constant_epochs=100, total_epochs=200,
        )
        for index in range(args.smoke_steps):
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            record = train_step(next(iterator), lr)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - tick
            record["step"] = index + 1
            record["seconds"] = elapsed
            steps.append(record)
            if index > 0:
                step_seconds.append(elapsed)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        median_step = float(statistics.median(step_seconds))
        extrapolated = median_step * STEPS_PER_EPOCH * 32
        temp_path = args.output_dir / "smoke_checkpoint.pth"
        checkpoint = save_checkpoint(
            temp_path,
            model=student,
            args=args,
            global_epoch=169,
            learning_rate=lr,
            data_role=data_role,
            sampling_contract=sampling_contract,
            initial_state_sha256=initial_state_sha,
            optimizer_spec=optimizer_spec,
        )
        reload_diff = strict_reload_difference(student, temp_path, reload_loader, device)
        gates = {
            "four_steps_completed": len(steps) == 4,
            "all_losses_gradients_finite": all(
                np.isfinite(x[key])
                for x in steps
                for key in ("total", "supervised", "kd", "grad_norm_before_clip")
            ),
            "all_student_grad_norms_positive": all(
                x["grad_norm_before_clip"] > 0 for x in steps
            ),
            "teacher_gradients_absent": True,
            "peak_allocated_at_most_24gib": peak_allocated <= MAX_PEAK_BYTES,
            "formal_extrapolation_at_most_45min": extrapolated <= MAX_FORMAL_SECONDS,
            "strict_reload_exact": reload_diff == 0.0,
        }
        report = {
            "schema": SCHEMA + ".smoke",
            "schema_version": 1,
            "status": "passed" if all(gates.values()) else "failed",
            "arm": args.arm,
            "steps": steps,
            "median_measured_step_seconds": median_step,
            "extrapolated_formal_seconds": extrapolated,
            "peak_gpu_memory_bytes": peak_allocated,
            "peak_gpu_reserved_bytes": peak_reserved,
            "reload_max_abs_difference": reload_diff,
            "checkpoint": checkpoint,
            "gates": gates,
            "all_gates_passed": all(gates.values()),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "validation_access": False,
            "outer_selector_access": False,
            "confirmation_access": False,
            "test_access": False,
        }
        write_json_atomic(args.output_dir / "SMOKE.json", report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        if not report["all_gates_passed"]:
            raise RuntimeError("GPU smoke gates failed")
        return report

    history: list[dict[str, Any]] = []
    checkpoints: dict[str, dict[str, Any]] = {}
    for continuation_epoch in range(1, args.continuation_epochs + 1):
        global_epoch = args.start_global_epoch + continuation_epoch
        lr = learning_rate_for_epoch(
            global_epoch,
            base_lr=args.base_lr,
            min_lr=args.min_lr,
            constant_epochs=100,
            total_epochs=200,
        )
        epoch_steps: list[dict[str, Any]] = []
        epoch_case_ids: list[str] = []
        for batch in loader:
            record = train_step(batch, lr)
            epoch_steps.append(record)
            epoch_case_ids.extend(record.pop("case_ids"))
        if len(epoch_steps) != STEPS_PER_EPOCH or len(epoch_case_ids) != CONSUMED_DRAWS:
            raise RuntimeError("formal epoch draw/step count mismatch")
        epoch_record = {
            "continuation_epoch": continuation_epoch,
            "global_epoch": global_epoch,
            "learning_rate": lr,
            "train_total_loss": float(np.mean([x["total"] for x in epoch_steps])),
            "train_supervised_loss": float(
                np.mean([x["supervised"] for x in epoch_steps])
            ),
            "train_kd_loss": float(np.mean([x["kd"] for x in epoch_steps])),
            "train_dsc": float(
                np.mean([v for x in epoch_steps for v in x["dice"]])
            ),
            "grad_norm_before_clip_mean": float(
                np.mean([x["grad_norm_before_clip"] for x in epoch_steps])
            ),
            "optimizer_steps": len(epoch_steps),
            "consumed_draws": len(epoch_case_ids),
            "case_order_sha256": case_order_sha256(epoch_case_ids),
            "elapsed_seconds": time.time() - started,
        }
        history.append(epoch_record)
        if global_epoch in SAVE_GLOBAL_EPOCHS:
            path = args.output_dir / f"plain_128_{args.arm}_global_epoch{global_epoch}.pth"
            checkpoints[str(global_epoch)] = save_checkpoint(
                path,
                model=student,
                args=args,
                global_epoch=global_epoch,
                learning_rate=lr,
                data_role=data_role,
                sampling_contract=sampling_contract,
                initial_state_sha256=initial_state_sha,
                optimizer_spec=optimizer_spec,
            )
        write_json_atomic(
            args.output_dir / "progress.json",
            {"history": history, "checkpoints": checkpoints},
        )
        print(
            f"arm={args.arm} global_epoch={global_epoch}/200 "
            f"loss={epoch_record['train_total_loss']:.6f} "
            f"sup={epoch_record['train_supervised_loss']:.6f} "
            f"kd={epoch_record['train_kd_loss']:.6f} "
            f"dsc={epoch_record['train_dsc']:.6f}",
            flush=True,
        )
    reload_differences: dict[str, float] = {}
    for epoch in SAVE_GLOBAL_EPOCHS:
        path = Path(checkpoints[str(epoch)]["path"])
        current = _model(
            torch.load(path, map_location="cpu", weights_only=False), device
        )
        reload_differences[str(epoch)] = strict_reload_difference(
            current, path, reload_loader, device
        )
    elapsed = time.time() - started
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    gates = {
        "all_epochs_complete": len(history) == 32,
        "all_epoch_steps_94": all(x["optimizer_steps"] == 94 for x in history),
        "all_epoch_consumed_draws_752": all(
            x["consumed_draws"] == 752 for x in history
        ),
        "all_four_checkpoints_saved": set(checkpoints) == {"176", "184", "192", "200"},
        "all_checkpoint_reloads_exact": all(v == 0.0 for v in reload_differences.values()),
        "peak_allocated_at_most_24gib": peak_allocated <= MAX_PEAK_BYTES,
        "elapsed_at_most_45min": elapsed <= MAX_FORMAL_SECONDS,
    }
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete" if all(gates.values()) else "failed",
        "arm": args.arm,
        "checkpoints": checkpoints,
        "history": history,
        "reload_max_abs_difference_by_epoch": reload_differences,
        "initial_state_sha256": initial_state_sha,
        "data_role": data_role,
        "sampling_contract": sampling_contract,
        "optimizer_contract": optimizer_spec,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": peak_allocated,
        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    write_json_atomic(args.output_dir / "TRAIN_DONE.json", report)
    print(json.dumps({
        "arm": args.arm,
        "status": report["status"],
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": peak_allocated,
        "checkpoints": checkpoints,
    }, indent=2, sort_keys=True), flush=True)
    if not report["all_gates_passed"]:
        raise RuntimeError("formal training gates failed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--train-case-ids", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--ema-teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--idea", type=Path, default=ROOT / "archive/research_records/2026-08-21_voxel_kd_data972/IDEA.md")
    parser.add_argument("--prereg", type=Path, default=ROOT / "archive/research_records/2026-08-21_voxel_kd_data972/PREREG.md")
    parser.add_argument("--amendment", type=Path, default=ROOT / "archive/research_records/2026-08-21_voxel_kd_data972/AMENDMENT_001_MATCHED_CONTROL.md")
    parser.add_argument("--feasibility", type=Path, default=ROOT / "archive/research_records/2026-08-21_voxel_kd_data972/FEASIBILITY.json")
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
    parser.add_argument("--kd-weight", type=float, default=KD_WEIGHT)
    parser.add_argument("--sampler-draws", type=int, default=SAMPLER_DRAWS)
    parser.add_argument("--smoke-steps", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
