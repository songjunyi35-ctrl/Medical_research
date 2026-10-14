#!/usr/bin/env python3
"""Preregistered DCD increment on the frozen 32-epoch voxel-KD continuation."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from monai.losses import DiceCELoss
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator  # noqa: E402
from sammed3d.t1ce_wt_dcd import (  # noqa: E402
    DB4_FILTER_BANK_SHA256,
    ENCODER_CHANNELS,
    ENCODER_RESOLUTIONS,
    LEVELS,
    SELECTED_BANDS,
    DCDTrainingModel,
)
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
from scripts.train_t1ce_wt_plain_voxel_kd_continuation import (  # noqa: E402
    binary_voxel_kl,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_dcd_continuation"
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_dcd_checkpoint"
SMOKE_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_dcd_gpu_smoke"
IMPLEMENTATION_RECEIPT_SCHEMA = "aflow.sammed3d.t1ce_wt_dcd_implementation_receipt"

HISTORICAL_SHA256 = "f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1"
EMA168_SHA256 = "4ea97dacb9a538993e1c16593bc56b03ce6dc3df2f457c1f0426439c6252197e"
DETAIL_TEACHER_SHA256 = "56f54898ded11a0208e75f8000d57d4d053e80d6dedd760402730ad345c111b7"
TRAIN_MANIFEST_SHA256 = "a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16"
TRAIN_CACHE_INDEX_SHA256 = "d26b823a4715d9f3676d75cddabcf630180d48807352c23042be7faf240b622d"
PREREG_SHA256 = "7d0a0420cd55f6abfda45bce37c440ca5ac7fd803bff37904846dcbbc0e7f6cf"
AMENDMENT_001_SHA256 = "c66e134fe5f909e8181f36158b85dc4e5d8a4787dbbd0850257e3fa0d874ecaa"
AMENDMENT_002_SHA256 = "85540dabd2d028e511c526f48d8d5793d59c59fefada1d3e2e7a2752b7ef54a4"
BASE_RUNNER_SHA256 = "e10964163c22253af9773d41e72f9dceefc52a27cbcd42c3779073fd2fc38120"
LR_DEPENDENCY_SHA256 = "43d8fdbeeb11983e6327e9347306d5271f4f7a2e82adb96f3661b2d93d5b763b"
TRAIN_DEPENDENCY_SHA256 = "d74ff2f0cf190c8a5458a70d19125d9a30603894d5f1799f192f0e60dd088fdb"
MODEL_DEPENDENCY_SHA256 = "276ee7c131e413cf7943dc29e5372c1e5bbbb85c780e27f030343c5ef8bf4cd0"
SAMPLING_WEIGHTS_SHA256 = "757e463d8e60f96ef6429613f6783fb3aa1014b580932959f9ed354d37072124"

TEACHER_EMA_WEIGHT = 0.25
TEMPERATURE = 2.0
VOXEL_KD_WEIGHT = 0.3
DCD_MAX_WEIGHT = 0.01
DCD_START_INDEX = 5
DCD_RAMP_EPOCHS = 26
SAVE_GLOBAL_EPOCHS = (176, 184, 192, 200)
EXPECTED_DCD_WEIGHTS = {
    176: 0.00014529091286973994,
    184: 0.0032269755647873224,
    192: 0.007840323733655778,
    200: 0.01,
}
EXPECTED_DEPLOY_PARAMETERS = 4_700_897
GPU_MEMORY_STOP_BYTES = 72 * 1024**3
SMOKE_FORMAL_SECONDS_STOP = 7200.0
FORMAL_STEPS = (756 // 8) * 32


def dcd_weight_for_global_epoch(global_epoch: int) -> float:
    """Frozen zero-based 32-epoch adaptation of official cosine healing."""

    local_index = int(global_epoch) - 169
    if local_index < 0 or local_index > 31:
        raise ValueError("DCD global epoch must be in [169,200]")
    if local_index < DCD_START_INDEX:
        return 0.0
    progress = min((local_index - DCD_START_INDEX) / DCD_RAMP_EPOCHS, 1.0)
    return DCD_MAX_WEIGHT * 0.5 * (1.0 - math.cos(math.pi * progress))


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _require_frozen_inputs(args: argparse.Namespace) -> None:
    checks = (
        (args.preregistration, PREREG_SHA256, "preregistration"),
        (args.preregistration_amendment_001, AMENDMENT_001_SHA256, "amendment001"),
        (args.preregistration_amendment_002, AMENDMENT_002_SHA256, "amendment002"),
        (args.initial_checkpoint, HISTORICAL_SHA256, "historical168"),
        (args.ema_teacher_checkpoint, EMA168_SHA256, "EMA168"),
        (args.detail_teacher_checkpoint, DETAIL_TEACHER_SHA256, "KD200 detail teacher"),
        (args.train_case_ids, TRAIN_MANIFEST_SHA256, "train manifest"),
        (args.cache / "index.json", TRAIN_CACHE_INDEX_SHA256, "train cache index"),
        (ROOT / "scripts/train_t1ce_wt_plain_voxel_kd_continuation.py", BASE_RUNNER_SHA256, "base KD runner"),
        (ROOT / "scripts/train_t1ce_wt_plain_long_schedule.py", LR_DEPENDENCY_SHA256, "LR dependency"),
        (ROOT / "scripts/train_t1ce_wt_capg3d.py", TRAIN_DEPENDENCY_SHA256, "train dependency"),
        (ROOT / "sammed3d/t1ce_wt_capg3d.py", MODEL_DEPENDENCY_SHA256, "model dependency"),
    )
    for path, expected, label in checks:
        if sha256_file(path) != expected:
            raise ValueError(f"{label} SHA-256 mismatch")
    if args.seed != 20260846 or args.batch_size != 8:
        raise ValueError("frozen DCD protocol requires seed20260846/batch8")
    if args.start_global_epoch != 168 or args.continuation_epochs != 32:
        raise ValueError("frozen DCD continuation is global epoch169-200")
    if args.base_lr != 2e-4 or args.min_lr != 1e-5 or args.weight_decay != 1e-5:
        raise ValueError("optimizer fields differ from frozen base voxel-KD")
    if args.max_shift != 8 or args.ce_weight != 0.1:
        raise ValueError("augmentation/loss differs from frozen base voxel-KD")
    for epoch, expected in EXPECTED_DCD_WEIGHTS.items():
        if dcd_weight_for_global_epoch(epoch) != expected:
            raise RuntimeError(f"DCD schedule off-by-one at global epoch {epoch}")


def _validate_implementation_receipt(path: Path | None) -> tuple[dict[str, Any], str]:
    if path is None:
        raise ValueError("formal DCD training requires --implementation-receipt")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != IMPLEMENTATION_RECEIPT_SCHEMA:
        raise ValueError("unexpected DCD implementation receipt schema")
    for field in (
        "synthetic_all_gates_passed",
        "gpu_smoke_all_gates_passed",
        "all_gates_passed",
    ):
        if value.get(field) is not True:
            raise ValueError(f"implementation receipt has {field} != true")
    for field in ("validation_access", "outer_selector_access", "confirmation_access", "test_access"):
        if value.get(field) is not False:
            raise ValueError(f"implementation receipt {field} must be false")
    implementation = value.get("implementation", {})
    if implementation.get("trainer_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("implementation receipt trainer SHA-256 mismatch")
    return value, sha256_file(path)


def _load_plain(path: Path, expected_sha: str, device: torch.device) -> AutomaticPromptLocator:
    if sha256_file(path) != expected_sha:
        raise ValueError(f"checkpoint SHA-256 mismatch: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    model = AutomaticPromptLocator(
        center_aware=False,
        init_filters=int(architecture["init_filters"]),
        dropout=float(architecture["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def _load_dcd_model(path: Path, expected_sha: str, device: torch.device) -> DCDTrainingModel:
    if sha256_file(path) != expected_sha:
        raise ValueError(f"checkpoint SHA-256 mismatch: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = DCDTrainingModel.from_plain_checkpoint(checkpoint).to(device)
    if sum(value.numel() for value in model.deploy_state_dict().values()) != EXPECTED_DEPLOY_PARAMETERS:
        raise RuntimeError("unexpected DCD deploy parameter count")
    return model


def _freeze_teacher(model: nn.Module) -> nn.Module:
    return model.eval().requires_grad_(False)


def _teacher_gradient_gate(teachers: Iterable[nn.Module]) -> bool:
    return all(parameter.grad is None for teacher in teachers for parameter in teacher.parameters())


def _make_dataset(args: argparse.Namespace) -> VariableResolutionBrainCropDataset:
    case_ids = read_case_ids(args.train_case_ids)
    assert case_ids is not None
    data = VariableResolutionBrainCropDataset(
        args.cache,
        expected_resolution=128,
        expected_manifest_sha256=TRAIN_MANIFEST_SHA256,
        case_ids=case_ids,
    )
    if len(data) != 756 or data.index.get("test_access") is not False:
        raise ValueError("unexpected full train cache")
    return data


def _construct_models(
    args: argparse.Namespace, device: torch.device
) -> tuple[DCDTrainingModel, AutomaticPromptLocator, AutomaticPromptLocator, DCDTrainingModel]:
    student = _load_dcd_model(args.initial_checkpoint, HISTORICAL_SHA256, device)
    history_teacher = _freeze_teacher(
        _load_plain(args.initial_checkpoint, HISTORICAL_SHA256, device)
    )
    ema_teacher = _freeze_teacher(
        _load_plain(args.ema_teacher_checkpoint, EMA168_SHA256, device)
    )
    detail_teacher = _freeze_teacher(
        _load_dcd_model(args.detail_teacher_checkpoint, DETAIL_TEACHER_SHA256, device)
    )
    return student, history_teacher, ema_teacher, detail_teacher


def _make_optimizer(model: DCDTrainingModel, args: argparse.Namespace) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
    )


def _make_supervised_loss(args: argparse.Namespace) -> DiceCELoss:
    return DiceCELoss(
        sigmoid=True,
        squared_pred=False,
        lambda_dice=1.0,
        lambda_ce=args.ce_weight,
    )


def _forward_losses(
    *,
    student: DCDTrainingModel,
    history_teacher: AutomaticPromptLocator,
    ema_teacher: AutomaticPromptLocator,
    detail_teacher: DCDTrainingModel,
    image: torch.Tensor,
    label: torch.Tensor,
    supervised_loss: DiceCELoss,
    dcd_weight: float,
) -> tuple[Any, torch.Tensor, torch.Tensor, Any, torch.Tensor, tuple[torch.Tensor, ...]]:
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        history_logits = history_teacher(image).segmentation_logits.float()
        ema_logits = ema_teacher(image).segmentation_logits.float()
        teacher_logits = (
            history_logits * (1.0 - TEACHER_EMA_WEIGHT)
            + ema_logits * TEACHER_EMA_WEIGHT
        )
        detail_features = detail_teacher.encode_features(image)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        student_output = student(image)
        supervised = supervised_loss(student_output.final_logits, label)
        dcd = student.detail_loss(student_output.encoder_features, detail_features)
    voxel_kd = binary_voxel_kl(
        student_output.final_logits.float(),
        teacher_logits,
        temperature=TEMPERATURE,
    )
    total = supervised + VOXEL_KD_WEIGHT * voxel_kd + float(dcd_weight) * dcd.total
    if not all(
        bool(torch.isfinite(value))
        for value in (supervised, voxel_kd, dcd.total, total, *dcd.stage_losses)
    ):
        raise RuntimeError("DCD training produced non-finite loss")
    return student_output, supervised, voxel_kd, dcd, total, detail_features


def _stage_gradient_audit(dcd: Any, student_features: tuple[torch.Tensor, ...]) -> dict[str, Any]:
    report: OrderedDict[str, Any] = OrderedDict()
    for index, (stage_loss, feature) in enumerate(zip(dcd.stage_losses, student_features, strict=True)):
        gradient = torch.autograd.grad(
            stage_loss,
            feature,
            retain_graph=True,
            allow_unused=True,
        )[0]
        finite = gradient is not None and bool(torch.isfinite(gradient).all())
        norm = 0.0 if gradient is None else float(gradient.detach().float().norm())
        report[f"stage{index}"] = {
            "channels": ENCODER_CHANNELS[index],
            "resolution": ENCODER_RESOLUTIONS[index],
            "finite": finite,
            "nonzero": gradient is not None and norm > 0.0,
            "norm": norm,
        }
    failed = [key for key, value in report.items() if not value["finite"] or not value["nonzero"]]
    if failed:
        raise RuntimeError(f"DCD stage-specific gradient audit failed: {failed}")
    return report


@torch.inference_mode()
def _wrapper_difference(
    left: DCDTrainingModel,
    right: DCDTrainingModel,
    image: torch.Tensor,
) -> float:
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if image.is_cuda else None
    left_mode, right_mode = left.training, right.training
    try:
        left.eval()
        right.eval()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            a = left(image).final_logits
            b = right(image).final_logits
        return float((a.float() - b.float()).abs().max())
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        left.train(left_mode)
        right.train(right_mode)


def _method_contract() -> dict[str, Any]:
    return {
        "wavelet": "db4",
        "levels": LEVELS,
        "selected_bands": list(SELECTED_BANDS),
        "filter_bank_sha256": DB4_FILTER_BANK_SHA256,
        "analysis_synthesis": "same frozen decomposition bank; PyTorch adjoint correction",
        "encoder_channels": list(ENCODER_CHANNELS),
        "encoder_resolutions": list(ENCODER_RESOLUTIONS),
        "level_aggregation": "mean",
        "stage_aggregation": "sum",
        "projection": None,
        "detail_teacher_sha256": DETAIL_TEACHER_SHA256,
        "student_ema": False,
        "inference": "plain student final raw logit only",
    }


def _checkpoint_value(
    *,
    student: DCDTrainingModel,
    args: argparse.Namespace,
    global_epoch: int,
    learning_rate: float,
    dcd_weight: float,
    data_role: dict[str, Any],
    sampling_contract: dict[str, Any],
    loss_diagnostics: dict[str, Any],
    stripped_difference: float,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "arm": "dcd",
        "global_epoch": global_epoch,
        "continuation_epoch": global_epoch - args.start_global_epoch,
        "resolution": 128,
        "learning_rate": learning_rate,
        "dcd_weight": dcd_weight,
        "model_state_dict": student.deploy_state_dict(cpu=True),
        "architecture": {
            "class": "AutomaticPromptLocator",
            "backbone": "MONAI SegResNet",
            "center_aware": False,
            "init_filters": 16,
            "dropout": 0.1,
            "input_channels": ["T1ce"],
            "outputs": ["WT segmentation logit"],
            "deploy_parameter_count": EXPECTED_DEPLOY_PARAMETERS,
        },
        "base_voxel_kd": {
            "teacher_formula": "0.75 historical168 logit + 0.25 EMA168 logit",
            "temperature": TEMPERATURE,
            "weight": VOXEL_KD_WEIGHT,
            "historical_teacher_sha256": HISTORICAL_SHA256,
            "ema_teacher_sha256": EMA168_SHA256,
        },
        "detail_teacher": {
            "checkpoint_sha256": DETAIL_TEACHER_SHA256,
            "selection_basis": "existing fixed validation epoch200 and history168 same-basin path",
            "historical_validation_access": True,
            "historical_test_access_after_selection": True,
        },
        "dcd": _method_contract(),
        "loss_diagnostics": loss_diagnostics,
        "stripped_deploy_max_abs_difference": stripped_difference,
        "checkpoint_reload_max_abs_difference": None,
        "data_role": data_role,
        "sampling_contract": sampling_contract,
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }


def _smoke(
    args: argparse.Namespace,
    data: VariableResolutionBrainCropDataset,
) -> dict[str, Any]:
    device = torch.device(args.device)
    write_json_atomic(
        args.output_dir / "SMOKE_STARTED.json",
        {
            "schema": SMOKE_SCHEMA + ".started",
            "status": "started_before_optimizer_step",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "smoke_dcd_weight": DCD_MAX_WEIGHT,
            "validation_access": False,
            "outer_selector_access": False,
            "confirmation_access": False,
            "test_access": False,
        },
    )
    student, history_teacher, ema_teacher, detail_teacher = _construct_models(args, device)
    optimizer = _make_optimizer(student, args)
    supervised_loss = _make_supervised_loss(args)
    loader = DataLoader(
        Subset(data, list(range(args.batch_size))),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    raw_batch = next(iter(loader))
    raw_image = raw_batch["image"].to(device=device, dtype=torch.float32)
    raw_label = raw_batch["label"].to(device=device, dtype=torch.float32)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    student.train()
    smoke_started = time.time()
    measured: list[float] = []
    latest: dict[str, Any] = {}
    stage_gradient_audit: dict[str, Any] | None = None
    for step in range(4):
        image, label = augment_batch(raw_image.clone(), raw_label.clone(), max_shift=args.max_shift)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        started = time.time()
        output, supervised, voxel_kd, dcd, total, _ = _forward_losses(
            student=student,
            history_teacher=history_teacher,
            ema_teacher=ema_teacher,
            detail_teacher=detail_teacher,
            image=image,
            label=label,
            supervised_loss=supervised_loss,
            dcd_weight=DCD_MAX_WEIGHT,
        )
        if stage_gradient_audit is None:
            stage_gradient_audit = _stage_gradient_audit(dcd, output.encoder_features)
        total.backward()
        if not _teacher_gradient_gate((history_teacher, ema_teacher, detail_teacher)):
            raise RuntimeError("a frozen DCD teacher received gradient")
        grad_norm = nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError("DCD smoke student gradient norm is non-finite")
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.time() - started
        if step > 0:
            measured.append(elapsed)
        latest = {
            "supervised": float(supervised.detach()),
            "voxel_kd": float(voxel_kd.detach()),
            "dcd_raw": float(dcd.total.detach()),
            "dcd_weighted": float((DCD_MAX_WEIGHT * dcd.total).detach()),
            "total": float(total.detach()),
            "stage_losses": [float(value.detach()) for value in dcd.stage_losses],
            "level_losses": [
                [float(value.detach()) for value in levels]
                for levels in dcd.level_losses
            ],
            "student_grad_norm_before_clip": float(grad_norm.detach()),
        }
    assert stage_gradient_audit is not None
    median_seconds = statistics.median(measured)
    extrapolated = median_seconds * FORMAL_STEPS
    stripped_difference = student.deploy_max_abs_difference(raw_image)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    gates = {
        "all_losses_finite": True,
        "four_stage_specific_dcd_gradients_finite_nonzero": all(
            item["finite"] and item["nonzero"] for item in stage_gradient_audit.values()
        ),
        "all_teacher_gradients_null": _teacher_gradient_gate(
            (history_teacher, ema_teacher, detail_teacher)
        ),
        "stripped_deploy_max_abs_difference_zero": stripped_difference == 0.0,
        "peak_allocated_at_most_72_gib": peak_allocated <= GPU_MEMORY_STOP_BYTES,
        "mechanical_formal_extrapolation_at_most_2h": extrapolated <= SMOKE_FORMAL_SECONDS_STOP,
    }
    report = {
        "schema": SMOKE_SCHEMA,
        "schema_version": 1,
        "status": "complete" if all(gates.values()) else "failed_gate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "batch_size": args.batch_size,
        "warmup_steps": 1,
        "measured_steps": 3,
        "measured_step_seconds": measured,
        "median_step_seconds": median_seconds,
        "estimated_formal_seconds": extrapolated,
        "elapsed_seconds": time.time() - smoke_started,
        "smoke_dcd_weight": DCD_MAX_WEIGHT,
        "losses": latest,
        "weighted_dcd_to_supervised_ratio": latest["dcd_weighted"] / latest["supervised"],
        "stage_gradient_audit": stage_gradient_audit,
        "stripped_deploy_max_abs_difference": stripped_difference,
        "peak_gpu_memory_bytes": peak_allocated,
        "peak_gpu_reserved_bytes": peak_reserved,
        "method": _method_contract(),
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "preregistration_sha256": PREREG_SHA256,
        "amendment_001_sha256": AMENDMENT_001_SHA256,
        "amendment_002_sha256": AMENDMENT_002_SHA256,
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    write_json_atomic(args.output_dir / "GPU_SMOKE.json", report)
    if not report["all_gates_passed"]:
        raise RuntimeError(f"DCD GPU smoke failed gates: {gates}")
    return report


def _formal(
    args: argparse.Namespace,
    data: VariableResolutionBrainCropDataset,
    implementation_receipt: dict[str, Any],
    implementation_receipt_sha256: str,
) -> dict[str, Any]:
    device = torch.device(args.device)
    student, history_teacher, ema_teacher, detail_teacher = _construct_models(args, device)
    optimizer = _make_optimizer(student, args)
    supervised_loss = _make_supervised_loss(args)
    sampling_weights = data.sampling_weights()
    if tensor_sha256(sampling_weights) != SAMPLING_WEIGHTS_SHA256:
        raise ValueError("sampling weights SHA-256 mismatch")
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            sampling_weights,
            num_samples=len(data),
            replacement=True,
            generator=generator,
        ),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    reload_loader = DataLoader(
        Subset(data, list(range(args.batch_size))),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    data_role = {
        "train": case_view_metadata(args.train_case_ids),
        "cases": len(data),
        "subjects": len({subject_id(case) for case in data.case_ids}),
        "cache_index_sha256": TRAIN_CACHE_INDEX_SHA256,
    }
    sampling_contract = {
        "weights_sha256": SAMPLING_WEIGHTS_SHA256,
        "generator_seed": args.seed,
        "num_samples": len(data),
        "replacement": True,
        "batch_size": args.batch_size,
        "drop_last": True,
        "steps_per_epoch": len(loader),
    }
    if len(loader) != 94:
        raise RuntimeError("frozen DCD loader must have 94 optimizer steps per epoch")
    write_json_atomic(
        args.output_dir / "STARTED.json",
        {
            "schema": SCHEMA,
            "schema_version": 1,
            "status": "started_before_optimizer_step",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "implementation_receipt_sha256": implementation_receipt_sha256,
            "preregistration_sha256": PREREG_SHA256,
            "amendment_001_sha256": AMENDMENT_001_SHA256,
            "amendment_002_sha256": AMENDMENT_002_SHA256,
            "method": _method_contract(),
            "data_role": data_role,
            "sampling_contract": sampling_contract,
            "hyperparameters": serializable_args(args),
            "detail_teacher_historical_validation_access": True,
            "detail_teacher_historical_test_access_after_selection": True,
            "validation_access": False,
            "outer_selector_access": False,
            "confirmation_access": False,
            "test_access": False,
        },
    )
    # Reset after every model, fixed wavelet module, optimizer and loader is built.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    history: list[dict[str, Any]] = []
    checkpoints: dict[str, Any] = {}
    started = time.time()
    last_image: torch.Tensor | None = None
    for continuation_epoch in range(1, args.continuation_epochs + 1):
        global_epoch = args.start_global_epoch + continuation_epoch
        learning_rate = learning_rate_for_epoch(
            global_epoch,
            base_lr=args.base_lr,
            min_lr=args.min_lr,
            constant_epochs=100,
            total_epochs=200,
        )
        weight = dcd_weight_for_global_epoch(global_epoch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        student.train()
        total_values: list[float] = []
        supervised_values: list[float] = []
        voxel_values: list[float] = []
        dcd_values: list[float] = []
        weighted_dcd_values: list[float] = []
        train_dsc: list[float] = []
        stage_values: list[list[float]] = [[] for _ in ENCODER_CHANNELS]
        level_values: list[list[list[float]]] = [
            [[] for _ in range(LEVELS)] for _ in ENCODER_CHANNELS
        ]
        for batch in loader:
            image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            image, label = augment_batch(image, label, max_shift=args.max_shift)
            optimizer.zero_grad(set_to_none=True)
            output, supervised, voxel_kd, dcd, total, _ = _forward_losses(
                student=student,
                history_teacher=history_teacher,
                ema_teacher=ema_teacher,
                detail_teacher=detail_teacher,
                image=image,
                label=label,
                supervised_loss=supervised_loss,
                dcd_weight=weight,
            )
            total.backward()
            if not _teacher_gradient_gate((history_teacher, ema_teacher, detail_teacher)):
                raise RuntimeError("a frozen teacher received gradient during formal DCD")
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()
            total_values.append(float(total.detach()))
            supervised_values.append(float(supervised.detach()))
            voxel_values.append(float(voxel_kd.detach()))
            dcd_values.append(float(dcd.total.detach()))
            weighted_dcd_values.append(float((weight * dcd.total).detach()))
            train_dsc.extend(
                dice_per_case(output.final_logits.detach().float(), label).cpu().tolist()
            )
            for stage_index, stage_loss in enumerate(dcd.stage_losses):
                stage_values[stage_index].append(float(stage_loss.detach()))
                for level_index, level_loss in enumerate(dcd.level_losses[stage_index]):
                    level_values[stage_index][level_index].append(float(level_loss.detach()))
            last_image = image.detach()
        assert last_image is not None
        loss_diagnostics = {
            "train_total_loss": float(np.mean(total_values)),
            "train_supervised_loss": float(np.mean(supervised_values)),
            "train_voxel_kd_loss": float(np.mean(voxel_values)),
            "train_dcd_raw_loss": float(np.mean(dcd_values)),
            "train_dcd_weighted_loss": float(np.mean(weighted_dcd_values)),
            "stage_raw_losses": [float(np.mean(values)) for values in stage_values],
            "level_raw_losses": [
                [float(np.mean(values)) for values in per_stage]
                for per_stage in level_values
            ],
        }
        record = {
            "continuation_epoch": continuation_epoch,
            "local_zero_based_index": global_epoch - 169,
            "global_epoch": global_epoch,
            "learning_rate": learning_rate,
            "dcd_weight": weight,
            **loss_diagnostics,
            "train_dsc": float(np.mean(train_dsc)),
            "optimizer_steps": len(loader),
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        if global_epoch in SAVE_GLOBAL_EPOCHS:
            if weight != EXPECTED_DCD_WEIGHTS[global_epoch]:
                raise RuntimeError("saved-epoch DCD weight differs from preregistration")
            stripped = student.deploy_max_abs_difference(last_image)
            if stripped != 0.0:
                raise RuntimeError("DCD stripped deploy logits differ")
            checkpoint = _checkpoint_value(
                student=student,
                args=args,
                global_epoch=global_epoch,
                learning_rate=learning_rate,
                dcd_weight=weight,
                data_role=data_role,
                sampling_contract=sampling_contract,
                loss_diagnostics=loss_diagnostics,
                stripped_difference=stripped,
            )
            path = args.output_dir / f"plain_128_dcd_global_epoch{global_epoch}.pth"
            torch.save(checkpoint, path)
            # Reload construction initializes a temporary network before strict
            # loading. Preserve RNG so save-epoch audits cannot perturb the
            # next epoch's dropout stream relative to archived base voxel-KD.
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all()
            try:
                loaded = torch.load(path, map_location="cpu", weights_only=False)
                reloaded = DCDTrainingModel.from_plain_checkpoint(loaded).to(device)
                reload_difference = _wrapper_difference(student, reloaded, last_image)
            finally:
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state_all(cuda_rng)
            if reload_difference != 0.0:
                raise RuntimeError(f"DCD checkpoint epoch{global_epoch} reload is not exact")
            loaded["checkpoint_reload_max_abs_difference"] = reload_difference
            torch.save(loaded, path)
            checkpoints[str(global_epoch)] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "global_epoch": global_epoch,
                "dcd_weight": weight,
                "stripped_deploy_max_abs_difference": stripped,
                "checkpoint_reload_max_abs_difference": reload_difference,
            }
            del reloaded
            torch.cuda.empty_cache()
        write_json_atomic(
            args.output_dir / "progress.json",
            {"history": history, "checkpoints": checkpoints},
        )
        print(
            f"arm=dcd global_epoch={global_epoch}/200 mu={weight:.9f} "
            f"loss={record['train_total_loss']:.6f} sup={record['train_supervised_loss']:.6f} "
            f"voxel={record['train_voxel_kd_loss']:.6f} dcd={record['train_dcd_raw_loss']:.6f} "
            f"dsc={record['train_dsc']:.6f}",
            flush=True,
        )
    if set(checkpoints) != {str(value) for value in SAVE_GLOBAL_EPOCHS}:
        raise RuntimeError("DCD formal training did not save the frozen checkpoint grid")
    all_reload_exact = all(
        float(value["checkpoint_reload_max_abs_difference"]) == 0.0
        for value in checkpoints.values()
    )
    all_stripped_exact = all(
        float(value["stripped_deploy_max_abs_difference"]) == 0.0
        for value in checkpoints.values()
    )
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "arm": "dcd",
        "checkpoints": checkpoints,
        "history": history,
        "reload_max_abs_difference": 0.0 if all_reload_exact else None,
        "stripped_deploy_max_abs_difference": 0.0 if all_stripped_exact else None,
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "implementation_receipt_sha256": implementation_receipt_sha256,
        "method": _method_contract(),
        "data_role": data_role,
        "sampling_contract": sampling_contract,
        "detail_teacher_historical_validation_access": True,
        "detail_teacher_historical_test_access_after_selection": True,
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    write_json_atomic(args.output_dir / "TRAIN_DONE.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    _require_frozen_inputs(args)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse DCD output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("DCD smoke/formal training requires CUDA")
    torch.cuda.set_device(device)
    data = _make_dataset(args)
    if args.mode == "smoke":
        return _smoke(args, data)
    receipt, receipt_sha = _validate_implementation_receipt(args.implementation_receipt)
    return _formal(args, data, receipt, receipt_sha)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--train-case-ids", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--ema-teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--detail-teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-amendment-001", type=Path, required=True)
    parser.add_argument("--preregistration-amendment-002", type=Path, required=True)
    parser.add_argument("--implementation-receipt", type=Path, default=None)
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
    parser.add_argument("--max-shift", type=int, default=8)
    parser.add_argument("--ce-weight", type=float, default=0.1)
    args = parser.parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result["status"],
                "all_gates_passed": result.get("all_gates_passed"),
                "elapsed_seconds": result.get("elapsed_seconds"),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
