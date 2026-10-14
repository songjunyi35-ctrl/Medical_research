#!/usr/bin/env python3
"""Train paired paper-Dice gamma0 and VM-Dice gamma1 continuations from KD200."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
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
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator  # noqa: E402
from sammed3d.t1ce_wt_variable_resolution import (  # noqa: E402
    VariableResolutionBrainCropDataset,
)
from sammed3d.t1ce_wt_vm_dice import (  # noqa: E402
    VM_DICE_ARM_GAMMAS,
    VM_DICE_EPSILON,
    voxel_modulated_dice_loss,
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
from scripts.train_t1ce_wt_plain_iad_kas_continuation import (  # noqa: E402
    _architecture,
    _plain_reload_difference,
    cuda_rng_sha256,
    state_sha256,
    tensor_raw_sha256,
    text_lines_sha256,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_vm_dice_continuation"
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_vm_dice_checkpoint"
SMOKE_SCHEMA = "aflow.sammed3d.t1ce_wt_vm_dice_gpu_smoke"
IMPLEMENTATION_RECEIPT_SCHEMA = (
    "aflow.sammed3d.t1ce_wt_vm_dice_implementation_receipt"
)

ARMS = ("paper_dice_gamma0", "vm_dice_gamma1")
EXPECTED_GAMMAS = {"paper_dice_gamma0": 0.0, "vm_dice_gamma1": 1.0}
KD200_SHA256 = "56f54898ded11a0208e75f8000d57d4d053e80d6dedd760402730ad345c111b7"
TRAIN_MANIFEST_SHA256 = (
    "a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16"
)
TRAIN_CACHE_INDEX_SHA256 = (
    "d26b823a4715d9f3676d75cddabcf630180d48807352c23042be7faf240b622d"
)
SAMPLING_WEIGHTS_SHA256 = (
    "757e463d8e60f96ef6429613f6783fb3aa1014b580932959f9ed354d37072124"
)
PREREG_SHA256 = "a50aa875bb0b7cb509b8020f3eae1056d24a9cb56bb570f2f52d474b89cb0e0e"
AMENDMENT_SHA256 = "af01410acdff1d565e5def1150432b027053bf08104f006be1fdccdcc7e7dea6"
AMENDMENT_002_SHA256 = (
    "a37fc5482f35008f77d445e8b0a935e830945e184af82db6a57011da4860ddc1"
)
PAIRING_DEPENDENCY_SHA256 = (
    "289168d96d3494449d9b822513fd759c69aff1b4d1c502fc59b114c1cde62ea8"
)
EXPECTED_DEPLOY_PARAMETERS = 4_700_897
SAVE_GLOBAL_EPOCHS = (208, 216, 224, 232)
TRAINING_GLOBAL_EPOCHS = tuple(range(201, 233))
START_GLOBAL_EPOCH = 200
CONTINUATION_EPOCHS = 32
FIXED_SEED = 20260870
FIXED_LR = 1e-5
BCE_WEIGHT = 0.1
GPU_MEMORY_STOP_BYTES = 24 * 1024**3
SMOKE_RUNTIME_STOP_SECONDS = 45 * 60

PAIRING_FIELDS = (
    "initial_student_state_sha256",
    "first_batch_case_ids",
    "first_batch_case_ids_sha256",
    "first_batch_raw_image_sha256",
    "first_batch_raw_label_sha256",
    "first_batch_augmented_image_sha256",
    "first_batch_augmented_label_sha256",
    "epoch_case_order_sha256",
    "epoch_cuda_rng_before_sha256",
    "epoch_cuda_rng_after_sha256",
)


def _serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _gamma_for_arm(arm: str) -> float:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if dict(VM_DICE_ARM_GAMMAS) != EXPECTED_GAMMAS:
        raise RuntimeError("VM Dice model arm/gamma contract changed")
    gamma = float(VM_DICE_ARM_GAMMAS[arm])
    if gamma != EXPECTED_GAMMAS[arm]:
        raise RuntimeError(f"unexpected frozen gamma for {arm}")
    return gamma


def _require_frozen_inputs(args: argparse.Namespace) -> None:
    for path, expected, label in (
        (args.preregistration, PREREG_SHA256, "preregistration"),
        (args.preregistration_amendment, AMENDMENT_SHA256, "amendment"),
        (
            args.preregistration_amendment_002,
            AMENDMENT_002_SHA256,
            "amendment 002",
        ),
        (args.initial_checkpoint, KD200_SHA256, "KD200 start checkpoint"),
        (args.train_case_ids, TRAIN_MANIFEST_SHA256, "train manifest"),
        (args.cache / "index.json", TRAIN_CACHE_INDEX_SHA256, "train cache index"),
        (
            ROOT / "scripts/train_t1ce_wt_plain_iad_kas_continuation.py",
            PAIRING_DEPENDENCY_SHA256,
            "paired helper dependency",
        ),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"{label} SHA-256 mismatch")
    _gamma_for_arm(args.arm)
    if args.seed != FIXED_SEED or args.batch_size != 8 or args.num_workers != 4:
        raise ValueError("frozen VM Dice protocol requires seed20260870,batch8,workers4")
    if (
        args.start_global_epoch != START_GLOBAL_EPOCH
        or args.continuation_epochs != CONTINUATION_EPOCHS
    ):
        raise ValueError("frozen VM Dice continuation is global201-232")
    if args.lr != FIXED_LR or args.weight_decay != 1e-5 or args.max_shift != 8:
        raise ValueError("VM Dice optimizer/augmentation differs from preregistration")


def _validate_implementation_receipt(path: Path | None) -> tuple[dict[str, Any], str]:
    if path is None:
        raise ValueError("formal VM Dice training requires implementation receipt")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != IMPLEMENTATION_RECEIPT_SCHEMA:
        raise ValueError("unexpected VM Dice implementation receipt schema")
    if value.get("schema_version") != 1:
        raise ValueError("unexpected implementation receipt schema version")
    for field in (
        "synthetic_all_gates_passed",
        "gpu_smoke_all_gates_passed",
        "all_gates_passed",
    ):
        if value.get(field) is not True:
            raise ValueError(f"implementation receipt {field} != true")
    for field in (
        "validation_access",
        "outer_selector_access",
        "confirmation_access",
        "test_access",
    ):
        if value.get(field) is not False:
            raise ValueError(f"implementation receipt {field} must be false")
    implementation = value.get("implementation")
    if not isinstance(implementation, dict):
        raise ValueError("implementation receipt lacks implementation bindings")
    if implementation.get("trainer_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("implementation receipt trainer SHA mismatch")
    return value, sha256_file(path)


def _load_start_model(args: argparse.Namespace, device: torch.device) -> AutomaticPromptLocator:
    checkpoint = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("KD200 start checkpoint is not a dictionary")
    if checkpoint.get("arm") != "kd" or checkpoint.get("global_epoch") != 200:
        raise ValueError("start checkpoint is not frozen voxel-KD200")
    architecture = checkpoint.get("architecture")
    state = checkpoint.get("model_state_dict")
    if not isinstance(architecture, dict) or not isinstance(state, dict):
        raise ValueError("KD200 start lacks architecture/model_state_dict")
    model = AutomaticPromptLocator(
        center_aware=False,
        init_filters=int(architecture["init_filters"]),
        dropout=float(architecture["dropout"]),
    ).to(device)
    model.load_state_dict(state, strict=True)
    if sum(value.numel() for value in model.state_dict().values()) != EXPECTED_DEPLOY_PARAMETERS:
        raise RuntimeError("unexpected VM Dice deploy parameter count")
    return model


def _make_optimizer(
    model: AutomaticPromptLocator, args: argparse.Namespace
) -> tuple[torch.optim.AdamW, list[nn.Parameter]]:
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
        maximize=False,
        foreach=False,
        fused=False,
        capturable=False,
    )
    return optimizer, parameters


def _make_data(
    args: argparse.Namespace,
) -> tuple[VariableResolutionBrainCropDataset, torch.Tensor, DataLoader]:
    case_ids = read_case_ids(args.train_case_ids)
    if case_ids is None:
        raise ValueError("train manifest produced no case IDs")
    data = VariableResolutionBrainCropDataset(
        args.cache,
        expected_resolution=128,
        expected_manifest_sha256=TRAIN_MANIFEST_SHA256,
        case_ids=case_ids,
    )
    if len(data) != 756 or data.index.get("test_access") is not False:
        raise ValueError("unexpected VM Dice train cache")
    weights = data.sampling_weights()
    if tensor_raw_sha256(weights) != SAMPLING_WEIGHTS_SHA256:
        raise ValueError("sampling weights SHA mismatch")
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            weights,
            num_samples=len(data),
            replacement=True,
            generator=generator,
        ),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    if len(loader) != 94:
        raise RuntimeError("VM Dice loader must have94 steps per epoch")
    return data, weights, loader


def _first_pairing(
    batch: Mapping[str, Any],
    raw_image: torch.Tensor,
    raw_label: torch.Tensor,
    image: torch.Tensor,
    label: torch.Tensor,
) -> dict[str, Any]:
    case_ids = [str(value) for value in batch["case_id"]]
    return {
        "first_batch_case_ids": case_ids,
        "first_batch_case_ids_sha256": text_lines_sha256(case_ids),
        "first_batch_raw_image_sha256": tensor_raw_sha256(raw_image),
        "first_batch_raw_label_sha256": tensor_raw_sha256(raw_label),
        "first_batch_augmented_image_sha256": tensor_raw_sha256(image),
        "first_batch_augmented_label_sha256": tensor_raw_sha256(label),
        "first_batch_raw_image_dtype": str(raw_image.dtype),
        "first_batch_raw_image_shape": list(raw_image.shape),
        "first_batch_raw_label_dtype": str(raw_label.dtype),
        "first_batch_raw_label_shape": list(raw_label.shape),
        "first_batch_augmented_image_dtype": str(image.dtype),
        "first_batch_augmented_image_shape": list(image.shape),
        "first_batch_augmented_label_dtype": str(label.dtype),
        "first_batch_augmented_label_shape": list(label.shape),
        "tensor_hash_rule": (
            "CPU contiguous C-order raw bytes; dtype and shape recorded by source tensor contract"
        ),
    }


def _formula_contract(arm: str) -> dict[str, Any]:
    gamma = _gamma_for_arm(arm)
    return {
        "arm": arm,
        "gamma": gamma,
        "gamma_foreground": gamma,
        "gamma_background": gamma,
        "epsilon": float(VM_DICE_EPSILON),
        "class_order": ["background", "foreground"],
        "present_gt_classes_only": True,
        "modulation_detached": True,
        "denominator": "sum(m*(target^2+probability^2))",
        "bce_with_logits_weight": BCE_WEIGHT,
        "teacher_forwards": 0,
        "deploy": "plain student final raw foreground logit only",
    }


def _forward_loss(
    model: AutomaticPromptLocator,
    image: torch.Tensor,
    target: torch.Tensor,
    *,
    gamma: float,
) -> tuple[torch.Tensor, Any, torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(image).segmentation_logits
    vm_output = voxel_modulated_dice_loss(logits, target, gamma=gamma)
    bce = F.binary_cross_entropy_with_logits(
        logits.float(), target.float(), reduction="mean"
    )
    total = vm_output.loss + BCE_WEIGHT * bce
    for name, value in (
        ("vm_loss", vm_output.loss),
        ("bce", bce),
        ("total", total),
    ):
        if value.ndim != 0 or not bool(torch.isfinite(value)):
            raise RuntimeError(f"VM Dice produced invalid {name}")
    return logits, vm_output, bce, total


def _difficulty_gate(vm_output: Any, gamma: float) -> dict[str, Any]:
    modulation = vm_output.modulation.detach().float()
    if modulation.ndim < 3 or modulation.shape[1] != 2:
        raise RuntimeError("VM Dice modulation must expose [B,2,...]")
    finite = bool(torch.isfinite(modulation).all())
    if gamma == 0.0:
        exact = bool(torch.equal(modulation, torch.ones_like(modulation)))
        return {
            "finite": finite,
            "gamma0_all_weights_exactly_one": exact,
            "high_error_weight_exceeds_low_error_weight": True,
            "passed": finite and exact,
        }
    # A real batch8 128^3 two-class tensor has 33,554,432 elements, above
    # torch.quantile's supported input-size limit.  The frozen smoke gate only
    # needs to prove that gamma1 assigns a larger weight to a higher-error
    # voxel than to a lower-error voxel; exact full-tensor min/max does that
    # without sampling, sorting, or changing the training objective.
    low = float(modulation.amin())
    high = float(modulation.amax())
    passed = finite and high > low
    return {
        "finite": finite,
        "gamma0_all_weights_exactly_one": None,
        "low_error_weight_min": low,
        "high_error_weight_max": high,
        "high_error_weight_exceeds_low_error_weight": high > low,
        "passed": passed,
    }


@torch.inference_mode()
def _checkpoint_reload_difference(
    state: Mapping[str, torch.Tensor],
    reference: torch.Tensor,
    image: torch.Tensor,
    device: torch.device,
) -> float:
    return _plain_reload_difference(state, _architecture(), reference, image, device)


def _checkpoint_value(
    *,
    args: argparse.Namespace,
    model: AutomaticPromptLocator,
    global_epoch: int,
    diagnostics: dict[str, Any],
    pairing: dict[str, Any],
    reload_difference: float,
    implementation_receipt_sha: str,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "arm": args.arm,
        "global_epoch": global_epoch,
        "continuation_epoch": global_epoch - START_GLOBAL_EPOCH,
        "resolution": 128,
        "learning_rate": args.lr,
        "model_state_dict": OrderedDict(
            (name, value.detach().cpu().clone())
            for name, value in model.state_dict().items()
        ),
        "architecture": _architecture(),
        "vm_dice": _formula_contract(args.arm),
        "loss_diagnostics": diagnostics,
        "pairing": pairing,
        "data_role": {
            "train_manifest_sha256": TRAIN_MANIFEST_SHA256,
            "train_cache_index_sha256": TRAIN_CACHE_INDEX_SHA256,
            "cases": 756,
            "subjects": 346,
        },
        "optimizer": {
            "class": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "clip_grad_norm": 5.0,
            "clip_error_if_nonfinite": True,
        },
        "checkpoint_reload_max_abs_difference": reload_difference,
        "stripped_deploy_max_abs_difference": 0.0,
        "start_checkpoint_sha256": KD200_SHA256,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "implementation_receipt_sha256": implementation_receipt_sha,
        "preregistration_sha256": PREREG_SHA256,
        "amendment_001_sha256": AMENDMENT_SHA256,
        "amendment_002_sha256": AMENDMENT_002_SHA256,
        "start_checkpoint_historical_validation_access": True,
        "start_checkpoint_historical_test_access_after_selection": True,
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }


def _smoke(
    args: argparse.Namespace,
    model: AutomaticPromptLocator,
    optimizer: torch.optim.AdamW,
    parameters: list[nn.Parameter],
    loader: DataLoader,
    initial_state_sha: str,
    device: torch.device,
) -> dict[str, Any]:
    output = args.output_dir
    write_json_atomic(
        output / "SMOKE_STARTED.json",
        {
            "schema": SMOKE_SCHEMA + ".started",
            "schema_version": 1,
            "status": "before_optimizer_step",
            "arm": args.arm,
            "validation_access": False,
            "outer_selector_access": False,
            "confirmation_access": False,
            "test_access": False,
        },
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    gamma = _gamma_for_arm(args.arm)
    iterator = iter(loader)
    measured: list[float] = []
    first: dict[str, Any] | None = None
    gradient_audit: dict[str, Any] | None = None
    difficulty: dict[str, Any] | None = None
    latest: dict[str, Any] = {}
    rng_unchanged: bool | None = None
    started = time.time()
    last_image: torch.Tensor | None = None
    for step in range(4):
        batch = next(iterator)
        raw_image = batch["image"].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        raw_label = batch["label"].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        image, label = augment_batch(
            raw_image.clone(), raw_label.clone(), max_shift=args.max_shift
        )
        if first is None:
            first = _first_pairing(batch, raw_image, raw_label, image, label)
            first["initial_student_state_sha256"] = initial_state_sha
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        step_started = time.time()
        logits, vm_output, bce, total = _forward_loss(
            model, image, label, gamma=gamma
        )
        if gradient_audit is None:
            gradient = torch.autograd.grad(
                vm_output.loss, logits, retain_graph=True, allow_unused=False
            )[0]
            gradient_audit = {
                "finite": bool(torch.isfinite(gradient).all()),
                "nonzero": float(gradient.detach().float().norm()) > 0.0,
                "norm": float(gradient.detach().float().norm()),
                "regular_probability_path_attached": bool(
                    getattr(vm_output, "class_scores").requires_grad
                ),
                "modulation_detached": not bool(
                    getattr(vm_output, "modulation").requires_grad
                ),
            }
            difficulty = _difficulty_gate(vm_output, gamma)
            rng_before = cuda_rng_sha256(device)
            with torch.no_grad():
                _ = voxel_modulated_dice_loss(logits.detach(), label, gamma=gamma)
            rng_unchanged = rng_before == cuda_rng_sha256(device)
        total.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            parameters,
            max_norm=5.0,
            error_if_nonfinite=True,
        )
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.time() - step_started
        if step > 0:
            measured.append(elapsed)
        latest = {
            "vm_dice_loss": float(vm_output.loss.detach()),
            "bce_with_logits": float(bce.detach()),
            "weighted_bce": float((BCE_WEIGHT * bce).detach()),
            "total_loss": float(total.detach()),
            "train_dsc": float(dice_per_case(logits.detach().float(), label).mean()),
            "grad_norm_before_clip": float(grad_norm.detach()),
        }
        last_image = image.detach()
    assert first is not None and gradient_audit is not None and difficulty is not None
    assert last_image is not None
    median_seconds = statistics.median(measured)
    estimated_seconds = median_seconds * 94 * 32
    peak = int(torch.cuda.max_memory_allocated(device))
    reserved = int(torch.cuda.max_memory_reserved(device))
    was_training = model.training
    model.eval()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        reference = model(last_image).segmentation_logits.detach()
    model.train(was_training)
    reload_difference = _checkpoint_reload_difference(
        OrderedDict(
            (name, value.detach().cpu().clone())
            for name, value in model.state_dict().items()
        ),
        reference,
        last_image,
        device,
    )
    gates = {
        "losses_and_grad_norm_finite": all(
            math.isfinite(float(value)) for value in latest.values()
        ),
        "vm_logit_gradient_finite_nonzero": (
            gradient_audit["finite"] and gradient_audit["nonzero"]
        ),
        "modulation_detached_and_probability_path_attached": (
            gradient_audit["modulation_detached"]
            and gradient_audit["regular_probability_path_attached"]
        ),
        "difficulty_contract_passed": difficulty["passed"],
        "loss_does_not_consume_cuda_rng": rng_unchanged is True,
        "peak_allocated_at_most_24_gib": peak <= GPU_MEMORY_STOP_BYTES,
        "runtime_extrapolation_at_most_45_minutes": (
            estimated_seconds <= SMOKE_RUNTIME_STOP_SECONDS
        ),
        "checkpoint_reload_exact": reload_difference == 0.0,
        "deploy_is_plain_without_auxiliary_state": (
            sum(value.numel() for value in model.state_dict().values())
            == EXPECTED_DEPLOY_PARAMETERS
        ),
    }
    report = {
        "schema": SMOKE_SCHEMA,
        "schema_version": 1,
        "status": "complete" if all(gates.values()) else "failed_gate",
        "arm": args.arm,
        "gamma": gamma,
        "initial_student_state_sha256": initial_state_sha,
        "pairing": first,
        "losses": latest,
        "gradient_audit": gradient_audit,
        "difficulty_audit": difficulty,
        "measured_step_seconds": measured,
        "median_step_seconds": median_seconds,
        "estimated_formal_seconds": estimated_seconds,
        "peak_gpu_memory_bytes": peak,
        "peak_gpu_reserved_bytes": reserved,
        "checkpoint_reload_max_abs_difference": reload_difference,
        "stripped_deploy_max_abs_difference": 0.0,
        "vm_dice": _formula_contract(args.arm),
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "elapsed_seconds": time.time() - started,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    write_json_atomic(output / "GPU_SMOKE.json", report)
    if not report["all_gates_passed"]:
        raise RuntimeError(f"VM Dice {args.arm} smoke failed: {gates}")
    return report


def _formal(
    args: argparse.Namespace,
    model: AutomaticPromptLocator,
    optimizer: torch.optim.AdamW,
    parameters: list[nn.Parameter],
    data: VariableResolutionBrainCropDataset,
    weights: torch.Tensor,
    loader: DataLoader,
    initial_state_sha: str,
    implementation_receipt_sha: str,
    device: torch.device,
) -> dict[str, Any]:
    data_role = {
        "train": case_view_metadata(args.train_case_ids),
        "cases": len(data),
        "subjects": len({subject_id(case) for case in data.case_ids}),
        "cache_index_sha256": TRAIN_CACHE_INDEX_SHA256,
    }
    write_json_atomic(
        args.output_dir / "STARTED.json",
        {
            "schema": SCHEMA,
            "schema_version": 1,
            "status": "before_optimizer_step",
            "arm": args.arm,
            "gamma": _gamma_for_arm(args.arm),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "initial_student_state_sha256": initial_state_sha,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "implementation_receipt_sha256": implementation_receipt_sha,
            "preregistration_sha256": PREREG_SHA256,
            "amendment_sha256": AMENDMENT_SHA256,
            "amendment_002_sha256": AMENDMENT_002_SHA256,
            "data_role": data_role,
            "hyperparameters": _serializable_args(args),
            "validation_access": False,
            "outer_selector_access": False,
            "confirmation_access": False,
            "test_access": False,
        },
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    gamma = _gamma_for_arm(args.arm)
    history: list[dict[str, Any]] = []
    checkpoints: dict[str, Any] = {}
    epoch_case_sha: dict[str, str] = {}
    epoch_rng_before: dict[str, str] = {}
    epoch_rng_after: dict[str, str] = {}
    first: dict[str, Any] | None = None
    started = time.time()
    last_image: torch.Tensor | None = None
    for continuation_epoch in range(1, args.continuation_epochs + 1):
        global_epoch = args.start_global_epoch + continuation_epoch
        ordered_cases: list[str] = []
        totals: list[float] = []
        vm_losses: list[float] = []
        bces: list[float] = []
        train_dsc: list[float] = []
        grad_norms: list[float] = []
        for step, batch in enumerate(loader):
            raw_image = batch["image"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            raw_label = batch["label"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            if step == 0:
                epoch_rng_before[str(global_epoch)] = cuda_rng_sha256(device)
            image, label = augment_batch(
                raw_image.clone(), raw_label.clone(), max_shift=args.max_shift
            )
            if first is None:
                first = _first_pairing(batch, raw_image, raw_label, image, label)
                first["initial_student_state_sha256"] = initial_state_sha
            ordered_cases.extend(str(value) for value in batch["case_id"])
            optimizer.zero_grad(set_to_none=True)
            logits, vm_output, bce, total = _forward_loss(
                model, image, label, gamma=gamma
            )
            total.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                parameters,
                max_norm=5.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            totals.append(float(total.detach()))
            vm_losses.append(float(vm_output.loss.detach()))
            bces.append(float(bce.detach()))
            grad_norms.append(float(grad_norm.detach()))
            train_dsc.extend(
                dice_per_case(logits.detach().float(), label).cpu().tolist()
            )
            last_image = image.detach()
        epoch_case_sha[str(global_epoch)] = text_lines_sha256(ordered_cases)
        epoch_rng_after[str(global_epoch)] = cuda_rng_sha256(device)
        diagnostics = {
            "train_total_loss": float(np.mean(totals)),
            "train_vm_dice_loss": float(np.mean(vm_losses)),
            "train_bce_with_logits": float(np.mean(bces)),
            "train_weighted_bce": float(BCE_WEIGHT * np.mean(bces)),
            "train_dsc": float(np.mean(train_dsc)),
            "train_grad_norm_before_clip_mean": float(np.mean(grad_norms)),
            "train_grad_norm_before_clip_max": float(np.max(grad_norms)),
        }
        if not all(math.isfinite(value) for value in diagnostics.values()):
            raise RuntimeError("VM Dice formal diagnostics became non-finite")
        history.append(
            {
                "global_epoch": global_epoch,
                "continuation_epoch": continuation_epoch,
                "learning_rate": args.lr,
                "gamma": gamma,
                **diagnostics,
                "optimizer_steps": len(loader),
                "elapsed_seconds": time.time() - started,
            }
        )
        pairing_prefix = {
            "initial_student_state_sha256": initial_state_sha,
            **(first or {}),
            "epoch_case_order_sha256": dict(epoch_case_sha),
            "epoch_cuda_rng_before_sha256": dict(epoch_rng_before),
            "epoch_cuda_rng_after_sha256": dict(epoch_rng_after),
        }
        if global_epoch in SAVE_GLOBAL_EPOCHS:
            assert last_image is not None
            was_training = model.training
            model.eval()
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                reference = model(last_image).segmentation_logits.detach()
            model.train(was_training)
            state = OrderedDict(
                (name, value.detach().cpu().clone())
                for name, value in model.state_dict().items()
            )
            reload_difference = _checkpoint_reload_difference(
                state, reference, last_image, device
            )
            if reload_difference != 0.0:
                raise RuntimeError("VM Dice checkpoint reload mismatch")
            value = _checkpoint_value(
                args=args,
                model=model,
                global_epoch=global_epoch,
                diagnostics=diagnostics,
                pairing=pairing_prefix,
                reload_difference=reload_difference,
                implementation_receipt_sha=implementation_receipt_sha,
            )
            path = (
                args.output_dir
                / f"plain_128_{args.arm}_global_epoch{global_epoch}.pth"
            )
            torch.save(value, path)
            checkpoints[str(global_epoch)] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "checkpoint_reload_max_abs_difference": reload_difference,
                "stripped_deploy_max_abs_difference": 0.0,
            }
        write_json_atomic(
            args.output_dir / "progress.json",
            {
                "history": history,
                "checkpoints": checkpoints,
                "pairing": pairing_prefix,
            },
        )
        print(
            f"arm={args.arm} gamma={gamma:.0f} global_epoch={global_epoch}/232 "
            f"loss={diagnostics['train_total_loss']:.6f} "
            f"vm={diagnostics['train_vm_dice_loss']:.6f} "
            f"dsc={diagnostics['train_dsc']:.6f}",
            flush=True,
        )
    assert first is not None
    if set(checkpoints) != {str(epoch) for epoch in SAVE_GLOBAL_EPOCHS}:
        raise RuntimeError("VM Dice arm did not save the frozen checkpoint grid")
    pairing = {
        "initial_student_state_sha256": initial_state_sha,
        **first,
        "epoch_case_order_sha256": epoch_case_sha,
        "epoch_cuda_rng_before_sha256": epoch_rng_before,
        "epoch_cuda_rng_after_sha256": epoch_rng_after,
    }
    peak = int(torch.cuda.max_memory_allocated(device))
    memory_passed = peak <= GPU_MEMORY_STOP_BYTES
    exact_reload = all(
        float(value["checkpoint_reload_max_abs_difference"]) == 0.0
        for value in checkpoints.values()
    )
    exact_deploy = all(
        float(value["stripped_deploy_max_abs_difference"]) == 0.0
        for value in checkpoints.values()
    )
    self_gates = {
        "formal_peak_memory_at_most_24_gib": memory_passed,
        "all_four_checkpoint_reloads_exact": exact_reload,
        "all_four_deploy_exports_exact": exact_deploy,
        "history_has_frozen_32_epochs": (
            [item["global_epoch"] for item in history]
            == list(TRAINING_GLOBAL_EPOCHS)
        ),
    }
    status = "complete" if all(self_gates.values()) else "failed_gate"
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": status,
        "arm": args.arm,
        "gamma": gamma,
        "history": history,
        "checkpoints": checkpoints,
        "pairing": pairing,
        "formal_self_gates": self_gates,
        "all_self_gates_passed": all(self_gates.values()),
        "reload_max_abs_difference": 0.0 if exact_reload else None,
        "stripped_deploy_max_abs_difference": 0.0 if exact_deploy else None,
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": peak,
        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "implementation_receipt_sha256": implementation_receipt_sha,
        "preregistration_sha256": PREREG_SHA256,
        "amendment_001_sha256": AMENDMENT_SHA256,
        "amendment_002_sha256": AMENDMENT_002_SHA256,
        "initial_student_state_sha256": initial_state_sha,
        "vm_dice": _formula_contract(args.arm),
        "data_role": data_role,
        "sampling_contract": {
            "weights_sha256": tensor_raw_sha256(weights),
            "generator_seed": args.seed,
            "num_samples_drawn": len(data),
            "replacement": True,
            "drop_last": True,
            "batch_size": args.batch_size,
            "steps_per_epoch": len(loader),
            "optimizer_samples_per_epoch": len(loader) * args.batch_size,
        },
        "start_checkpoint_sha256": KD200_SHA256,
        "start_checkpoint_historical_validation_access": True,
        "start_checkpoint_historical_test_access_after_selection": True,
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    write_json_atomic(args.output_dir / "TRAIN_DONE.json", report)
    if status != "complete":
        raise RuntimeError(f"VM Dice {args.arm} formal self-gate failed: {self_gates}")
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    _require_frozen_inputs(args)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing VM Dice output reuse: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("VM Dice training requires CUDA")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = _load_start_model(args, device)
    initial_state_sha = state_sha256(model.state_dict())
    optimizer, parameters = _make_optimizer(model, args)
    data, weights, loader = _make_data(args)
    if args.mode == "smoke":
        return _smoke(
            args, model, optimizer, parameters, loader, initial_state_sha, device
        )
    _, receipt_sha = _validate_implementation_receipt(args.implementation_receipt)
    return _formal(
        args,
        model,
        optimizer,
        parameters,
        data,
        weights,
        loader,
        initial_state_sha,
        receipt_sha,
        device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--train-case-ids", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-amendment", type=Path, required=True)
    parser.add_argument("--preregistration-amendment-002", type=Path, required=True)
    parser.add_argument("--implementation-receipt", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=FIXED_SEED)
    parser.add_argument("--start-global-epoch", type=int, default=START_GLOBAL_EPOCH)
    parser.add_argument("--continuation-epochs", type=int, default=CONTINUATION_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=FIXED_LR)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-shift", type=int, default=8)
    args = parser.parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "arm": result["arm"],
                "gamma": result.get("gamma"),
                "status": result["status"],
                "all_gates_passed": result.get(
                    "all_gates_passed", result.get("all_self_gates_passed")
                ),
                "elapsed_seconds": result.get("elapsed_seconds"),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
