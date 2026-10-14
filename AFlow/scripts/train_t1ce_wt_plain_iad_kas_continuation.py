#!/usr/bin/env python3
"""Paired teacher-free control/KAS continuation from frozen voxel-KD200."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
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
from sammed3d.t1ce_wt_iad_kas import (  # noqa: E402
    KAS_GAMMA,
    KASTrainingModel,
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


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_iad_kas_continuation"
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_iad_kas_checkpoint"
SMOKE_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_iad_kas_gpu_smoke"
IMPLEMENTATION_RECEIPT_SCHEMA = "aflow.sammed3d.t1ce_wt_iad_kas_implementation_receipt"

KD200_SHA256 = "56f54898ded11a0208e75f8000d57d4d053e80d6dedd760402730ad345c111b7"
TRAIN_MANIFEST_SHA256 = "a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16"
TRAIN_CACHE_INDEX_SHA256 = "d26b823a4715d9f3676d75cddabcf630180d48807352c23042be7faf240b622d"
SAMPLING_WEIGHTS_SHA256 = "757e463d8e60f96ef6429613f6783fb3aa1014b580932959f9ed354d37072124"
PREREG_SHA256 = "af4946913b31de2148ec11ecdf739823fcd31d2b0fff025ff19c346340b0edb0"
AMENDMENT_SHA256 = "124f319cebb3b37d542603da53adcb1bda8ab90e2b19573798c09c79efe30813"
PROJECTOR_INITIAL_SHA256 = "61b7c474d78ce3ea275e66e88f0ee1d02840417064cdd46fa438b33523e95fe1"
EXPECTED_DEPLOY_PARAMETERS = 4_700_897
SAVE_GLOBAL_EPOCHS = (208, 216, 224, 232)
START_GLOBAL_EPOCH = 200
CONTINUATION_EPOCHS = 32
FIXED_LR = 1e-5
GPU_MEMORY_STOP_BYTES = 32 * 1024**3


def tensor_raw_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def text_lines_sha256(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(str(value) for value in values).encode("utf-8")).hexdigest()


def cuda_rng_sha256(device: torch.device) -> str:
    return tensor_raw_sha256(torch.cuda.get_rng_state(device))


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _require_frozen_inputs(args: argparse.Namespace) -> None:
    for path, expected, label in (
        (args.preregistration, PREREG_SHA256, "preregistration"),
        (args.preregistration_amendment, AMENDMENT_SHA256, "amendment"),
        (args.initial_checkpoint, KD200_SHA256, "KD200 start checkpoint"),
        (args.train_case_ids, TRAIN_MANIFEST_SHA256, "train manifest"),
        (args.cache / "index.json", TRAIN_CACHE_INDEX_SHA256, "train cache index"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"{label} SHA-256 mismatch")
    if args.arm not in {"control", "kas"}:
        raise ValueError("arm must be control or kas")
    if args.seed != 20260860 or args.batch_size != 8:
        raise ValueError("frozen KAS protocol requires seed20260860/batch8")
    if args.start_global_epoch != START_GLOBAL_EPOCH or args.continuation_epochs != CONTINUATION_EPOCHS:
        raise ValueError("frozen KAS stage is project-global epoch201-232")
    if args.lr != FIXED_LR or args.weight_decay != 1e-5 or args.max_shift != 8:
        raise ValueError("KAS optimizer/augmentation fields differ from preregistration")
    if args.num_workers != 4:
        raise ValueError("paired KAS protocol freezes num_workers=4")


def _validate_implementation_receipt(path: Path | None) -> tuple[dict[str, Any], str]:
    if path is None:
        raise ValueError("formal KAS training requires --implementation-receipt")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != IMPLEMENTATION_RECEIPT_SCHEMA:
        raise ValueError("unexpected KAS implementation receipt schema")
    for field in ("synthetic_all_gates_passed", "gpu_smoke_all_gates_passed", "all_gates_passed"):
        if value.get(field) is not True:
            raise ValueError(f"implementation receipt has {field} != true")
    for field in ("validation_access", "outer_selector_access", "confirmation_access", "test_access"):
        if value.get(field) is not False:
            raise ValueError(f"implementation receipt {field} must be false")
    if value.get("implementation", {}).get("trainer_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("implementation receipt trainer SHA mismatch")
    return value, sha256_file(path)


def _load_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("global_epoch") != 200 or checkpoint.get("arm") != "kd":
        raise ValueError("start checkpoint is not frozen voxel-KD200")
    return checkpoint


def _load_arm_model(
    args: argparse.Namespace, checkpoint: Mapping[str, Any], device: torch.device
) -> AutomaticPromptLocator | KASTrainingModel:
    if args.arm == "control":
        architecture = checkpoint["architecture"]
        model = AutomaticPromptLocator(
            center_aware=False,
            init_filters=int(architecture["init_filters"]),
            dropout=float(architecture["dropout"]),
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return model.to(device)
    model = KASTrainingModel.from_plain_checkpoint(checkpoint).to(device)
    projector_state = OrderedDict(model.training_only_named_parameters())
    projector_raw = (
        projector_state["projector.weight"].detach().cpu().contiguous().numpy().tobytes()
        + projector_state["projector.bias"].detach().cpu().contiguous().numpy().tobytes()
    )
    if hashlib.sha256(projector_raw).hexdigest() != PROJECTOR_INITIAL_SHA256:
        raise RuntimeError("KAS projector initialization SHA mismatch")
    return model


def _deploy_state(model: AutomaticPromptLocator | KASTrainingModel) -> OrderedDict[str, torch.Tensor]:
    if isinstance(model, KASTrainingModel):
        return model.deploy_state_dict(cpu=True)
    return OrderedDict((name, value.detach().cpu().clone()) for name, value in model.state_dict().items())


def _make_optimizer(
    model: AutomaticPromptLocator | KASTrainingModel,
    args: argparse.Namespace,
) -> tuple[torch.optim.AdamW, list[nn.Parameter]]:
    if isinstance(model, KASTrainingModel):
        parameters = list(model.network.parameters()) + list(model.training_only_parameters())
    else:
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


def _supervised_loss() -> DiceCELoss:
    return DiceCELoss(
        sigmoid=True,
        squared_pred=False,
        lambda_dice=1.0,
        lambda_ce=0.1,
    )


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
        raise ValueError("unexpected paired KAS train cache")
    return data


def _make_loader(
    data: VariableResolutionBrainCropDataset,
    args: argparse.Namespace,
) -> tuple[DataLoader, torch.Tensor]:
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
        raise RuntimeError("paired KAS loader must have94 steps per epoch")
    return loader, weights


def _first_batch_pairing(
    batch: Mapping[str, Any],
    image: torch.Tensor,
    label: torch.Tensor,
    augmented_image: torch.Tensor,
    augmented_label: torch.Tensor,
) -> dict[str, Any]:
    case_ids = [str(value) for value in batch["case_id"]]
    return {
        "first_batch_case_ids": case_ids,
        "first_batch_case_ids_sha256": text_lines_sha256(case_ids),
        "first_batch_raw_image_sha256": tensor_raw_sha256(image),
        "first_batch_raw_label_sha256": tensor_raw_sha256(label),
        "first_batch_augmented_image_sha256": tensor_raw_sha256(augmented_image),
        "first_batch_augmented_label_sha256": tensor_raw_sha256(augmented_label),
        "first_batch_raw_image_dtype": str(image.dtype),
        "first_batch_raw_image_shape": list(image.shape),
        "first_batch_raw_label_dtype": str(label.dtype),
        "first_batch_raw_label_shape": list(label.shape),
        "first_batch_augmented_image_dtype": str(augmented_image.dtype),
        "first_batch_augmented_image_shape": list(augmented_image.shape),
        "first_batch_augmented_label_dtype": str(augmented_label.dtype),
        "first_batch_augmented_label_shape": list(augmented_label.shape),
        "tensor_hash_rule": "CPU contiguous C-order raw bytes; dtype and shape recorded by source tensor contract",
    }


def _forward_loss(
    model: AutomaticPromptLocator | KASTrainingModel,
    image: torch.Tensor,
    label: torch.Tensor,
    loss_function: DiceCELoss,
) -> tuple[torch.Tensor, torch.Tensor, Any | None, torch.Tensor]:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if isinstance(model, KASTrainingModel):
            output = model(image)
            supervised = loss_function(output.final_logits, label)
        else:
            output = model(image)
            supervised = loss_function(output.segmentation_logits, label)
    if isinstance(model, KASTrainingModel):
        kas = model.kas_loss(output)
        total = supervised + KAS_GAMMA * kas.total
        logits = output.final_logits
    else:
        kas = None
        total = supervised
        logits = output.segmentation_logits
    values = [supervised, total]
    if kas is not None:
        values.extend((kas.total, kas.mse, kas.dice, kas.contrastive))
    if any(not bool(torch.isfinite(value)) for value in values):
        raise RuntimeError("paired KAS arm produced non-finite loss")
    return logits, supervised, kas, total


def _kas_gradient_audit(
    model: KASTrainingModel,
    output: Any,
    kas: Any,
) -> dict[str, Any]:
    projector_parameters = list(model.training_only_parameters())
    targets = [output.deepest_feature, output.final_logits, *projector_parameters]
    gradients = torch.autograd.grad(
        kas.total,
        targets,
        retain_graph=True,
        allow_unused=True,
    )
    names = ["deepest_feature", "final_logits", "projector_weight", "projector_bias"]
    report: dict[str, Any] = {}
    for name, gradient in zip(names, gradients, strict=True):
        finite = gradient is not None and bool(torch.isfinite(gradient).all())
        norm = 0.0 if gradient is None else float(gradient.detach().float().norm())
        report[name] = {"finite": finite, "nonzero": gradient is not None and norm > 0.0, "norm": norm}
    failed = [name for name, item in report.items() if not item["finite"] or not item["nonzero"]]
    if failed:
        raise RuntimeError(f"KAS-only gradient audit failed: {failed}")
    return report


@torch.inference_mode()
def _plain_reload_difference(
    state: Mapping[str, torch.Tensor],
    architecture: Mapping[str, Any],
    reference_logits: torch.Tensor,
    image: torch.Tensor,
    device: torch.device,
) -> float:
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    try:
        deploy = AutomaticPromptLocator(
            center_aware=False,
            init_filters=int(architecture["init_filters"]),
            dropout=float(architecture["dropout"]),
        ).to(device).eval()
        deploy.load_state_dict(state, strict=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            reloaded = deploy(image).segmentation_logits
        return float((reference_logits.float() - reloaded.float()).abs().max())
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)


def _kas_contract(enabled: bool) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "method": "teacher-free KAS-on-voxel-KD" if enabled else "matched teacher-free DiceCE control",
        "gamma": KAS_GAMMA if enabled else 0.0,
        "weight": KAS_GAMMA if enabled else 0.0,
        "projector": "Conv3d128to1 kernel1 training-only" if enabled else None,
        "components": ["raw_mse", "sigmoid_dice_consistency", "per_volume_contrastive"] if enabled else [],
        "teacher_forwards": 0,
        "deploy": "plain student final raw logit only",
    }


def _architecture() -> dict[str, Any]:
    return {
        "class": "AutomaticPromptLocator",
        "backbone": "MONAI SegResNet",
        "center_aware": False,
        "init_filters": 16,
        "dropout": 0.1,
        "input_channels": ["T1ce"],
        "outputs": ["WT segmentation logit"],
        "deploy_parameter_count": EXPECTED_DEPLOY_PARAMETERS,
    }


def _smoke(
    args: argparse.Namespace,
    model: AutomaticPromptLocator | KASTrainingModel,
    optimizer: torch.optim.AdamW,
    parameters: list[nn.Parameter],
    loader: DataLoader,
    initial_state_sha: str,
    device: torch.device,
) -> dict[str, Any]:
    write_json_atomic(
        args.output_dir / "SMOKE_STARTED.json",
        {
            "schema": SMOKE_SCHEMA + ".started",
            "arm": args.arm,
            "status": "started_before_optimizer_step",
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
    loss_function = _supervised_loss()
    iterator = iter(loader)
    measured: list[float] = []
    first_pairing: dict[str, Any] | None = None
    gradient_audit: dict[str, Any] | None = None
    latest: dict[str, Any] = {}
    smoke_started = time.time()
    for step in range(4):
        batch = next(iterator)
        raw_image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        raw_label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
        image, label = augment_batch(raw_image.clone(), raw_label.clone(), max_shift=args.max_shift)
        if first_pairing is None:
            first_pairing = _first_batch_pairing(batch, raw_image, raw_label, image, label)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        started = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if isinstance(model, KASTrainingModel):
                output = model(image)
                supervised = loss_function(output.final_logits, label)
                logits = output.final_logits
            else:
                output = model(image)
                supervised = loss_function(output.segmentation_logits, label)
                logits = output.segmentation_logits
        if isinstance(model, KASTrainingModel):
            kas = model.kas_loss(output)
            if gradient_audit is None:
                gradient_audit = _kas_gradient_audit(model, output, kas)
            total = supervised + KAS_GAMMA * kas.total
        else:
            kas = None
            total = supervised
        total.backward()
        grad_norm = nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.time() - started
        if step > 0:
            measured.append(elapsed)
        latest = {
            "supervised": float(supervised.detach()),
            "kas_raw": 0.0 if kas is None else float(kas.total.detach()),
            "kas_weighted": 0.0 if kas is None else float((KAS_GAMMA * kas.total).detach()),
            "mse": 0.0 if kas is None else float(kas.mse.detach()),
            "dice_consistency": 0.0 if kas is None else float(kas.dice.detach()),
            "contrastive": 0.0 if kas is None else float(kas.contrastive.detach()),
            "total": float(total.detach()),
            "train_dsc": float(dice_per_case(logits.detach().float(), label).mean()),
            "grad_norm_before_clip": float(grad_norm.detach()),
        }
    assert first_pairing is not None
    median_seconds = statistics.median(measured)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    if isinstance(model, KASTrainingModel):
        stripped = model.deploy_max_abs_difference(raw_image)
    else:
        stripped = 0.0
    gates = {
        "losses_finite": all(np.isfinite(value) for value in latest.values()),
        "kas_gradient_gate": True if args.arm == "control" else gradient_audit is not None and all(
            item["finite"] and item["nonzero"] for item in gradient_audit.values()
        ),
        "peak_allocated_at_most_32_gib": peak_allocated <= GPU_MEMORY_STOP_BYTES,
        "stripped_deploy_max_abs_difference_zero": stripped == 0.0,
    }
    report = {
        "schema": SMOKE_SCHEMA,
        "schema_version": 1,
        "status": "complete" if all(gates.values()) else "failed_gate",
        "arm": args.arm,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "initial_student_state_sha256": initial_state_sha,
        "pairing": first_pairing,
        "warmup_steps": 1,
        "measured_steps": 3,
        "measured_step_seconds": measured,
        "median_step_seconds": median_seconds,
        "estimated_formal_seconds": median_seconds * 94 * 32,
        "losses": latest,
        "gradient_audit": gradient_audit,
        "peak_gpu_memory_bytes": peak_allocated,
        "peak_gpu_reserved_bytes": peak_reserved,
        "stripped_deploy_max_abs_difference": stripped,
        "kas": _kas_contract(args.arm == "kas"),
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "elapsed_seconds": time.time() - smoke_started,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    write_json_atomic(args.output_dir / "GPU_SMOKE.json", report)
    if not report["all_gates_passed"]:
        raise RuntimeError(f"paired {args.arm} smoke failed: {gates}")
    return report


def _checkpoint_value(
    *,
    args: argparse.Namespace,
    model: AutomaticPromptLocator | KASTrainingModel,
    global_epoch: int,
    loss_diagnostics: dict[str, Any],
    pairing: dict[str, Any],
    stripped: float,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "arm": args.arm,
        "global_epoch": global_epoch,
        "continuation_epoch": global_epoch - START_GLOBAL_EPOCH,
        "resolution": 128,
        "learning_rate": args.lr,
        "model_state_dict": _deploy_state(model),
        "architecture": _architecture(),
        "kas": _kas_contract(args.arm == "kas"),
        "loss_diagnostics": loss_diagnostics,
        "pairing": pairing,
        "checkpoint_reload_max_abs_difference": None,
        "stripped_deploy_max_abs_difference": stripped,
        "start_checkpoint_sha256": KD200_SHA256,
        "start_checkpoint_historical_validation_access": True,
        "start_checkpoint_historical_test_access_after_selection": True,
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    if isinstance(model, KASTrainingModel):
        value["kas_training_state_dict"] = OrderedDict(
            (name, tensor.detach().cpu().clone()) for name, tensor in model.state_dict().items()
        )
    return value


def _formal(
    args: argparse.Namespace,
    model: AutomaticPromptLocator | KASTrainingModel,
    optimizer: torch.optim.AdamW,
    parameters: list[nn.Parameter],
    loader: DataLoader,
    data: VariableResolutionBrainCropDataset,
    weights: torch.Tensor,
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
            "status": "started_before_optimizer_step",
            "arm": args.arm,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "initial_student_state_sha256": initial_state_sha,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "implementation_receipt_sha256": implementation_receipt_sha,
            "preregistration_sha256": PREREG_SHA256,
            "amendment_sha256": AMENDMENT_SHA256,
            "data_role": data_role,
            "hyperparameters": serializable_args(args),
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
    loss_function = _supervised_loss()
    history: list[dict[str, Any]] = []
    checkpoints: dict[str, Any] = {}
    epoch_case_sha: dict[str, str] = {}
    epoch_rng_before: dict[str, str] = {}
    epoch_rng_after: dict[str, str] = {}
    first_pairing: dict[str, Any] | None = None
    started = time.time()
    last_image: torch.Tensor | None = None
    for continuation_epoch in range(1, args.continuation_epochs + 1):
        global_epoch = args.start_global_epoch + continuation_epoch
        ordered_cases: list[str] = []
        total_values: list[float] = []
        supervised_values: list[float] = []
        kas_values: list[float] = []
        mse_values: list[float] = []
        dice_values: list[float] = []
        contrastive_values: list[float] = []
        train_dsc: list[float] = []
        for step, batch in enumerate(loader):
            raw_image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            raw_label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            if step == 0:
                epoch_rng_before[str(global_epoch)] = cuda_rng_sha256(device)
            image, label = augment_batch(raw_image.clone(), raw_label.clone(), max_shift=args.max_shift)
            if first_pairing is None:
                first_pairing = _first_batch_pairing(batch, raw_image, raw_label, image, label)
            ordered_cases.extend(str(value) for value in batch["case_id"])
            optimizer.zero_grad(set_to_none=True)
            logits, supervised, kas, total = _forward_loss(model, image, label, loss_function)
            total.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            optimizer.step()
            total_values.append(float(total.detach()))
            supervised_values.append(float(supervised.detach()))
            kas_values.append(0.0 if kas is None else float(kas.total.detach()))
            mse_values.append(0.0 if kas is None else float(kas.mse.detach()))
            dice_values.append(0.0 if kas is None else float(kas.dice.detach()))
            contrastive_values.append(0.0 if kas is None else float(kas.contrastive.detach()))
            train_dsc.extend(dice_per_case(logits.detach().float(), label).cpu().tolist())
            last_image = image.detach()
        epoch_case_sha[str(global_epoch)] = text_lines_sha256(ordered_cases)
        epoch_rng_after[str(global_epoch)] = cuda_rng_sha256(device)
        loss_diagnostics = {
            "train_total_loss": float(np.mean(total_values)),
            "train_supervised_loss": float(np.mean(supervised_values)),
            "train_kas_raw_loss": float(np.mean(kas_values)),
            "train_kas_weighted_loss": float(KAS_GAMMA * np.mean(kas_values)) if args.arm == "kas" else 0.0,
            "train_kas_mse": float(np.mean(mse_values)),
            "train_kas_dice_consistency": float(np.mean(dice_values)),
            "train_kas_contrastive": float(np.mean(contrastive_values)),
            "train_dsc": float(np.mean(train_dsc)),
        }
        record = {
            "global_epoch": global_epoch,
            "continuation_epoch": continuation_epoch,
            "learning_rate": args.lr,
            **loss_diagnostics,
            "optimizer_steps": len(loader),
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        pairing_prefix = {
            "initial_student_state_sha256": initial_state_sha,
            **(first_pairing or {}),
            "epoch_case_order_sha256": dict(epoch_case_sha),
            "epoch_cuda_rng_before_sha256": dict(epoch_rng_before),
            "epoch_cuda_rng_after_sha256": dict(epoch_rng_after),
        }
        if global_epoch in SAVE_GLOBAL_EPOCHS:
            assert last_image is not None
            if isinstance(model, KASTrainingModel):
                stripped = model.deploy_max_abs_difference(last_image)
                was_training = model.training
                model.eval()
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    reference_logits = model(last_image).final_logits.detach()
                model.train(was_training)
            else:
                stripped = 0.0
                model.eval()
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    reference_logits = model(last_image).segmentation_logits.detach()
                model.train()
            checkpoint = _checkpoint_value(
                args=args,
                model=model,
                global_epoch=global_epoch,
                loss_diagnostics=loss_diagnostics,
                pairing=pairing_prefix,
                stripped=stripped,
            )
            path = args.output_dir / f"plain_128_{args.arm}_global_epoch{global_epoch}.pth"
            torch.save(checkpoint, path)
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            reload_difference = _plain_reload_difference(
                loaded["model_state_dict"], loaded["architecture"], reference_logits, last_image, device
            )
            if reload_difference != 0.0 or stripped != 0.0:
                raise RuntimeError(f"{args.arm} epoch{global_epoch} reload/deploy mismatch")
            if args.arm == "kas":
                cpu_rng = torch.get_rng_state()
                cuda_rng = torch.cuda.get_rng_state_all()
                try:
                    wrapper = KASTrainingModel.from_plain_checkpoint(loaded).to(device)
                    wrapper.load_state_dict(loaded["kas_training_state_dict"], strict=True)
                    wrapper.eval()
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        wrapper_logits = wrapper(last_image).final_logits
                    full_difference = float((wrapper_logits.float() - reference_logits.float()).abs().max())
                finally:
                    torch.set_rng_state(cpu_rng)
                    torch.cuda.set_rng_state_all(cuda_rng)
                if full_difference != 0.0:
                    raise RuntimeError("KAS full training checkpoint reload mismatch")
                loaded["kas_training_reload_max_abs_difference"] = full_difference
            loaded["checkpoint_reload_max_abs_difference"] = reload_difference
            torch.save(loaded, path)
            checkpoints[str(global_epoch)] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "checkpoint_reload_max_abs_difference": reload_difference,
                "stripped_deploy_max_abs_difference": stripped,
            }
        write_json_atomic(
            args.output_dir / "progress.json",
            {"history": history, "checkpoints": checkpoints, "pairing": pairing_prefix},
        )
        print(
            f"arm={args.arm} global_epoch={global_epoch}/232 loss={record['train_total_loss']:.6f} "
            f"sup={record['train_supervised_loss']:.6f} kas={record['train_kas_raw_loss']:.6f} "
            f"dsc={record['train_dsc']:.6f}",
            flush=True,
        )
    assert first_pairing is not None
    if set(checkpoints) != {str(epoch) for epoch in SAVE_GLOBAL_EPOCHS}:
        raise RuntimeError(f"{args.arm} did not save the frozen checkpoint grid")
    pairing = {
        "initial_student_state_sha256": initial_state_sha,
        **first_pairing,
        "epoch_case_order_sha256": epoch_case_sha,
        "epoch_cuda_rng_before_sha256": epoch_rng_before,
        "epoch_cuda_rng_after_sha256": epoch_rng_after,
    }
    exact_reload = all(float(value["checkpoint_reload_max_abs_difference"]) == 0.0 for value in checkpoints.values())
    exact_deploy = all(float(value["stripped_deploy_max_abs_difference"]) == 0.0 for value in checkpoints.values())
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "arm": args.arm,
        "history": history,
        "checkpoints": checkpoints,
        "pairing": pairing,
        "reload_max_abs_difference": 0.0 if exact_reload else None,
        "stripped_deploy_max_abs_difference": 0.0 if exact_deploy else None,
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "initial_student_state_sha256": initial_state_sha,
        "kas": _kas_contract(args.arm == "kas"),
        "data_role": data_role,
        "sampling_contract": {
            "weights_sha256": tensor_raw_sha256(weights),
            "generator_seed": args.seed,
            "num_samples": len(data),
            "replacement": True,
            "batch_size": args.batch_size,
            "steps_per_epoch": len(loader),
        },
        "start_checkpoint_historical_validation_access": True,
        "start_checkpoint_historical_test_access_after_selection": True,
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
        raise FileExistsError(f"refusing to reuse {args.arm} output directory")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("paired KAS training requires CUDA")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint = _load_checkpoint(args)
    model = _load_arm_model(args, checkpoint, device)
    initial_state_sha = state_sha256(_deploy_state(model))
    optimizer, parameters = _make_optimizer(model, args)
    data = _make_dataset(args)
    loader, weights = _make_loader(data, args)
    if args.mode == "smoke":
        return _smoke(args, model, optimizer, parameters, loader, initial_state_sha, device)
    _, receipt_sha = _validate_implementation_receipt(args.implementation_receipt)
    return _formal(
        args, model, optimizer, parameters, loader, data, weights,
        initial_state_sha, receipt_sha, device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("control", "kas"), required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--train-case-ids", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-amendment", type=Path, required=True)
    parser.add_argument("--implementation-receipt", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260860)
    parser.add_argument("--start-global-epoch", type=int, default=200)
    parser.add_argument("--continuation-epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-shift", type=int, default=8)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "schema": result["schema"], "arm": result["arm"],
        "status": result["status"], "all_gates_passed": result.get("all_gates_passed"),
        "elapsed_seconds": result.get("elapsed_seconds"),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
