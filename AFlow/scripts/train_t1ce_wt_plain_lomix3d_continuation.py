#!/usr/bin/env python3
"""Preregistered three-decoder-stage LoMix-3D continuation and GPU smoke."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch
from monai.losses import DiceCELoss
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_lomix3d import (  # noqa: E402
    DECODER_RESOLUTIONS,
    MAP_NAMES,
    OPERATORS,
    SUBSETS,
    LoMix3DTrainingModel,
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


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_lomix3d_continuation"
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_lomix3d_checkpoint"
SMOKE_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_lomix3d_gpu_smoke"
HISTORICAL_SHA256 = "f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1"
TRAIN_MANIFEST_SHA256 = "a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16"
TRAIN_CACHE_INDEX_SHA256 = "d26b823a4715d9f3676d75cddabcf630180d48807352c23042be7faf240b622d"
PREREG_SHA256 = "82658b571ed427b8391ade2512b62c91169886b751d9ebb7910a3fec3c9638de"
AMENDMENT_001_SHA256 = "378a43cfc7505d069011be43e0f452c40a2d9c6ade69967bf11a17734dce2552"
AMENDMENT_002_SHA256 = "03638e77229f3529a8ad820fd108323c78f5b57f54094d3e333b9e17ff67a7c3"
SAVE_GLOBAL_EPOCHS = (176, 184, 192, 200)
INITIAL_SOFTPLUS = float(torch.nn.functional.softplus(torch.zeros(())).item())
SOFTPLUS_COLLAPSE_THRESHOLD = 0.173286795
GPU_MEMORY_STOP_BYTES = 72 * 1024**3
EXPECTED_DEPLOY_PARAMETERS = 4_700_897


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _require_frozen_inputs(args: argparse.Namespace) -> None:
    if sha256_file(args.preregistration) != PREREG_SHA256:
        raise ValueError("LoMix3D preregistration SHA-256 mismatch")
    if sha256_file(args.preregistration_amendment_001) != AMENDMENT_001_SHA256:
        raise ValueError("LoMix3D preregistration amendment 001 SHA-256 mismatch")
    if sha256_file(args.preregistration_amendment_002) != AMENDMENT_002_SHA256:
        raise ValueError("LoMix3D preregistration amendment 002 SHA-256 mismatch")
    if sha256_file(args.initial_checkpoint) != HISTORICAL_SHA256:
        raise ValueError("historical epoch168 checkpoint SHA-256 mismatch")
    if sha256_file(args.train_case_ids) != TRAIN_MANIFEST_SHA256:
        raise ValueError("756-case train manifest SHA-256 mismatch")
    if sha256_file(args.cache / "index.json") != TRAIN_CACHE_INDEX_SHA256:
        raise ValueError("true-128 train cache index SHA-256 mismatch")
    if args.seed != 20260846 or args.batch_size != 8:
        raise ValueError("frozen LoMix3D protocol requires seed20260846/batch8")
    if args.start_global_epoch != 168 or args.continuation_epochs != 32:
        raise ValueError("frozen LoMix3D formal protocol is global epoch169-200")
    if args.base_lr != 2e-4 or args.min_lr != 1e-5 or args.weight_decay != 1e-5:
        raise ValueError("optimizer fields differ from frozen LoMix3D protocol")
    if args.max_shift != 8 or args.ce_weight != 0.1:
        raise ValueError("augmentation/loss differs from frozen LoMix3D protocol")


def _training_only_named_parameters(
    model: LoMix3DTrainingModel,
) -> Iterable[tuple[str, nn.Parameter]]:
    method = getattr(model, "training_only_named_parameters", None)
    if method is not None:
        yield from method()
        return
    for name, parameter in model.named_parameters():
        if any(token in name for token in ("aux", "concat", "attention", "alpha", "loss_weight")):
            yield name, parameter


def _gradient_audit(model: LoMix3DTrainingModel) -> dict[str, dict[str, float | bool]]:
    values: dict[str, dict[str, float | bool]] = {}
    for name, parameter in _training_only_named_parameters(model):
        gradient = parameter.grad
        finite = gradient is not None and bool(torch.isfinite(gradient).all())
        maximum = 0.0 if gradient is None else float(gradient.detach().abs().max())
        nonzero = gradient is not None and maximum > 0.0
        values[name] = {"finite": finite, "nonzero": nonzero, "max_abs": maximum}
    if not values:
        raise RuntimeError("no registered LoMix3D training-only parameters found")
    failed = [name for name, item in values.items() if not item["finite"] or not item["nonzero"]]
    if failed:
        raise RuntimeError(f"LoMix3D registered parameters failed gradient audit: {failed}")
    return values


def _weight_diagnostics(effective: dict[str, torch.Tensor]) -> dict[str, Any]:
    if tuple(effective) != tuple(MAP_NAMES):
        raise RuntimeError("LoMix3D effective-weight names/order changed")
    values = {name: float(weight.detach().float()) for name, weight in effective.items()}
    array = list(values.values())
    return {
        "effective_weights": values,
        "mean": float(np.mean(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "sum": float(np.sum(array)),
        "initial_softplus_zero": INITIAL_SOFTPLUS,
        "collapse_threshold": SOFTPLUS_COLLAPSE_THRESHOLD,
        "collapsed": float(np.mean(array)) < SOFTPLUS_COLLAPSE_THRESHOLD,
    }


def _current_effective_weights(
    model: LoMix3DTrainingModel,
) -> dict[str, torch.Tensor]:
    return {
        name: torch.nn.functional.softplus(model.loss_alphas[name])
        for name in MAP_NAMES
    }


@torch.inference_mode()
def _wrapper_reload_difference(
    left: LoMix3DTrainingModel,
    right: LoMix3DTrainingModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    left.eval()
    right.eval()
    maximum = 0.0
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            left_logits = left(image).final_logits
            right_logits = right(image).final_logits
        maximum = max(maximum, float((left_logits.float() - right_logits.float()).abs().max()))
    return maximum


def _checkpoint_value(
    model: LoMix3DTrainingModel,
    *,
    args: argparse.Namespace,
    global_epoch: int,
    learning_rate: float,
    data_role: dict[str, Any],
    sampling_contract: dict[str, Any],
    alpha_diagnostics: dict[str, Any],
    stripped_difference: float,
) -> dict[str, Any]:
    deploy_state = model.deploy_state_dict()
    deploy_parameters = sum(value.numel() for value in deploy_state.values())
    if deploy_parameters != EXPECTED_DEPLOY_PARAMETERS:
        raise RuntimeError("LoMix3D stripped deploy state parameter count changed")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "arm": "lomix3d",
        "continuation_epoch": global_epoch - args.start_global_epoch,
        "global_epoch": global_epoch,
        "resolution": 128,
        "learning_rate": learning_rate,
        "model_state_dict": deploy_state,
        "lomix_training_state_dict": model.state_dict(),
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
        "lomix": {
            "adaptation": "three decoder-stage raw-logit LoMix3D",
            "decoder_resolutions": list(DECODER_RESOLUTIONS),
            "subsets": [list(item) for item in SUBSETS],
            "operators": list(OPERATORS),
            "map_names": list(MAP_NAMES),
            "total_supervised_maps": len(MAP_NAMES),
            "teacher": None,
            "inference": "original final raw-logit head only",
        },
        "alpha_diagnostics": alpha_diagnostics,
        "stripped_deploy_max_abs_difference": stripped_difference,
        "data_role": data_role,
        "sampling_contract": sampling_contract,
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }


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


def _load_model(args: argparse.Namespace, device: torch.device) -> LoMix3DTrainingModel:
    checkpoint = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
    model = LoMix3DTrainingModel.from_plain_checkpoint(checkpoint).to(device)
    if sum(value.numel() for value in model.deploy_state_dict().values()) != EXPECTED_DEPLOY_PARAMETERS:
        raise RuntimeError("unexpected stripped deploy parameter count")
    return model


def _smoke(args: argparse.Namespace, data: VariableResolutionBrainCropDataset) -> dict[str, Any]:
    device = torch.device(args.device)
    write_json_atomic(
        args.output_dir / "SMOKE_STARTED.json",
        {
            "schema": SMOKE_SCHEMA + ".started",
            "schema_version": 1,
            "status": "started_before_optimizer_step",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "batch_size": args.batch_size,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "preregistration_sha256": PREREG_SHA256,
            "validation_access": False,
            "outer_selector_access": False,
            "confirmation_access": False,
            "test_access": False,
        },
    )
    model = _load_model(args, device)
    loss_function = DiceCELoss(
        sigmoid=True,
        squared_pred=False,
        lambda_dice=1.0,
        lambda_ce=args.ce_weight,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay
    )
    loader = DataLoader(
        Subset(data, list(range(args.batch_size))),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    started = time.time()
    batch = next(iter(loader))
    image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
    label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
    image, label = augment_batch(image, label, max_shift=args.max_shift)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(image)
        losses = model.lomix_loss(output, label, loss_function)
    if not torch.isfinite(losses.total):
        raise RuntimeError("LoMix3D GPU smoke produced non-finite loss")
    losses.total.backward()
    gradients = _gradient_audit(model)
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    torch.cuda.synchronize(device)
    elapsed = time.time() - started
    weights = _weight_diagnostics(losses.effective_weights)
    if weights["collapsed"]:
        raise RuntimeError("LoMix3D Softplus weights collapsed in GPU smoke")
    stripped_difference = model.deploy_max_abs_difference(image)
    if stripped_difference != 0.0:
        raise RuntimeError("LoMix3D stripped deployment logits differ in GPU smoke")
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    gates = {
        "finite_loss": True,
        "all_registered_training_parameters_have_finite_nonzero_gradient": True,
        "stripped_deploy_max_abs_difference_zero": stripped_difference == 0.0,
        "mean_softplus_above_collapse_threshold": not weights["collapsed"],
        "peak_allocated_at_most_72_gib": peak_allocated <= GPU_MEMORY_STOP_BYTES,
    }
    report = {
        "schema": SMOKE_SCHEMA,
        "schema_version": 1,
        "status": "complete" if all(gates.values()) else "failed_gate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "batch_size": args.batch_size,
        "cases": args.batch_size,
        "optimizer_steps": 1,
        "loss": float(losses.total.detach()),
        "per_map_loss": {name: float(value.detach()) for name, value in losses.per_map.items()},
        "alpha_diagnostics": weights,
        "gradient_audit": gradients,
        "stripped_deploy_max_abs_difference": stripped_difference,
        "elapsed_seconds": elapsed,
        "estimated_formal_seconds_from_one_step": elapsed * (756 // args.batch_size) * 32,
        "peak_gpu_memory_bytes": peak_allocated,
        "peak_gpu_reserved_bytes": peak_reserved,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "preregistration_sha256": PREREG_SHA256,
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    write_json_atomic(args.output_dir / "GPU_SMOKE.json", report)
    if not report["all_gates_passed"]:
        raise RuntimeError(f"LoMix3D GPU smoke failed gates: {gates}")
    return report


def _formal(args: argparse.Namespace, data: VariableResolutionBrainCropDataset) -> dict[str, Any]:
    device = torch.device(args.device)
    model = _load_model(args, device)
    loss_function = DiceCELoss(
        sigmoid=True,
        squared_pred=False,
        lambda_dice=1.0,
        lambda_ce=args.ce_weight,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay
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
        "weights_sha256": tensor_sha256(sampling_weights),
        "generator_seed": args.seed,
        "num_samples": len(data),
        "replacement": True,
    }
    write_json_atomic(
        args.output_dir / "STARTED.json",
        {
            "schema": SCHEMA,
            "schema_version": 1,
            "status": "started_before_optimizer_step",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "arm": "lomix3d",
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "preregistration_sha256": PREREG_SHA256,
            "preregistration_amendment_001_sha256": AMENDMENT_001_SHA256,
            "preregistration_amendment_002_sha256": AMENDMENT_002_SHA256,
            "data_role": data_role,
            "sampling_contract": sampling_contract,
            "hyperparameters": serializable_args(args),
            "validation_access": False,
            "outer_selector_access": False,
            "confirmation_access": False,
            "test_access": False,
        },
    )
    # Auxiliary-head construction consumes RNG. Reset here so the shared backbone's
    # dropout stream matches the archived control continuation.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    history: list[dict[str, Any]] = []
    checkpoints: dict[str, dict[str, Any]] = {}
    first_step_gradient_audit: dict[str, Any] | None = None
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
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        total_values: list[float] = []
        final_values: list[float] = []
        train_dsc: list[float] = []
        latest_weights: dict[str, torch.Tensor] | None = None
        for batch in loader:
            image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            image, label = augment_batch(image, label, max_shift=args.max_shift)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(image)
                losses = model.lomix_loss(output, label, loss_function)
            if not torch.isfinite(losses.total):
                raise RuntimeError(f"non-finite LoMix3D loss at global epoch {global_epoch}")
            losses.total.backward()
            if first_step_gradient_audit is None:
                first_step_gradient_audit = _gradient_audit(model)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_values.append(float(losses.total.detach()))
            final_values.append(float(losses.per_map["original_128"].detach()))
            train_dsc.extend(
                dice_per_case(output.final_logits.detach().float(), label).cpu().tolist()
            )
            latest_weights = losses.effective_weights
            last_image = image.detach()
        assert latest_weights is not None and last_image is not None
        # Record the weights after the final optimizer step of this epoch.
        alpha_diagnostics = _weight_diagnostics(_current_effective_weights(model))
        record = {
            "continuation_epoch": continuation_epoch,
            "global_epoch": global_epoch,
            "learning_rate": learning_rate,
            "train_total_loss": float(np.mean(total_values)),
            "train_final_map_loss": float(np.mean(final_values)),
            "train_dsc": float(np.mean(train_dsc)),
            "optimizer_steps": len(loader),
            "alpha_diagnostics": alpha_diagnostics,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        if global_epoch in SAVE_GLOBAL_EPOCHS:
            if alpha_diagnostics["collapsed"]:
                raise RuntimeError(
                    f"LoMix3D Softplus collapse gate failed at global epoch {global_epoch}"
                )
            stripped_difference = model.deploy_max_abs_difference(last_image)
            if stripped_difference != 0.0:
                raise RuntimeError("LoMix3D stripped deployment logits differ")
            value = _checkpoint_value(
                model,
                args=args,
                global_epoch=global_epoch,
                learning_rate=learning_rate,
                data_role=data_role,
                sampling_contract=sampling_contract,
                alpha_diagnostics=alpha_diagnostics,
                stripped_difference=stripped_difference,
            )
            path = args.output_dir / f"plain_128_lomix3d_global_epoch{global_epoch}.pth"
            torch.save(value, path)
            checkpoints[str(global_epoch)] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "global_epoch": global_epoch,
                "stripped_deploy_max_abs_difference": stripped_difference,
            }
        write_json_atomic(
            args.output_dir / "progress.json",
            {"history": history, "checkpoints": checkpoints},
        )
        print(
            f"arm=lomix3d global_epoch={global_epoch}/200 "
            f"loss={record['train_total_loss']:.6f} "
            f"final={record['train_final_map_loss']:.6f} "
            f"dsc={record['train_dsc']:.6f} "
            f"mean_w={alpha_diagnostics['mean']:.6f}",
            flush=True,
        )
    final_path = Path(checkpoints["200"]["path"])
    final_value = torch.load(final_path, map_location="cpu", weights_only=False)
    reloaded = LoMix3DTrainingModel.from_plain_checkpoint(final_value).to(device)
    reloaded.load_state_dict(final_value["lomix_training_state_dict"], strict=True)
    reload_difference = _wrapper_reload_difference(model, reloaded, reload_loader, device)
    if reload_difference != 0.0:
        raise RuntimeError("LoMix3D formal training-wrapper reload is not exact")
    stripped_difference = 0.0
    for batch in reload_loader:
        image = batch["image"].to(device=device, dtype=torch.float32)
        stripped_difference = max(
            stripped_difference, model.deploy_max_abs_difference(image)
        )
    if stripped_difference != 0.0:
        raise RuntimeError("LoMix3D formal stripped deployment reload is not exact")
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "arm": "lomix3d",
        "checkpoints": checkpoints,
        "history": history,
        "first_step_gradient_audit": first_step_gradient_audit,
        "reload_max_abs_difference": reload_difference,
        "stripped_deploy_max_abs_difference": stripped_difference,
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "preregistration_sha256": PREREG_SHA256,
        "preregistration_amendment_001_sha256": AMENDMENT_001_SHA256,
        "preregistration_amendment_002_sha256": AMENDMENT_002_SHA256,
        "data_role": data_role,
        "sampling_contract": sampling_contract,
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
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("LoMix3D smoke/formal training requires CUDA")
    torch.cuda.set_device(device)
    data = _make_dataset(args)
    if args.mode == "smoke":
        return _smoke(args, data)
    return _formal(args, data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--train-case-ids", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-amendment-001", type=Path, required=True)
    parser.add_argument("--preregistration-amendment-002", type=Path, required=True)
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
    print(json.dumps({
        "schema": result["schema"],
        "status": result["status"],
        "all_gates_passed": result.get("all_gates_passed"),
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
