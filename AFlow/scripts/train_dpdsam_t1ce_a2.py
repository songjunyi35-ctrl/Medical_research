#!/usr/bin/env python3
"""Train the T1ce binary SAM-Med3D A2 spatial-prompt arm."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sammed3d import dpdsam_binary_model as binary_model  # noqa: E402
from sammed3d import engine  # noqa: E402
from sammed3d.t1ce_binary import (  # noqa: E402
    T1ceInferenceCase,
    T1ceSupervisedCase,
    load_t1ce_inference_view,
    load_t1ce_supervised_view,
    score_t1ce_dual_grid,
)
from scripts import train_dpdsam_t1ce_b0 as b0_runner  # noqa: E402


TRAINING_SCHEMA = "aflow.sammed3d.dpdsam_t1ce_binary_a2_training"
TRAINING_SCHEMA_VERSION = 1
BEST_CHECKPOINT_NAME = "best_model_grid_dsc.pth"


def validate_resume_payload(
    payload: Mapping[str, Any], implementation_manifest: Mapping[str, Any]
) -> None:
    if payload.get("training_schema") != TRAINING_SCHEMA:
        raise ValueError("resume checkpoint has an incompatible training schema")
    if payload.get("training_schema_version") != TRAINING_SCHEMA_VERSION:
        raise ValueError("resume checkpoint has an incompatible training schema version")
    expected_hash = b0_runner.canonical_json_sha256(implementation_manifest)
    if payload.get("implementation_manifest_sha256") != expected_hash:
        raise ValueError("resume checkpoint implementation manifest hash mismatch")
    if payload.get("implementation_manifest") != implementation_manifest:
        raise ValueError("resume checkpoint run configuration mismatch")
    model_manifest = implementation_manifest.get("model")
    if not isinstance(model_manifest, Mapping):
        raise ValueError("implementation manifest lacks model metadata")
    if payload.get("adapter_config") != model_manifest.get("adapter_config"):
        raise ValueError("resume checkpoint adapter architecture mismatch")
    if payload.get("dpdsam_binary_config") != model_manifest:
        raise ValueError("resume checkpoint model metadata mismatch")
    required = {
        "model_state_dict",
        "a2_spatial_prompt_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "epoch",
        "history",
        "best",
        "rng_state",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"resume checkpoint is missing fields: {sorted(missing)}")


def restore_training_checkpoint(
    payload: Mapping[str, Any],
    *,
    model: binary_model.SpatialAutomaticPromptBinarySAM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    generator: torch.Generator,
    implementation_manifest: Mapping[str, Any],
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    validate_resume_payload(payload, implementation_manifest)
    model.sam_model.load_state_dict(payload["model_state_dict"], strict=True)
    model.spatial_prompt_generator.load_state_dict(
        payload["a2_spatial_prompt_state_dict"], strict=True
    )
    optimizer_manifest = implementation_manifest.get("optimizer")
    if not isinstance(optimizer_manifest, Mapping):
        raise ValueError("implementation manifest lacks optimizer metadata")
    saved_groups = payload["optimizer_state_dict"].get("param_groups", [])
    if len(saved_groups) != len(optimizer.param_groups):
        raise ValueError("resume optimizer parameter-group count mismatch")
    for group in saved_groups:
        if float(group["lr"]) != float(optimizer_manifest["learning_rate"]):
            raise ValueError("resume optimizer learning-rate mismatch")
        if float(group["weight_decay"]) != float(optimizer_manifest["weight_decay"]):
            raise ValueError("resume optimizer weight-decay mismatch")
        if list(group["betas"]) != list(optimizer_manifest["betas"]):
            raise ValueError("resume optimizer beta mismatch")
        if float(group["eps"]) != float(optimizer_manifest["epsilon"]):
            raise ValueError("resume optimizer epsilon mismatch")
        if bool(group["amsgrad"]) != bool(optimizer_manifest["amsgrad"]):
            raise ValueError("resume optimizer amsgrad mismatch")
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    saved_scaler = payload["scaler_state_dict"]
    if (saved_scaler is None) != (scaler is None):
        raise ValueError("resume checkpoint AMP scaler contract mismatch")
    if scaler is not None:
        scaler.load_state_dict(saved_scaler)
    epoch = int(payload["epoch"])
    history = payload["history"]
    best = payload["best"]
    if epoch < 0 or not isinstance(history, list) or len(history) != epoch:
        raise ValueError("resume checkpoint epoch/history contract is invalid")
    if not isinstance(best, dict):
        raise ValueError("resume checkpoint best record must be a dict")
    b0_runner.restore_rng_state(payload["rng_state"], generator)
    return epoch, list(history), dict(best)


def train_epoch(
    model: binary_model.SpatialAutomaticPromptBinarySAM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    model.train()
    model.sam_model.prompt_encoder.eval()
    totals = {"loss": 0.0, "bce": 0.0, "dice_loss": 0.0, "samples": 0.0}
    iterator = tqdm(loader, desc="train", leave=False)
    for batch in iterator:
        image = batch["image"].to(device, non_blocking=True).float()
        target = batch["target"].to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            "cuda",
            enabled=amp_dtype is not None and device.type == "cuda",
            dtype=amp_dtype or torch.float16,
        ):
            output = model(image)
            loss, parts = b0_runner.dice_bce_loss(output.logits, target)
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        size = int(image.shape[0])
        totals["loss"] += float(loss.detach()) * size
        totals["bce"] += parts["bce"] * size
        totals["dice_loss"] += parts["dice_loss"] * size
        totals["samples"] += size
        iterator.set_postfix(loss=f"{totals['loss'] / totals['samples']:.4f}")
    samples = max(totals.pop("samples"), 1.0)
    return {name: value / samples for name, value in totals.items()}


@torch.no_grad()
def validate(
    model: binary_model.SpatialAutomaticPromptBinarySAM,
    rows: list[dict],
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict[str, Any]:
    model.eval()
    cases: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="validation", leave=False):
        supervised_case = T1ceSupervisedCase.from_manifest_row(row)
        inference_case = T1ceInferenceCase(
            case_id=supervised_case.case_id,
            t1ce_path=supervised_case.t1ce_path,
        )
        inference_view = load_t1ce_inference_view(inference_case)
        image = inference_view.image.unsqueeze(0).to(device, non_blocking=True).float()
        with torch.amp.autocast(
            "cuda",
            enabled=amp_dtype is not None and device.type == "cuda",
            dtype=amp_dtype or torch.float16,
        ):
            probability = model(image).probabilities
        prediction = probability[0, 0].float().cpu() >= 0.5
        supervised_view = load_t1ce_supervised_view(supervised_case)
        if inference_view.geometry != supervised_view.geometry:
            raise RuntimeError(f"validation geometry changed for {inference_case.case_id}")
        dual = score_t1ce_dual_grid(prediction, supervised_view).to_dict()
        dual["headline_metric"] = "model_grid.dsc"
        cases.append({"case_id": inference_case.case_id, **dual})
    return b0_runner.summarize_dual_grid_cases(cases)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--validation-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path(engine.DEFAULT_CKPT))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--encoder-qv-lora-rank", type=int, default=0)
    parser.add_argument("--encoder-qv-lora-alpha", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--subset-seed", type=int, default=20260721)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--max-train-cases", type=int, default=0)
    parser.add_argument("--max-validation-cases", type=int, default=0)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--no-preload", action="store_true")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.validation_interval <= 0:
        raise ValueError("epochs, batch size, and validation interval must be positive")
    if args.encoder_qv_lora_rank < 0 or args.encoder_qv_lora_alpha <= 0:
        raise ValueError("LoRA rank must be nonnegative and alpha must be positive")
    latest_checkpoint = args.output_dir / "latest.pth"
    resume_payload: Mapping[str, Any] | None = None
    if args.resume is None:
        if args.output_dir.exists():
            raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=False)
    else:
        if not args.output_dir.is_dir():
            raise FileNotFoundError(f"resume output directory does not exist: {args.output_dir}")
        if args.resume.resolve() != latest_checkpoint.resolve():
            raise ValueError("--resume must point to OUTPUT_DIR/latest.pth")
        if not args.resume.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {args.resume}")
        loaded = torch.load(args.resume, map_location="cpu", weights_only=False)
        if not isinstance(loaded, Mapping):
            raise TypeError("resume checkpoint must contain a mapping")
        resume_payload = loaded

    b0_runner.seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[
        args.amp_dtype
    ]
    if device.type != "cuda":
        amp_dtype = None

    train_rows = b0_runner.read_rows(
        args.train_jsonl,
        args.max_train_cases,
        expected_split="train",
        subset_seed=args.subset_seed,
    )
    validation_rows = b0_runner.read_rows(
        args.validation_jsonl,
        args.max_validation_cases,
        expected_split="validation",
        subset_seed=args.subset_seed,
    )
    train_subjects = {str(row["subject_id"]) for row in train_rows}
    validation_subjects = {str(row["subject_id"]) for row in validation_rows}
    overlap = train_subjects & validation_subjects
    if overlap:
        raise RuntimeError(f"train/validation subject leakage: {sorted(overlap)[:5]}")

    model, model_metadata = binary_model.load_untrained_a2(
        args.checkpoint,
        lambda model_type: engine.sam_model_registry3D[model_type](checkpoint=None),
        device=device,
        model_type=engine.MODEL_TYPE,
        lora_rank=args.encoder_qv_lora_rank,
        lora_alpha=args.encoder_qv_lora_alpha,
    )
    trainable = binary_model.configure_a2_trainable(model)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda" and amp_dtype == torch.float16
        else None
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        b0_runner.T1ceTrainDataset(
            train_rows,
            augment=not args.no_augmentation,
            preload=not args.no_preload,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=b0_runner.seed_worker,
        generator=generator,
        persistent_workers=False,
    )
    configuration_args = {
        key: (str(value.resolve()) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
        if key not in {"output_dir", "resume"}
    }
    config = {
        "schema": TRAINING_SCHEMA,
        "schema_version": TRAINING_SCHEMA_VERSION,
        "args": configuration_args,
        "model": model_metadata,
        "data": {
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "train_manifest_sha256": b0_runner.sha256_file(args.train_jsonl),
            "validation_manifest_sha256": b0_runner.sha256_file(args.validation_jsonl),
            "split_manifest_sha256": b0_runner.sha256_file(args.split_manifest),
            "train_subjects": len(train_subjects),
            "validation_subjects": len(validation_subjects),
            "subset_algorithm": "sha256(subset_seed:case_id)_ascending",
        },
        "preprocessing": {
            "geometry": "whole_volume_exact_128cubed",
            "normalization": "source_t1ce_nonzero_population_zscore",
            "n4_bias_correction": "unavailable_not_applied",
            "train_preloaded_in_host_memory": not args.no_preload,
        },
        "augmentation": {
            "enabled": not args.no_augmentation,
            "random_affine": {"p": 0.5, "degrees": 15, "translation_voxels": 10},
            "random_elastic": {"p": 0.2, "control_points": 7, "max_displacement": 5},
            "random_noise": {"p": 0.15, "std_range": [0, 0.1]},
        },
        "loss": {"dice_weight": 1.0, "bce_weight": 1.0, "smooth": 1e-5},
        "optimizer": {
            "class": "torch.optim.AdamW",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "betas": list(optimizer.defaults["betas"]),
            "epsilon": optimizer.defaults["eps"],
            "amsgrad": optimizer.defaults["amsgrad"],
        },
        "resume": {
            "checkpoint": "OUTPUT_DIR/latest.pth_only",
            "rng": "python_numpy_torch_cpu_cuda_all_and_dataloader_generator",
            "persistent_workers": False,
            "restored_states": "sam_model_and_a2_spatial_prompt_generator",
        },
        "metrics": {
            "headline": "model_grid_case_macro_DSC_IoU_HD95",
            "model_grid": "128cubed_target_with_physical_model_spacing",
            "secondary": "source_grid_case_macro_DSC_IoU_HD95",
            "source_grid_restore": "nearest-exact",
            "empty_counts": "both_empty_and_one_empty_per_grid",
        },
        "selection": "maximum_validation_model_grid_macro_dsc",
        "metric_threshold": 0.5,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "source_checkpoint_sha256": b0_runner.sha256_file(args.checkpoint),
    }
    implementation_manifest_path = args.output_dir / "implementation_manifest.json"
    if resume_payload is None:
        b0_runner.atomic_json(config, implementation_manifest_path)
        start_epoch = 0
        history: list[dict[str, Any]] = []
        best: dict[str, Any] = {
            "dsc": -1.0,
            "epoch": -1,
            "checkpoint": BEST_CHECKPOINT_NAME,
            "grid": "model_grid",
            "metric": "dsc",
        }
    else:
        if not implementation_manifest_path.is_file():
            raise FileNotFoundError("resume output lacks implementation_manifest.json")
        saved_manifest = json.loads(
            implementation_manifest_path.read_text(encoding="utf-8")
        )
        if saved_manifest != config:
            raise ValueError("on-disk implementation manifest does not match current configuration")
        start_epoch, history, best = restore_training_checkpoint(
            resume_payload,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            generator=generator,
            implementation_manifest=config,
        )
        if start_epoch > args.epochs:
            raise ValueError(
                f"resume epoch {start_epoch} exceeds configured epochs {args.epochs}"
            )

    elapsed_offset = float(history[-1]["elapsed_seconds"]) if history else 0.0
    start = time.time()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, amp_dtype)
        validation_metrics = None
        if epoch % args.validation_interval == 0 or epoch == args.epochs:
            validation_metrics = validate(model, validation_rows, device, amp_dtype)
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": (
                {key: value for key, value in validation_metrics.items() if key != "case_metrics"}
                if validation_metrics is not None
                else None
            ),
            "elapsed_seconds": elapsed_offset + time.time() - start,
        }
        history.append(record)
        selection_improved = bool(
            validation_metrics is not None
            and b0_runner.validation_selection_score(validation_metrics) > best["dsc"]
        )
        if selection_improved:
            best = {
                "dsc": b0_runner.validation_selection_score(validation_metrics),
                "epoch": epoch,
                "checkpoint": BEST_CHECKPOINT_NAME,
                "grid": "model_grid",
                "metric": "dsc",
            }
        payload = binary_model.checkpoint_payload(
            model,
            adapter_config=model_metadata["adapter_config"],
            extra={
                "training_schema": TRAINING_SCHEMA,
                "training_schema_version": TRAINING_SCHEMA_VERSION,
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "history": history,
                "best": best,
                "implementation_manifest": config,
                "implementation_manifest_sha256": b0_runner.canonical_json_sha256(config),
                "rng_state": b0_runner.capture_rng_state(generator),
            },
        )
        if validation_metrics is not None:
            b0_runner.atomic_json(
                {"epoch": epoch, **validation_metrics},
                args.output_dir / f"validation_epoch_{epoch:03d}.json",
            )
            if selection_improved:
                b0_runner.atomic_torch_save(payload, args.output_dir / best["checkpoint"])
        b0_runner.atomic_torch_save(payload, latest_checkpoint)
        b0_runner.atomic_json(
            {
                "status": "running",
                "epoch": epoch,
                "history": history,
                "best": best,
                "implementation_manifest_sha256": b0_runner.canonical_json_sha256(config),
            },
            args.output_dir / "progress.json",
        )
        print(json.dumps(record, sort_keys=True), flush=True)
    b0_runner.atomic_json(
        {
            "status": "complete",
            "epoch": args.epochs,
            "history": history,
            "best": best,
            "implementation_manifest_sha256": b0_runner.canonical_json_sha256(config),
        },
        args.output_dir / "progress.json",
    )


if __name__ == "__main__":
    main()
