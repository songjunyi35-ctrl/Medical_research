#!/usr/bin/env python3
"""Independent T1ce/WT last-four-block partial-unfreeze trainer.

The protocol-hash-anchored generic trainer remains untouched.  This entry
point reuses its stable metric/checkpoint utilities, but owns the optimizer,
finite-gradient checks, conservative early stopping, preregistration binding,
and strict checkpoint reload required by this one development-only arm.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d import engine, partial_unfreeze, peft  # noqa: E402
from sammed3d.paths import SAMMED3D_CHECKPOINT  # noqa: E402
from sammed3d.t1ce_wt_training import (  # noqa: E402
    T1ceWtPreprocessConfig,
    T1ceWtTrainingDataset,
)
from scripts import train_sammed3d_brats as generic  # noqa: E402
from scripts.train_sammed3d_t1ce_wt import (  # noqa: E402
    GENERIC_TRAINER_SHA256,
    cumulative_interaction_forward,
)


PREREG_PATH = Path(
    "workspace/BraTS/training_protocols/"
    "fv_last4_partial_unfreeze_disclr_prereg_20260724.json"
)
PREREG_SHA256 = "b55e763c4ab422a8b38d49df7fe6dde9e95af8d5e154d485eaef5fdcc97a5f91"
TRAIN_MANIFEST = Path(
    "data/datasets/brats2024_sammed3d_t1ce_wt_inner90_10_seed20260723_v1/"
    "train_t1ce_wt.jsonl"
)
VAL_MANIFEST = Path(
    "data/datasets/brats2024_sammed3d_t1ce_wt_inner90_10_seed20260723_v1/"
    "validation_t1ce_wt.jsonl"
)
BASE_CHECKPOINT = SAMMED3D_CHECKPOINT
FORMAL_OUTPUT = Path(
    "workspace/BraTS/sammed3d_t1ce_wt/"
    "fv_partial_last4_disclr10_train972_val108_e50_s20260725"
)
SMOKE_OUTPUT = Path(
    "workspace/BraTS/sammed3d_t1ce_wt/"
    "smoke_fv_partial_last4_disclr10_2train_2val_e1_s20260725"
)
TRAIN_SEED = 20260725
VAL_SEED = 20260723
MAX_EPOCHS = 50
MIN_EPOCHS = 20
PATIENCE = 20
TRAIN_ROWS = 972
TRAIN_SUBJECTS = 444
VAL_ROWS = 108
VAL_SUBJECTS = 50


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def _nested(record: dict[str, Any], *keys: str) -> Any:
    current: Any = record
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise RuntimeError(f"preregistration is missing {'.'.join(keys)}")
        current = current[key]
    return current


def _require_prereg_value(
    prereg: dict[str, Any],
    keys: tuple[str, ...],
    expected: Any,
) -> None:
    actual = _nested(prereg, *keys)
    if actual != expected:
        raise RuntimeError(
            f"preregistration {'.'.join(keys)}={actual!r}, expected {expected!r}"
        )


def load_and_validate_prereg() -> tuple[dict[str, Any], Path]:
    path = _resolve(PREREG_PATH)
    actual_sha = sha256_file(path)
    if actual_sha != PREREG_SHA256:
        raise RuntimeError(
            f"machine preregistration SHA mismatch: {actual_sha}"
        )
    prereg = json.loads(path.read_text(encoding="utf-8"))
    fixed = {
        ("schema",): "aflow.sammed3d.t1ce_wt_partial_unfreeze_prereg",
        ("schema_version",): 1,
        ("status",): "development_only_train_authorized",
        ("authorization", "development_training"): True,
        ("authorization", "test_access"): False,
        ("optimization", "batch_size"): 1,
        ("optimization", "accumulation_steps"): 2,
        ("optimization", "amp_dtype"): "bf16",
        ("optimization", "num_workers"): 2,
        ("optimization", "train_seed"): TRAIN_SEED,
        ("optimization", "rng_reseed_after_model_setup"): True,
        ("optimization", "validation_seed"): VAL_SEED,
        ("optimization", "early_stopping", "max_epochs"): MAX_EPOCHS,
        ("optimization", "early_stopping", "minimum_complete_epochs"): MIN_EPOCHS,
        ("optimization", "early_stopping", "patience_complete_epochs"): PATIENCE,
        ("optimization", "early_stopping", "min_delta"): 0.0,
        ("optimization", "early_stopping", "tie_rule"): "retain earliest",
        ("preprocessing", "geometry", "name"): "whole_resize",
        ("preprocessing", "augmentation", "name"): "affine_elastic_noise",
        ("preprocessing", "augmentation", "validation_augmentation"): False,
        ("interaction", "clicks", "num_clicks"): 3,
        ("initialization", "checkpoint_sha256"):
            "899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c",
    }
    for keys, expected in fixed.items():
        _require_prereg_value(prereg, keys, expected)

    parent = _resolve(Path(prereg["parent_md_path"]))
    if sha256_file(parent) != prereg["parent_md_sha256"]:
        raise RuntimeError("parent preregistration MD SHA mismatch")
    for split, expected_path, rows, subjects in (
        ("train", TRAIN_MANIFEST, TRAIN_ROWS, TRAIN_SUBJECTS),
        ("validation", VAL_MANIFEST, VAL_ROWS, VAL_SUBJECTS),
    ):
        _require_prereg_value(
            prereg,
            ("data", split, "manifest"),
            str(expected_path),
        )
        _require_prereg_value(prereg, ("data", split, "rows"), rows)
        _require_prereg_value(prereg, ("data", split, "subjects"), subjects)
        manifest = _resolve(expected_path)
        if sha256_file(manifest) != prereg["data"][split]["manifest_sha256"]:
            raise RuntimeError(f"{split} manifest SHA mismatch")
    return prereg, path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preregistered T1ce/WT last-four partial-unfreeze trainer"
    )
    parser.add_argument("--mode", choices=["formal", "smoke"], required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def resolve_output(args: argparse.Namespace, prereg: dict[str, Any]) -> Path:
    if args.mode == "formal":
        expected = _resolve(Path(prereg["output"]["run_path"]))
        output = _resolve(args.output_dir) if args.output_dir else expected
        if output != expected:
            raise RuntimeError("formal output path must exactly match preregistration")
        if args.device != "cuda":
            raise RuntimeError("formal arm requires CUDA BF16 execution")
    else:
        output = _resolve(args.output_dir or SMOKE_OUTPUT)
        if "smoke_fv_partial_last4" not in output.name:
            raise RuntimeError("smoke output basename must identify the smoke arm")
        if output == _resolve(FORMAL_OUTPUT):
            raise RuntimeError("smoke may not use the formal output path")
    if output.exists():
        raise FileExistsError(f"refusing non-empty or pre-existing output: {output}")
    return output


def read_and_validate_manifests() -> tuple[list[dict], list[dict]]:
    train_rows = generic.read_jsonl(_resolve(TRAIN_MANIFEST))
    val_rows = generic.read_jsonl(_resolve(VAL_MANIFEST))
    for rows, split, expected_rows, expected_subjects in (
        (train_rows, "train", TRAIN_ROWS, TRAIN_SUBJECTS),
        (val_rows, "validation", VAL_ROWS, VAL_SUBJECTS),
    ):
        if len(rows) != expected_rows:
            raise RuntimeError(f"{split} row count mismatch")
        cases = [str(row.get("case_id")) for row in rows]
        subjects = {str(row.get("subject_id")) for row in rows}
        if len(set(cases)) != expected_rows or len(subjects) != expected_subjects:
            raise RuntimeError(f"{split} case/subject count mismatch")
        for row in rows:
            if (
                row.get("split") != split
                or row.get("target") != "WT"
                or row.get("modality") != "t1ce"
                or row.get("foreground_labels") != [1, 2, 3]
                or row.get("excluded_labels") != [4]
            ):
                raise RuntimeError(f"{split} manifest protocol row mismatch")
    if {row["case_id"] for row in train_rows} & {
        row["case_id"] for row in val_rows
    }:
        raise RuntimeError("train/validation case overlap")
    if {row["subject_id"] for row in train_rows} & {
        row["subject_id"] for row in val_rows
    }:
        raise RuntimeError("train/validation subject overlap")
    return train_rows, val_rows


def ensure_finite_gradients(selection: partial_unfreeze.ParameterSelection) -> None:
    for group_name, parameters in (
        ("encoder_last4", selection.encoder_parameters),
        ("prompt_and_mask_heads", selection.head_parameters),
    ):
        gradients = [
            parameter.grad for parameter in parameters
            if parameter.grad is not None
        ]
        # Some prompt embeddings and the decoder IoU head are legitimately
        # unused by one click/loss path. Each optimizer group must still have
        # a finite gradient signal before every update.
        if not gradients:
            raise RuntimeError(f"{group_name} has no gradients")
        if any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise RuntimeError(f"{group_name} has non-finite gradients")


def run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    selection: partial_unfreeze.ParameterSelection,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    amp_dtype: torch.dtype | None,
    accumulation_steps: int,
) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    dice_total = 0.0
    samples = 0
    start = time.perf_counter()
    iterator = tqdm(loader, desc="train" if training else "val", leave=False)
    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for step, batch in enumerate(iterator):
            image = batch["image"].to(device, non_blocking=True).float()
            target = batch["label"].to(device, non_blocking=True).float()
            autocast = (
                torch.amp.autocast("cuda", dtype=amp_dtype)
                if amp_dtype is not None and device.type == "cuda"
                else nullcontext()
            )
            with autocast:
                logits, loss = cumulative_interaction_forward(
                    model=model,
                    image=image,
                    gt=target,
                    loss_fn=loss_fn,
                    num_clicks=3,
                    click_method="random",
                    train_image_encoder=True,
                )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite loss")
            if training:
                scaled_loss = loss / accumulation_steps
                if scaler is None:
                    scaled_loss.backward()
                else:
                    scaler.scale(scaled_loss).backward()
                should_step = (
                    (step + 1) % accumulation_steps == 0
                    or (step + 1) == len(loader)
                )
                if should_step:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    ensure_finite_gradients(selection)
                    if scaler is None:
                        optimizer.step()
                    else:
                        scaler.step(optimizer)
                        scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            scores = generic.dice_scores_from_logits(logits.detach(), target)
            if any(score is None or not math.isfinite(score) for score in scores):
                raise RuntimeError("non-finite or undefined validation/training Dice")
            batch_size = int(target.shape[0])
            loss_total += float(loss.detach().cpu()) * batch_size
            dice_total += sum(float(score) for score in scores if score is not None)
            samples += batch_size
            iterator.set_postfix(
                loss=f"{loss_total / samples:.4f}",
                dice=f"{dice_total / samples:.4f}",
            )
    elapsed = time.perf_counter() - start
    result: dict[str, float | int] = {
        "loss": loss_total / samples,
        "dice": dice_total / samples,
        "samples": samples,
        "elapsed_sec": elapsed,
        "cases_per_sec": samples / elapsed,
    }
    if not all(math.isfinite(float(result[key])) for key in ("loss", "dice")):
        raise RuntimeError("non-finite epoch metric")
    return result


def checkpoint_metadata(
    *,
    kind: str,
    epoch: int,
    metric_value: float,
    selected: bool,
    selection: partial_unfreeze.ParameterSelection,
    prereg_path: Path,
    mode: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "metric": "val_dice",
        "metric_value": metric_value,
        "metric_epoch": epoch,
        "is_selected_best": selected,
        "mode": mode,
        "scientific_result_eligible": mode == "formal",
        "partial_unfreeze_policy": selection.record,
        "preregistration": {
            "path": str(prereg_path),
            "sha256": PREREG_SHA256,
        },
    }


def strict_reload_checkpoint(
    path: Path,
    expected_selection: partial_unfreeze.ParameterSelection,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("checkpoint payload must be a mapping")
    partial_unfreeze.validate_checkpoint_payload(
        payload,
        expected_selection.record,
    )
    expected_adapter = peft.make_adapter_config(
        target_token=False,
        lora_rank=0,
        model_type=engine.MODEL_TYPE,
    )
    if peft.normalize_adapter_config(payload["adapter_config"]) != expected_adapter:
        raise RuntimeError("checkpoint adapter_config must remain disabled")
    reloaded, adapter_config = peft.load_model_with_adapters(
        path,
        lambda model_type: engine.sam_model_registry3D[model_type](
            checkpoint=None
        ),
        device=torch.device("cpu"),
        requested_config=expected_adapter,
        default_model_type=engine.MODEL_TYPE,
    )
    if adapter_config != expected_adapter:
        raise RuntimeError("strict reload adapter configuration mismatch")
    reloaded_selection = partial_unfreeze.configure_vit_b_ori_last_four(reloaded)
    if reloaded_selection.record != expected_selection.record:
        raise RuntimeError("strict reload partial-unfreeze policy mismatch")
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "epoch": int(payload["epoch"]),
        "strict_state_load": True,
        "partial_unfreeze_policy_exact": True,
        "optimizer_groups_exact": True,
    }


def write_json(path: Path, record: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "train_dice",
    "train_cases_per_sec",
    "val_loss",
    "val_dice",
    "val_cases_per_sec",
    "strict_new_high",
    "stale_complete_epochs",
    "cuda_peak_memory_allocated",
    "cuda_peak_memory_reserved",
)


def append_history_csv(path: Path, row: dict[str, Any]) -> None:
    if set(row) != set(HISTORY_FIELDS):
        raise RuntimeError("history row fields differ from the fixed CSV schema")
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def reseed_after_model_setup() -> None:
    """Match the anchored arms by resetting RNG after model/optimizer setup."""
    generic.seed_everything(TRAIN_SEED)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    prereg, prereg_path = load_and_validate_prereg()
    output = resolve_output(args, prereg)
    train_rows, val_rows = read_and_validate_manifests()
    if args.mode == "smoke":
        train_rows = train_rows[:2]
        val_rows = val_rows[:2]
        epochs = 1
        num_workers = 0
    else:
        epochs = MAX_EPOCHS
        num_workers = 2

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA device does not support BF16")
    generic.seed_everything(TRAIN_SEED)

    base_checkpoint = _resolve(BASE_CHECKPOINT)
    if sha256_file(base_checkpoint) != prereg["initialization"]["checkpoint_sha256"]:
        raise RuntimeError("untouched base checkpoint SHA mismatch")
    adapter_request = peft.make_adapter_config(
        target_token=False,
        lora_rank=0,
        model_type=engine.MODEL_TYPE,
    )
    model, adapter_config = peft.load_model_with_adapters(
        base_checkpoint,
        lambda model_type: engine.sam_model_registry3D[model_type](
            checkpoint=None
        ),
        device=device,
        requested_config=adapter_request,
        default_model_type=engine.MODEL_TYPE,
    )
    if peft.adapters_enabled(adapter_config):
        raise RuntimeError("partial-unfreeze arm requires all adapters disabled")
    selection = partial_unfreeze.configure_vit_b_ori_last_four(model)
    if selection.record != prereg["optimization"]["partial_unfreeze_policy"]:
        raise RuntimeError("runtime partial-unfreeze policy differs from preregistration")
    optimizer = partial_unfreeze.build_adamw(selection)
    train_image_encoder = any(
        parameter.requires_grad for parameter in model.image_encoder.parameters()
    )
    if not train_image_encoder:
        raise RuntimeError("partial-unfreeze arm requires train_image_encoder=True")

    amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    scaler = generic.make_grad_scaler(device, amp_dtype)
    loss_fn = generic.DiceCELoss(
        sigmoid=True,
        squared_pred=True,
        reduction="mean",
    )

    # Model construction/loading consumes torch RNG. Reset here so shuffling,
    # stochastic augmentation, and clicks start from the matched-arm state.
    reseed_after_model_setup()
    config = T1ceWtPreprocessConfig(
        spatial_preprocessing="whole_resize",
        train_augmentation="affine_elastic_noise",
    )
    train_generator = generic.make_seeded_generator(TRAIN_SEED + 1)
    train_loader = DataLoader(
        T1ceWtTrainingDataset(train_rows, config=config),
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=train_generator,
    )
    val_loader = DataLoader(
        T1ceWtTrainingDataset(val_rows, config=config),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    runner_path = Path(__file__).resolve()
    generic_path = Path(generic.__file__).resolve()
    if sha256_file(generic_path) != GENERIC_TRAINER_SHA256:
        raise RuntimeError("anchored generic trainer SHA mismatch")
    run_config = {
        "schema": "aflow.sammed3d.t1ce_wt_partial_unfreeze_training",
        "schema_version": 1,
        "mode": args.mode,
        "scientific_result_eligible": args.mode == "formal",
        "resume_supported": False,
        "test_access": False,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "max_epochs": epochs,
        "minimum_epochs": MIN_EPOCHS if args.mode == "formal" else None,
        "patience": PATIENCE if args.mode == "formal" else None,
        "selection_metric": "val_dice",
        "tie_rule": "strict new high; retain earliest",
        "train_image_encoder": train_image_encoder,
        "partial_unfreeze_policy": selection.record,
        "adapter_config": adapter_config,
        "preprocessing": config.to_protocol_dict(),
        "interaction": prereg["interaction"],
        "train_seed": TRAIN_SEED,
        "validation_seed": VAL_SEED,
        "batch_size": 1,
        "accumulation_steps": 2,
        "num_workers": num_workers,
        "amp_dtype": "bf16" if amp_dtype is not None else "none",
        "device": str(device),
        "preregistration": {
            "path": str(prereg_path),
            "sha256": PREREG_SHA256,
            "parent_md_path": prereg["parent_md_path"],
            "parent_md_sha256": prereg["parent_md_sha256"],
        },
        "source": {
            "base_checkpoint": str(base_checkpoint),
            "base_checkpoint_sha256": sha256_file(base_checkpoint),
            "train_manifest": str(_resolve(TRAIN_MANIFEST)),
            "train_manifest_sha256": sha256_file(_resolve(TRAIN_MANIFEST)),
            "validation_manifest": str(_resolve(VAL_MANIFEST)),
            "validation_manifest_sha256": sha256_file(_resolve(VAL_MANIFEST)),
            "runner": str(runner_path),
            "runner_sha256": sha256_file(runner_path),
            "policy_helper": str(Path(partial_unfreeze.__file__).resolve()),
            "policy_helper_sha256": sha256_file(
                Path(partial_unfreeze.__file__).resolve()
            ),
            "generic_trainer": str(generic_path),
            "generic_trainer_sha256": sha256_file(generic_path),
        },
    }
    if args.preflight_only:
        print(json.dumps({
            "preflight_passed": True,
            "mode": args.mode,
            "output_would_be": str(output),
            "output_created": False,
            "preregistration_sha256": PREREG_SHA256,
            "partial_unfreeze_policy": selection.record,
            "train_image_encoder": train_image_encoder,
            "test_access": False,
        }, indent=2, sort_keys=True))
        return

    # All manifest, checkpoint, model, policy, optimizer, and source-hash
    # checks above must pass before the uniquely named output is created.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    write_json(output / "train_config.json", run_config)
    if args.mode == "smoke":
        write_json(
            output / "SMOKE_ONLY.json",
            {
                "scientific_result_eligible": False,
                "resume_allowed": False,
                "purpose": prereg["smoke"]["purpose"],
            },
        )

    history: list[dict[str, Any]] = []
    best = {
        "val_dice": {"value": -1.0, "epoch": -1},
        "selection": {
            "metric": "val_dice",
            "checkpoint": "sam_model_val_dice_best.pth",
            "value": -1.0,
            "epoch": -1,
        },
    }
    stale_epochs = 0
    stop_reason = "max_epochs"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch_index in range(epochs):
        epoch = epoch_index + 1
        partial_unfreeze.validate_optimizer(optimizer, selection.record)
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            selection=selection,
            optimizer=optimizer,
            scaler=scaler,
            amp_dtype=amp_dtype,
            accumulation_steps=2,
        )
        with generic.fixed_rng_seed(VAL_SEED, device):
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                selection=selection,
                optimizer=None,
                scaler=None,
                amp_dtype=amp_dtype,
                accumulation_steps=1,
            )
        val_dice = float(val_metrics["dice"])
        improved = val_dice > float(best["val_dice"]["value"])
        if improved:
            best["val_dice"] = {"value": val_dice, "epoch": epoch}
            best["selection"].update(value=val_dice, epoch=epoch)
            stale_epochs = 0
        else:
            stale_epochs += 1
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_cases_per_sec": train_metrics["cases_per_sec"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_dice,
            "val_cases_per_sec": val_metrics["cases_per_sec"],
            "strict_new_high": improved,
            "stale_complete_epochs": stale_epochs,
            "cuda_peak_memory_allocated": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
            "cuda_peak_memory_reserved": (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda"
                else None
            ),
        }
        history.append(row)
        append_history_csv(output / "history.csv", row)
        print(json.dumps(row, sort_keys=True), flush=True)

        common_save_args = (
            model,
            optimizer,
            scaler,
            epoch_index,
            args,
            history,
            best,
            adapter_config,
        )
        generic.save_checkpoint(
            output / "sam_model_latest.pth",
            *common_save_args,
            checkpoint_metadata=checkpoint_metadata(
                kind="latest",
                epoch=epoch,
                metric_value=val_dice,
                selected=improved,
                selection=selection,
                prereg_path=prereg_path,
                mode=args.mode,
            ),
        )
        if improved:
            selected_path = output / "sam_model_val_dice_best.pth"
            generic.save_checkpoint(
                selected_path,
                *common_save_args,
                checkpoint_metadata=checkpoint_metadata(
                    kind="selected_best",
                    epoch=epoch,
                    metric_value=val_dice,
                    selected=True,
                    selection=selection,
                    prereg_path=prereg_path,
                    mode=args.mode,
                ),
            )
            reload_record = strict_reload_checkpoint(selected_path, selection)
            write_json(output / "selected_checkpoint_reload.json", reload_record)
        if args.mode == "formal" and epoch >= MIN_EPOCHS and stale_epochs >= PATIENCE:
            stop_reason = "performance_early_stop"
            break

    selected_path = output / "sam_model_val_dice_best.pth"
    final_reload = strict_reload_checkpoint(selected_path, selection)
    selected_epoch = int(best["selection"]["epoch"])
    selected_value = float(best["selection"]["value"])
    final_row = history[-1]
    done = {
        "schema": "aflow.sammed3d.t1ce_wt_partial_unfreeze_train_done",
        "schema_version": 1,
        "mode": args.mode,
        "scientific_result_eligible": args.mode == "formal",
        "completed_epochs": len(history),
        "stop_reason": stop_reason,
        "selected_epoch": selected_epoch,
        "selected_val_dice": selected_value,
        "selected_to_final_val_dice_drop": selected_value
        - float(final_row["val_dice"]),
        "final_train_minus_validation_dice": float(final_row["train_dice"])
        - float(final_row["val_dice"]),
        "strict_reload": final_reload,
        "test_access": False,
    }
    write_json(output / "best.json", best)
    write_json(output / "TRAIN_DONE.json", done)
    print(f"TRAIN_DONE output={output} selected_epoch={selected_epoch}", flush=True)


if __name__ == "__main__":
    main()
