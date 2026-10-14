#!/usr/bin/env python
"""BraTS T1ce/WT training with the literal upstream SAM-Med3D interaction recipe.

Scientific scope
----------------
The deterministic input tensors come from the verified official15 VA cache:
1.5-mm paired resampling, ToCanonical, GT-centred CropOrPad(128^3), positive
voxel sample z-score, no N4, and train-only RandomFlip.

The training forward mirrors the behavior actually executed by upstream
SAM-Med3D ``train.py`` (SHA pinned below), including its quirks:

* 11 interactions are hard-coded (the random 1..20 value is sampled but unused);
* only the current click is passed (``multi_click`` defaults false);
* one random iteration in [2, 8] and the last iteration pass no sparse point;
* the recurrent low-resolution mask is passed at every iteration;
* error clicks and the train Dice threshold raw logits at > 0.5;
* all 11 DiceCE losses are summed, not averaged;
* AdamW uses encoder/head LR 8e-4/8e-5 and MultiStepLR(120,180; gamma=.1);
* source accumulation steps on indices 20,40,60,80, so the first update contains
  21 microbatches divided by 20.  With 2 GPUs x batch 6 this preserves the
  source single-GPU batch-12 loader's 81 steps and nominal batch 240.

Validation is an isolated, deterministic official-style diagnostic: five
current-point/recurrent-mask interactions, RITM for clicks 1-2 and uniform XOR
thereafter, reporting both final-click and upstream per-case best-of-five Dice.
It does not perturb the training RNG.  Upstream train.py has no validation loop
and selects checkpoints by train loss / train Dice; those checkpoints are saved
in addition to the development-only val-best checkpoint.

The sealed test split is not accepted or read.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from contextlib import nullcontext
from typing import Any

import edt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d import engine, peft  # noqa: E402
from sammed3d.paths import SAMMED3D_ROOT  # noqa: E402
from sammed3d.t1ce_wt_cached import CachedT1ceWtDataset  # noqa: E402
from sammed3d.t1ce_wt_training import T1ceWtPreprocessConfig  # noqa: E402
from scripts import train_sammed3d_brats as generic  # noqa: E402
from scripts import train_sammed3d_t1ce_wt_dpdsam_aligned as base  # noqa: E402
from scripts.train_sammed3d_t1ce_wt_official15_va import (  # noqa: E402
    TRAIN_COUNT,
    TRAIN_MANIFEST_SHA256,
    VAL_COUNT,
    VAL_MANIFEST_SHA256,
    official_protocol_dict,
    validate_cache_contract,
)

SCHEMA = "aflow.sammed3d.t1ce_wt_upstream_literal_va_training"
SCHEMA_VERSION = 1
RUN_IDENTITY = (
    "sammed3d_upstreamliteral11_officialval5_va_a100x2_"
    "train972_val108_e200_s2023"
)
UPSTREAM_TRAIN = SAMMED3D_ROOT / "train.py"
UPSTREAM_INFERENCE = SAMMED3D_ROOT / "inference.py"
UPSTREAM_CLICK = SAMMED3D_ROOT / "utils" / "click_method.py"
UPSTREAM_TRAIN_SHA256 = (
    "474d9765e29c0196e9cf1da2242869c2cbd6970632d9f113357d38a8094e1e3f"
)
UPSTREAM_INFERENCE_SHA256 = (
    "5a31aff32805f297eb5632b4362d1aa60bde4491d168868bc750a43e332c3960"
)
UPSTREAM_CLICK_SHA256 = (
    "0838972fb2f33de579a2652d3225f862d84dc6df113b152ea6018730c9b792d7"
)
TRAIN_SEED = 2023
VALIDATION_SEED = 2023
TRAIN_INTERACTIONS = 11
VALIDATION_CLICKS = 5
SOURCE_FOREGROUND_THRESHOLD = 1000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_upstream_sources() -> None:
    expected = {
        UPSTREAM_TRAIN: UPSTREAM_TRAIN_SHA256,
        UPSTREAM_INFERENCE: UPSTREAM_INFERENCE_SHA256,
        UPSTREAM_CLICK: UPSTREAM_CLICK_SHA256,
    }
    for path, digest in expected.items():
        actual = sha256_file(path)
        if actual != digest:
            raise RuntimeError(f"upstream source SHA mismatch: {path} {actual} != {digest}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--cache-root", default="/tmp/aflow_t1ce_wt_official15_va_128")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-identity", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=12, help="per GPU")
    parser.add_argument("--accumulation-steps", type=int, default=20)
    parser.add_argument("--encoder-lr", type=float, default=8e-4)
    parser.add_argument("--head-lr", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=12, help="per rank")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace, world: int) -> None:
    expected = {
        "epochs": 200,
        "encoder_lr": 8e-4,
        "head_lr": 8e-5,
        "weight_decay": 0.1,
    }
    failed = [name for name, value in expected.items() if getattr(args, name) != value]
    if world != 2:
        failed.append("world_size=2")
    if args.mode == "formal":
        if args.batch_size != 12:
            failed.append("per_gpu_batch_size=12")
        if args.accumulation_steps != 20:
            failed.append("accumulation_steps=20")
        if args.num_workers != 12:
            failed.append("per_rank_num_workers=12")
    else:
        if args.batch_size != 12:
            failed.append("smoke_per_gpu_batch_size=12")
        if args.accumulation_steps != 20:
            failed.append("smoke_accumulation_steps=20")
    if failed:
        raise ValueError(f"arguments violate upstream-literal protocol: {failed}")


class ForegroundThresholdReplacement(Dataset):
    """Mirror Dataset_Union_ALL's recursive replacement of small labels."""

    def __init__(self, dataset: Dataset, positive_voxels: list[int], threshold: int):
        self.dataset = dataset
        self.positive_voxels = [int(value) for value in positive_voxels]
        self.threshold = int(threshold)
        if len(dataset) != len(self.positive_voxels):
            raise ValueError("positive voxel list length mismatch")
        self.eligible = np.flatnonzero(np.asarray(self.positive_voxels) > threshold)
        if not len(self.eligible):
            raise ValueError("foreground threshold excludes the complete dataset")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        selected = int(index)
        rejected = 0
        while True:
            # Upstream augments before checking the threshold. This consumes
            # RandomFlip RNG even when the case is rejected recursively.
            item = dict(self.dataset[selected])
            if int(item["label"].sum()) > self.threshold:
                break
            rejected += 1
            selected = int(np.random.randint(len(self.dataset)))
        item["requested_index"] = int(index)
        item["selected_index"] = selected
        item["threshold_replaced"] = selected != int(index)
        item["threshold_rejections"] = rejected
        return item


def _sample_uniform_xor(
    previous_logits: torch.Tensor, target: torch.Tensor, *, threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact get_next_click3D_torch_2 behavior, generalized safely to batch."""
    prediction = previous_logits > threshold
    truth = target > 0
    false_negative = torch.logical_and(truth, ~prediction)
    false_positive = torch.logical_and(~truth, prediction)
    points: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for index in range(target.shape[0]):
        candidates = torch.argwhere(
            torch.logical_or(false_negative[index], false_positive[index])
        )
        if len(candidates) == 0:
            # Upstream would crash in this unlikely state. Fail closed rather
            # than silently inventing a different prompt.
            raise RuntimeError("upstream uniform XOR sampler received an empty error mask")
        point = candidates[np.random.randint(len(candidates))]
        positive = bool(false_negative[index, 0, point[1], point[2], point[3]])
        points.append(point[1:].reshape(1, 1, 3))
        labels.append(torch.tensor([[int(positive)]], device=target.device))
    return torch.cat(points).to(target.device), torch.cat(labels).long()


def _sample_ritm_single(
    previous_logits: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact upstream RITM prompt for its inference batch size of one."""
    if target.shape[0] != 1:
        raise ValueError("upstream RITM implementation requires batch size one")
    prediction = previous_logits > 0.5
    truth = target > 0
    false_negative = torch.logical_and(truth, ~prediction)
    false_positive = torch.logical_and(~truth, prediction)
    fn = F.pad(false_negative, (1, 1, 1, 1, 1, 1), value=0)[0, 0].byte()
    fp = F.pad(false_positive, (1, 1, 1, 1, 1, 1), value=0)[0, 0].byte()
    fn_dt = torch.from_numpy(edt.edt(fn.cpu().numpy(), black_border=True, parallel=4))[
        1:-1, 1:-1, 1:-1
    ]
    fp_dt = torch.from_numpy(edt.edt(fp.cpu().numpy(), black_border=True, parallel=4))[
        1:-1, 1:-1, 1:-1
    ]
    fn_max, fp_max = fn_dt.max(), fp_dt.max()
    distance = fn_dt if bool(fn_max > fp_max) else fp_dt
    candidates = torch.argwhere(distance > max(fn_max, fp_max) / 2.0)
    if len(candidates) == 0:
        raise RuntimeError("upstream RITM sampler received an empty error mask")
    point = candidates[np.random.randint(len(candidates))]
    positive = bool(false_negative[0, 0, point[0], point[1], point[2]])
    return (
        point.reshape(1, 1, 3).to(target.device),
        torch.tensor([[int(positive)]], device=target.device, dtype=torch.long),
    )


def upstream_train_interaction(
    *,
    model: torch.nn.Module,
    image: torch.Tensor,
    target: torch.Tensor,
    loss_fn: torch.nn.Module,
    random_insert: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Execute upstream train.py's hard-coded 11-interaction forward."""
    image_embedding = model.image_encoder(image)
    previous = torch.zeros_like(target, dtype=torch.float32)
    low_res = F.interpolate(previous, size=(engine.CROP_SIZE // 4,) * 3)
    insert = int(np.random.randint(2, 9)) if random_insert is None else int(random_insert)
    if insert < 2 or insert > 8:
        raise ValueError("random_insert must be in [2,8]")
    total_loss = image.sum() * 0.0
    no_point_iterations: list[int] = []
    for interaction in range(TRAIN_INTERACTIONS):
        point, label = _sample_uniform_xor(previous.detach(), target.detach(), threshold=0.5)
        no_point = interaction == insert or interaction == TRAIN_INTERACTIONS - 1
        points = None if no_point else (point, label)
        if no_point:
            no_point_iterations.append(interaction)
        sparse, dense = model.prompt_encoder(points=points, boxes=None, masks=low_res)
        low_res, _ = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        previous = F.interpolate(
            low_res, size=target.shape[-3:], mode="trilinear", align_corners=False
        )
        total_loss = total_loss + loss_fn(previous, target.float())
    return previous, total_loss, {
        "interactions": TRAIN_INTERACTIONS,
        "random_insert": insert,
        "no_point_iterations": no_point_iterations,
        "loss_reduction_across_interactions": "sum",
    }


def _dice_per_sample(
    logits: torch.Tensor, target: torch.Tensor, *, raw_threshold: float
) -> list[float]:
    prediction = logits > raw_threshold
    truth = target > 0
    scores: list[float] = []
    for pred, gt in zip(prediction, truth):
        denominator = int(pred.sum()) + int(gt.sum())
        if denominator == 0:
            raise RuntimeError("undefined Dice")
        intersection = int(torch.logical_and(pred, gt).sum())
        scores.append(2.0 * intersection / denominator)
    return scores


@torch.no_grad()
def upstream_validation_curve(
    *, model: torch.nn.Module, image: torch.Tensor, target: torch.Tensor
) -> list[float]:
    """Mirror inference.py: current point, recurrent mask, best of five."""
    if image.shape[0] != 1:
        raise ValueError("official-style validation uses batch size one")
    image_embedding = model.image_encoder(image)
    previous = torch.zeros_like(target, dtype=torch.float32)
    low_res = F.interpolate(previous, size=(engine.CROP_SIZE // 4,) * 3)
    curve: list[float] = []
    for click in range(VALIDATION_CLICKS):
        if click <= 1:
            point, label = _sample_ritm_single(previous, target)
        else:
            point, label = _sample_uniform_xor(previous, target, threshold=0.5)
        sparse, dense = model.prompt_encoder(
            points=(point, label), boxes=None, masks=low_res
        )
        low_res, _ = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        previous = F.interpolate(
            low_res, size=target.shape[-3:], mode="trilinear", align_corners=False
        )
        # inference.py applies sigmoid > .5, exactly raw logit > 0.
        curve.append(_dice_per_sample(previous, target, raw_threshold=0.0)[0])
    return curve


def stable_case_seed(case_id: str) -> int:
    payload = f"{VALIDATION_SEED}:{case_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _autocast(device: torch.device):
    return (
        torch.amp.autocast("cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )


def train_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    sampler: DistributedSampler | None,
    epoch: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    selection: Any,
    loss_fn: torch.nn.Module,
    accumulation_steps: int,
) -> dict[str, float]:
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)
    # train.py samples this and passes it to train_epoch, which ignores it.
    _unused_num_clicks = int(np.random.randint(1, 21))
    optimizer.zero_grad(set_to_none=True)
    batch_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    batch_dice_sum = torch.zeros((), device=device, dtype=torch.float64)
    batch_count = torch.zeros((), device=device, dtype=torch.float64)
    sample_count = torch.zeros((), device=device, dtype=torch.float64)
    replacements = torch.zeros((), device=device, dtype=torch.float64)
    optimizer_steps = 0
    optimizer_skipped_steps = 0
    started = time.perf_counter()
    for step, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True).float()
        target = batch["label"].to(device, non_blocking=True).long()
        with _autocast(device):
            logits, loss, _ = upstream_train_interaction(
                model=model, image=image, target=target, loss_fn=loss_fn
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite upstream summed interaction loss")
        scaler.scale(loss / accumulation_steps).backward()
        # Literal source boundary: step 0 does not update; 20,40,... do.
        if step % accumulation_steps == 0 and step != 0:
            scale_before = scaler.get_scale()
            base.average_gradients(selection)
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                optimizer_skipped_steps += 1
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        scores = _dice_per_sample(logits.detach(), target, raw_threshold=0.5)
        bs = int(target.shape[0])
        # Upstream averages batch-reduced metrics over loader steps.
        batch_loss_sum += float(loss.detach())
        batch_dice_sum += sum(scores) / bs
        batch_count += 1
        sample_count += bs
        replacements += batch["threshold_replaced"].to(device).sum()
    if base.is_dist():
        packed = torch.stack(
            [batch_loss_sum, batch_dice_sum, batch_count, sample_count, replacements]
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        batch_loss_sum, batch_dice_sum, batch_count, sample_count, replacements = packed
        steps_tensor = torch.tensor(
            [optimizer_steps, optimizer_skipped_steps], device=device
        )
        dist.all_reduce(steps_tensor, op=dist.ReduceOp.MAX)
        optimizer_steps = int(steps_tensor[0].item())
        optimizer_skipped_steps = int(steps_tensor[1].item())
    elapsed = time.perf_counter() - started
    batches = float(batch_count)
    samples = float(sample_count)
    return {
        "loss_sum11": float(batch_loss_sum) / batches,
        "dice_raw05_final11": float(batch_dice_sum) / batches,
        "samples": samples,
        "threshold_replacements": float(replacements),
        "optimizer_steps": float(optimizer_steps),
        "optimizer_skipped_steps": float(optimizer_skipped_steps),
        "optimizer_applied_steps": float(optimizer_steps - optimizer_skipped_steps),
        "elapsed_sec": elapsed,
        "cases_per_sec": samples / elapsed,
        "unused_epoch_num_clicks_draw": float(_unused_num_clicks),
    }


@torch.no_grad()
def validate_official_style(
    *, model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    best_sum = 0.0
    final_sum = 0.0
    click_sums = np.zeros(VALIDATION_CLICKS, dtype=np.float64)
    count = 0
    started = time.perf_counter()
    saved_numpy_state = np.random.get_state()
    try:
        for batch in loader:
            case_id = str(batch["case_id"][0])
            np.random.seed(stable_case_seed(case_id))
            image = batch["image"].to(device, non_blocking=True).float()
            target = batch["label"].to(device, non_blocking=True).float()
            with _autocast(device):
                curve = upstream_validation_curve(model=model, image=image, target=target)
            click_sums += np.asarray(curve)
            best_sum += max(curve)
            final_sum += curve[-1]
            count += 1
    finally:
        np.random.set_state(saved_numpy_state)
    if count <= 0:
        raise RuntimeError("validation processed zero cases")
    elapsed = time.perf_counter() - started
    result = {
        "best_of_5_dice": best_sum / count,
        "final_click_5_dice": final_sum / count,
        "samples": float(count),
        "elapsed_sec": elapsed,
        "cases_per_sec": count / elapsed,
    }
    for click, value in enumerate(click_sums / count, start=1):
        result[f"click_{click}_dice"] = float(value)
    return result


def save_state(
    path: Path,
    *,
    model: torch.nn.Module,
    epoch: int,
    adapter_config: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "adapter_config": adapter_config,
            "checkpoint_metadata": metadata,
        },
        tmp,
    )
    tmp.replace(path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=dt.timedelta(minutes=30))
    device = torch.device(f"cuda:{local_rank}")
    validate_args(args, base.world_size())
    validate_upstream_sources()
    if sha256_file(base.BASE_CHECKPOINT) != base.BASE_CHECKPOINT_SHA256:
        raise RuntimeError("base turbo checkpoint SHA mismatch")

    cache_root = Path(args.cache_root)
    if not cache_root.is_absolute():
        cache_root = ROOT / cache_root
    cache_indexes = validate_cache_contract(cache_root)
    config = T1ceWtPreprocessConfig(
        spatial_preprocessing="gt_crop", train_augmentation="random_flip"
    )
    train_cached = CachedT1ceWtDataset(
        cache_root / "train",
        split="train",
        config=config,
        expected_manifest_sha256=TRAIN_MANIFEST_SHA256,
        expected_count=TRAIN_COUNT,
    )
    val_dataset: Dataset = CachedT1ceWtDataset(
        cache_root / "validation",
        split="validation",
        config=config,
        expected_manifest_sha256=VAL_MANIFEST_SHA256,
        expected_count=VAL_COUNT,
    )
    positive_voxels = [
        int(cache_indexes["train"]["per_case_positive_voxels"][case_id])
        for case_id in train_cached.case_ids
    ]
    train_dataset: Dataset = ForegroundThresholdReplacement(
        train_cached, positive_voxels, SOURCE_FOREGROUND_THRESHOLD
    )
    epochs = args.epochs
    # Smoke covers source step 20: 21 batches/rank x 12 x 2 = 504 cases.
    if args.mode == "smoke":
        train_dataset = Subset(train_dataset, list(range(504)))
        val_dataset = Subset(val_dataset, list(range(4)))
        epochs = 1

    run_identity = args.run_identity or RUN_IDENTITY
    output_arg = args.output_dir or f"workspace/BraTS/sammed3d_t1ce_wt/{run_identity}"
    output = Path(output_arg)
    if not output.is_absolute():
        output = ROOT / output
    if base.rank() == 0 and output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to reuse non-empty output: {output}")

    generic.seed_everything(TRAIN_SEED)
    requested = peft.make_adapter_config(
        target_token=False, lora_rank=0, model_type=engine.MODEL_TYPE
    )
    model, adapter_config = peft.load_model_with_adapters(
        base.BASE_CHECKPOINT,
        lambda model_type: engine.sam_model_registry3D[model_type](checkpoint=None),
        device=device,
        requested_config=requested,
        default_model_type=engine.MODEL_TYPE,
    )
    selection = base.configure_full_unfreeze(model)
    optimizer_args = argparse.Namespace(
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        unfreeze="full",
    )
    optimizer, optimizer_record = base.build_optimizer(selection, optimizer_args)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[120, 180], gamma=0.1
    )
    scaler = torch.amp.GradScaler("cuda")
    if base.is_dist():
        for tensor in model.state_dict().values():
            if torch.is_tensor(tensor):
                dist.broadcast(tensor, src=0)
    generic.seed_everything(TRAIN_SEED + base.rank())

    train_sampler = (
        DistributedSampler(
            train_dataset, shuffle=True, seed=0, drop_last=False
        )
        if base.is_dist()
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    # Upstream inference is batch-one and the RITM code itself only supports
    # batch one. Rank zero evaluates all development cases; rank one waits.
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=min(args.num_workers, 4),
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        if base.rank() == 0
        else None
    )
    if args.mode == "formal" and len(train_loader) != 41:
        raise RuntimeError(f"formal upstream loader must have 41 steps, got {len(train_loader)}")

    protocol = official_protocol_dict()
    protocol.pop("reused_training_loop_sha256", None)
    protocol["training_interaction"] = {
        "source_train_sha256": UPSTREAM_TRAIN_SHA256,
        "interactions": 11,
        "sparse_points": "current point only; multi_click default false",
        "recurrent_low_resolution_mask": True,
        "no_point_iterations": "one random integer in [2,8] and final index 10",
        "click_error_threshold": "raw logit > 0.5",
        "loss": "sum of 11 DiceCELoss values",
        "foreground_replacement_threshold": SOURCE_FOREGROUND_THRESHOLD,
    }
    protocol["validation_interaction"] = {
        "source_inference_sha256": UPSTREAM_INFERENCE_SHA256,
        "clicks": 5,
        "clicks_1_2": "RITM current point",
        "clicks_3_5": "uniform XOR current point",
        "recurrent_low_resolution_mask": True,
        "prediction_threshold": "sigmoid > 0.5 (raw logit > 0)",
        "primary_diagnostic": "mean per-case max Dice over clicks 1..5",
        "case_seed_extension": "SHA256(seed=2023,case_id) for stable monitoring",
    }
    run_config = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "run_identity": run_identity,
        "mode": args.mode,
        "world_size": base.world_size(),
        "per_gpu_batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "nominal_effective_batch_size": (
            args.batch_size * base.world_size() * args.accumulation_steps
        ),
        "source_first_update_quirk_effective_batch": (
            args.batch_size * base.world_size() * (args.accumulation_steps + 1)
        ),
        "optimizer": optimizer_record,
        "scheduler": {"type": "MultiStepLR", "milestones": [120, 180], "gamma": 0.1},
        "precision": "upstream FP16 autocast + GradScaler",
        "max_epochs": epochs,
        "train_seed_rank_base": TRAIN_SEED,
        "upstream_source_has_no_seed": False,
        "cache_train": train_cached.provenance(),
        "cache_validation": (
            val_dataset.dataset.provenance()
            if isinstance(val_dataset, Subset)
            else val_dataset.provenance()
        ),
        "dataset_and_interaction_protocol": protocol,
        "source_sha256": {
            "train_py": UPSTREAM_TRAIN_SHA256,
            "inference_py": UPSTREAM_INFERENCE_SHA256,
            "click_method_py": UPSTREAM_CLICK_SHA256,
        },
        "engineering_corrections": [
            "explicit mean gradient all-reduce because upstream bypasses DDP forward",
            "binary target cast to float only at DiceCELoss boundary for current MONAI compatibility",
            "official train.sh two-GPU launch retained with source per-GPU batch 12 (nominal batch 480)",
            "validation RNG isolated from training RNG",
            "atomic checkpoints and strict reload records added",
        ],
        "checkpoint_selection": {
            "upstream_primary": ["lowest train loss", "highest train Dice"],
            "added_development_diagnostic": "highest official-style validation best-of-5 Dice",
        },
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "base_checkpoint_sha256": base.BASE_CHECKPOINT_SHA256,
        "test_access": False,
    }
    if args.preflight_only:
        if base.rank() == 0:
            print(json.dumps({**run_config, "preflight_passed": True}, indent=2))
        if base.is_dist():
            dist.barrier(device_ids=[local_rank])
        os._exit(0)

    if base.rank() == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "train_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
        if args.mode == "smoke":
            (output / "SMOKE_ONLY.json").write_text(
                json.dumps(
                    {
                        "scientific_result_eligible": False,
                        "purpose": "11-interaction FP16 two-GPU formal accumulation finite update/save/reload",
                        "test_access": False,
                    },
                    indent=2,
                )
                + "\n"
            )
    if base.is_dist():
        dist.barrier(device_ids=[local_rank])

    loss_fn = generic.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
    history_path = output / "history.csv"
    best = {"train_loss": math.inf, "train_dice": -1.0, "val_best5": -1.0}
    best_epochs = {"train_loss": 0, "train_dice": 0, "val_best5": 0}
    for epoch in range(1, epochs + 1):
        train_result = train_epoch(
            model=model,
            loader=train_loader,
            sampler=train_sampler,
            epoch=epoch,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            selection=selection,
            loss_fn=loss_fn,
            accumulation_steps=args.accumulation_steps,
        )
        if args.mode == "smoke" and train_result["optimizer_applied_steps"] < 1:
            raise RuntimeError("smoke did not complete a finite optimizer update")
        if base.is_dist():
            dist.barrier(device_ids=[local_rank])
        validation_result = None
        if base.rank() == 0:
            assert val_loader is not None
            validation_result = validate_official_style(
                model=model, loader=val_loader, device=device
            )
        if base.is_dist():
            payload = torch.zeros(7, device=device, dtype=torch.float64)
            if base.rank() == 0:
                assert validation_result is not None
                payload[:] = torch.tensor(
                    [
                        validation_result["best_of_5_dice"],
                        validation_result["final_click_5_dice"],
                        validation_result["click_1_dice"],
                        validation_result["click_2_dice"],
                        validation_result["click_3_dice"],
                        validation_result["click_4_dice"],
                        validation_result["click_5_dice"],
                    ],
                    dtype=torch.float64,
                    device=device,
                )
            dist.broadcast(payload, src=0)
            if base.rank() != 0:
                validation_result = {
                    "best_of_5_dice": float(payload[0]),
                    "final_click_5_dice": float(payload[1]),
                    **{f"click_{i}_dice": float(payload[i + 1]) for i in range(1, 6)},
                }
        assert validation_result is not None
        scheduler.step()

        new_train_loss = train_result["loss_sum11"] < best["train_loss"]
        new_train_dice = train_result["dice_raw05_final11"] > best["train_dice"]
        new_val = validation_result["best_of_5_dice"] > best["val_best5"]
        if new_train_loss:
            best["train_loss"] = train_result["loss_sum11"]
            best_epochs["train_loss"] = epoch
        if new_train_dice:
            best["train_dice"] = train_result["dice_raw05_final11"]
            best_epochs["train_dice"] = epoch
        if new_val:
            best["val_best5"] = validation_result["best_of_5_dice"]
            best_epochs["val_best5"] = epoch

        if base.rank() == 0:
            row = {
                "epoch": epoch,
                "train_loss_sum11": train_result["loss_sum11"],
                "train_dice_raw05_final11": train_result["dice_raw05_final11"],
                "train_cases_per_sec": train_result["cases_per_sec"],
                "optimizer_steps": int(train_result["optimizer_steps"]),
                "optimizer_skipped_steps": int(train_result["optimizer_skipped_steps"]),
                "optimizer_applied_steps": int(train_result["optimizer_applied_steps"]),
                "threshold_replacements": int(train_result["threshold_replacements"]),
                "unused_epoch_num_clicks_draw": int(train_result["unused_epoch_num_clicks_draw"]),
                "val_official_best5_dice": validation_result["best_of_5_dice"],
                "val_official_final5_dice": validation_result["final_click_5_dice"],
                **{f"val_click{i}_dice": validation_result[f"click_{i}_dice"] for i in range(1, 6)},
                "new_train_loss_best": new_train_loss,
                "new_train_dice_best": new_train_dice,
                "new_val_best5": new_val,
                "encoder_lr": optimizer.param_groups[0]["lr"],
                "head_lr": optimizer.param_groups[1]["lr"],
                "grad_scale": scaler.get_scale(),
                "cuda_peak_memory_allocated": int(torch.cuda.max_memory_allocated()),
            }
            new_file = not history_path.exists()
            with open(history_path, "a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
            metadata = {
                "run_identity": run_identity,
                "epoch": epoch,
                "train_loss_sum11": train_result["loss_sum11"],
                "train_dice_raw05_final11": train_result["dice_raw05_final11"],
                "val_official_best5_dice": validation_result["best_of_5_dice"],
                "test_access": False,
            }
            save_state(
                output / "sam_model_latest.pth",
                model=model,
                epoch=epoch,
                adapter_config=adapter_config,
                metadata=metadata,
            )
            for selected, filename in (
                (new_train_loss, "sam_model_train_loss_best.pth"),
                (new_train_dice, "sam_model_train_dice_best.pth"),
                (new_val, "sam_model_val_official_best5.pth"),
            ):
                if selected:
                    save_state(
                        output / filename,
                        model=model,
                        epoch=epoch,
                        adapter_config=adapter_config,
                        metadata=metadata,
                    )
            (output / "best.json").write_text(
                json.dumps({"values": best, "epochs": best_epochs}, indent=2) + "\n"
            )
            print(json.dumps(row, sort_keys=True), flush=True)

    if base.rank() == 0:
        reloads = {}
        for key, filename in (
            ("train_loss", "sam_model_train_loss_best.pth"),
            ("train_dice", "sam_model_train_dice_best.pth"),
            ("val_best5", "sam_model_val_official_best5.pth"),
        ):
            reloads[key] = base.strict_reload_checkpoint(
                output / filename, best_epochs[key], "full"
            )
        (output / "selected_checkpoint_reload.json").write_text(
            json.dumps(reloads, indent=2, sort_keys=True) + "\n"
        )
        (output / "TRAIN_DONE.json").write_text(
            json.dumps(
                {
                    "schema": "aflow.sammed3d.t1ce_wt_upstream_literal_train_done",
                    "run_identity": run_identity,
                    "completed_epochs": epochs,
                    "best": best,
                    "best_epochs": best_epochs,
                    "scientific_result_eligible": args.mode == "formal",
                    "test_access": False,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"TRAIN_DONE output={output} best={best} epochs={best_epochs}", flush=True)

    for loader in (train_loader, val_loader):
        if loader is None:
            continue
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
    if base.is_dist():
        dist.barrier(device_ids=[local_rank])
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
