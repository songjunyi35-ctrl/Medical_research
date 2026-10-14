#!/usr/bin/env python
"""SAM-Med3D-official-recipe T1ce/WT arm: full unfreeze, cached preprocessing, 2 GPUs.

Scope
-----
This is an INDEPENDENT arm. It deliberately abandons the matched-comparison
basis shared by the rank-0, LoRA, and last-4 partial-unfreeze arms in order to
follow SAM-Med3D's own published training recipe, which is the most likely
origin of the BraTS 2024 SAM-Med3D baseline DSC of 74.1 +/- 1.5 that DPD-SAM
reports and that this project is trying to approach.

Relative to the matched arms this changes, simultaneously and by intent:
  * FULL image-encoder unfreeze (100,509,232 trainable, vs 35,951,920);
  * the discriminative-LR direction is INVERTED to match SAM-Med3D's train.py,
    where image_encoder takes the full LR and both heads take 0.1x -- the
    matched arms did the opposite;
  * N4 bias field correction added to preprocessing;
  * 50 -> 200 epochs with the official MultiStepLR([120, 180], gamma 0.1);
  * effective batch 2 -> 12 (per-GPU 2 x 2 GPUs x 3 accumulation steps);
  * single-GPU -> 2-GPU data parallelism with explicit gradient all-reduce.

Because several factors move together, NO result from this arm may be
attributed to any single factor, and it may not be compared against the matched
arms as if it were a controlled contrast.

On LR magnitude: SAM-Med3D's 8e-4 goes with ITS effective batch of 240
(batch_size 12 x accumulation_steps 20), which on 972 cases would give only ~4
optimizer steps per epoch. The absolute LR is therefore NOT transplantable; only
the direction and schedule are. The configured 1e-4/1e-5 is a conservative
sqrt-scaling of 8e-4 from effective batch 240 down to 12.

Unchanged from the frozen task definition: T1ce single channel, binary official
WT={1,2,3} with RC={4} excluded, whole-volume 128^3, 3 accumulated corrective
clicks with the recurrent low-resolution mask, raw logit > 0, validation seed
20260723, and the frozen 972 train / 108 validation subject-clean split.

The 270-case test is never read.

Launch via scripts/run_sammed3d_t1ce_wt_dpdsam_aligned.sh (torchrun).
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d import engine, partial_unfreeze, peft  # noqa: E402
from sammed3d.paths import SAMMED3D_CHECKPOINT  # noqa: E402
from sammed3d.t1ce_wt_cached import CachedT1ceWtDataset  # noqa: E402
from sammed3d.t1ce_wt_training import (  # noqa: E402
    SPATIAL_PREPROCESSING_CHOICES,
    TRAIN_AUGMENTATION_CHOICES,
    T1ceWtPreprocessConfig,
)
from scripts import train_sammed3d_brats as generic  # noqa: E402
from scripts.train_sammed3d_t1ce_wt import cumulative_interaction_forward  # noqa: E402

SCHEMA = "aflow.sammed3d.t1ce_wt_dpdsam_aligned_training"
SCHEMA_VERSION = 1
RUN_IDENTITY = "sammed3d_official_fullunfreeze_ddp2_train972_val108_e200_s20260805"

# Full image-encoder unfreeze was prohibited by the standing research protocol
# until 2026-08-05, when the operator lifted that prohibition explicitly after
# reviewing the evidence below. Recorded here so the change of regime is not
# silent:
#   * SAM-Med3D's own train.py trains image_encoder at the FULL learning rate and
#     both heads at 0.1x, i.e. the opposite discriminative direction from the
#     matched arms (their `lr * 0.1` line for the encoder is commented out).
#   * This project's own results are monotone in encoder adaptation:
#     rank-0 frozen 0.4785 -> LoRA PEFT 0.5214 -> blocks 8-11 already 0.5072 by
#     epoch 7 and still climbing.
#   * The prohibition was recorded with no supporting experiment, grouped with
#     threshold/TTA/morphology/postprocess -- which are test-hacking techniques,
#     a different category from a legitimate training choice.
FULL_UNFREEZE_AUTHORIZED_BY_OPERATOR_UTC = "2026-08-05"

BASE_CHECKPOINT = SAMMED3D_CHECKPOINT
BASE_CHECKPOINT_SHA256 = (
    "899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c"
)
SPLIT_DIR = Path(
    "data/datasets/brats2024_sammed3d_t1ce_wt_inner90_10_seed20260723_v1"
)
TRAIN_MANIFEST_SHA256 = (
    "ea21e2306321b71d821146ed01198f52b612d390c0736f8703f81a259c06c23e"
)
VAL_MANIFEST_SHA256 = (
    "fc750ab15266b8a53c6d9a7ff62bd043b10cc21733e9b68354a2a989fd1f488c"
)
TRAIN_COUNT = 972
VAL_COUNT = 108

TRAIN_SEED = 20260805
VALIDATION_SEED = 20260723


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_dist() else 0


def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def log(message: str) -> None:
    if rank() == 0:
        print(message, flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("smoke", "formal"), required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--cache-root", default="workspace/BraTS/cache/t1ce_wt_n4_128")
    p.add_argument(
        "--spatial-preprocessing",
        choices=SPATIAL_PREPROCESSING_CHOICES,
        default="whole_resize",
        help="dataset geometry descriptor; cached tensors remain authoritative",
    )
    p.add_argument(
        "--train-augmentation",
        choices=TRAIN_AUGMENTATION_CHOICES,
        default="affine_elastic_noise",
    )
    p.add_argument("--epochs", type=int, default=200)
    # Official direction: encoder LR ABOVE head LR (10:1), unlike the matched
    # arms. Magnitude is sqrt-scaled from the official 8e-4 at effective batch
    # 240 down to effective batch 12, taking the conservative side.
    p.add_argument("--encoder-lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--unfreeze", choices=("full", "last4"), default="full")
    p.add_argument("--batch-size", type=int, default=2, help="per-GPU batch size")
    p.add_argument("--accumulation-steps", type=int, default=3,
                   help="per-GPU batch x world size x this = effective batch")
    p.add_argument("--lr-milestones", type=int, nargs="*", default=[120, 180],
                   help="official MultiStepLR milestones")
    p.add_argument("--lr-gamma", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--min-epochs-before-stop", type=int, default=40)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--run-identity", default=None,
                   help="override RUN_IDENTITY for an independently named arm")
    p.add_argument("--preflight-only", action="store_true")
    return p.parse_args(argv)


ENCODER_NUMEL = 92_924_928
HEAD_NUMEL = 7_584_304          # prompt_encoder 8,668 + mask_decoder 7,575,636
TOTAL_NUMEL = 100_509_232


def configure_full_unfreeze(
    model: torch.nn.Module,
) -> partial_unfreeze.ParameterSelection:
    """Train every parameter, split into the official two LR groups.

    Mirrors SAM-Med3D's own train.py, which puts image_encoder in one AdamW group
    and prompt_encoder + mask_decoder in the lower-LR groups.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    encoder = tuple(model.image_encoder.parameters())
    heads = tuple(model.prompt_encoder.parameters()) + tuple(
        model.mask_decoder.parameters()
    )
    encoder_numel = int(sum(p.numel() for p in encoder))
    head_numel = int(sum(p.numel() for p in heads))
    total = int(sum(p.numel() for p in model.parameters()))
    if encoder_numel != ENCODER_NUMEL or head_numel != HEAD_NUMEL:
        raise RuntimeError(
            f"full-unfreeze split {encoder_numel}/{head_numel} != "
            f"{ENCODER_NUMEL}/{HEAD_NUMEL}"
        )
    if encoder_numel + head_numel != total or total != TOTAL_NUMEL:
        raise RuntimeError(
            "full unfreeze must cover every parameter exactly once: "
            f"{encoder_numel}+{head_numel} != {total}"
        )
    record = {
        "update": "full_image_encoder_unfreeze",
        "image_encoder": {"trainable_blocks": "all", "frozen_blocks": []},
        "counts": {"trainable_parameters": total, "total_parameters": total},
        "authorized_by_operator_utc": FULL_UNFREEZE_AUTHORIZED_BY_OPERATOR_UTC,
    }
    return partial_unfreeze.ParameterSelection(
        encoder_parameters=encoder, head_parameters=heads, record=record
    )


def build_optimizer(
    selection: partial_unfreeze.ParameterSelection, args: argparse.Namespace
) -> tuple[torch.optim.AdamW, dict[str, Any]]:
    """Two discriminative-LR groups at this arm's LRs.

    `partial_unfreeze.build_adamw` is not reused because it hardcodes the matched
    arm's 1e-5/1e-4 and its module is byte-pinned by that arm's preregistration.
    Only the parameter *selection* is reused.
    """
    groups = [
        {
            "params": list(selection.encoder_parameters),
            "lr": args.encoder_lr,
            "weight_decay": args.weight_decay,
            "group_name": (
                "image_encoder_full" if args.unfreeze == "full"
                else "image_encoder_blocks_8_11"
            ),
        },
        {
            "params": list(selection.head_parameters),
            "lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "group_name": "prompt_encoder_and_mask_decoder",
        },
    ]
    optimizer = torch.optim.AdamW(groups)
    record = {
        "type": "AdamW",
        "groups": [
            {
                "name": g["group_name"],
                "lr": float(g["lr"]),
                "weight_decay": float(g["weight_decay"]),
                "parameter_tensors": len(g["params"]),
                "parameter_numel": int(sum(p.numel() for p in g["params"])),
            }
            for g in groups
        ],
    }
    # Fail closed on the parameter contract for the selected unfreeze policy.
    expected_encoder = ENCODER_NUMEL if args.unfreeze == "full" else 28_367_616
    if record["groups"][0]["parameter_numel"] != expected_encoder:
        raise RuntimeError(f"encoder group numel != {expected_encoder:,}")
    if record["groups"][1]["parameter_numel"] != HEAD_NUMEL:
        raise RuntimeError(f"head group numel != {HEAD_NUMEL:,}")
    if record["groups"][0]["lr"] <= record["groups"][1]["lr"]:
        raise RuntimeError(
            "this arm follows SAM-Med3D's direction: encoder LR must exceed head LR"
        )
    ids: set[int] = set()
    for g in groups:
        for prm in g["params"]:
            if id(prm) in ids:
                raise RuntimeError("optimizer parameter groups overlap")
            ids.add(id(prm))
    return optimizer, record


def average_gradients(selection: partial_unfreeze.ParameterSelection) -> None:
    """Average trainable gradients across ranks.

    DistributedDataParallel is deliberately NOT used. `cumulative_interaction_forward`
    invokes `model.image_encoder`, `model.prompt_encoder` and `model.mask_decoder`
    directly rather than calling `model(...)`, so DDP's forward would never run
    and its gradient-sync hooks would never fire. Explicitly all-reducing the
    gradients gives the same mathematics -- each rank holds identical weights and
    computes gradients on a disjoint shard, so the mean over ranks equals the
    gradient of the combined effective batch.
    """
    if not is_dist():
        return
    params = list(selection.trainable_parameters)
    device = params[0].device
    size = float(world_size())

    # Some trainable tensors are structurally unused by this forward, e.g.
    # prompt_encoder.no_mask_embed, because a recurrent low-resolution mask is
    # always supplied. Their grad is None on every rank.
    #
    # Skipping them per-rank would desynchronize the collective sequence and
    # deadlock, while zero-filling them would let AdamW's decoupled weight decay
    # shrink parameters that single-GPU training leaves untouched. So first agree
    # globally on which parameters were used, then reduce only those.
    local = torch.tensor(
        [1.0 if p.grad is not None else 0.0 for p in params], device=device
    )
    dist.all_reduce(local, op=dist.ReduceOp.SUM)
    used = local > 0

    for position, parameter in enumerate(params):
        if not bool(used[position]):
            # unused on every rank: leave grad None so the optimizer skips it
            parameter.grad = None
            continue
        if parameter.grad is None:
            # used somewhere else this step; contribute an explicit zero
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad /= size


def ensure_finite_gradients(selection: partial_unfreeze.ParameterSelection) -> None:
    for name, params in (
        ("image_encoder_blocks_8_11", selection.encoder_parameters),
        ("prompt_encoder_and_mask_decoder", selection.head_parameters),
    ):
        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            raise RuntimeError(f"{name} produced no gradients")
        if any(not bool(torch.isfinite(g).all()) for g in grads):
            raise RuntimeError(f"{name} has non-finite gradients")


def strict_reload_checkpoint(
    path: Path, expected_epoch: int, unfreeze: str
) -> dict[str, Any]:
    """Prove the selected checkpoint reloads bit-for-bit into a fresh model."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload["epoch"]) != int(expected_epoch):
        raise RuntimeError(
            f"selected checkpoint epoch {payload['epoch']} != {expected_epoch}"
        )
    model, adapter_config = peft.load_model_with_adapters(
        path,
        lambda mt: engine.sam_model_registry3D[mt](checkpoint=None),
        device="cpu",
        default_model_type=engine.MODEL_TYPE,
    )
    if peft.adapters_enabled(adapter_config):
        raise RuntimeError("reloaded checkpoint unexpectedly has adapters enabled")
    # Re-apply THIS arm's policy, not the matched arm's, or the record would
    # claim a trainable count the run never used.
    if unfreeze == "full":
        selection = configure_full_unfreeze(model)
        expected_trainable = TOTAL_NUMEL
    else:
        selection = partial_unfreeze.configure_vit_b_ori_last_four(model)
        expected_trainable = 35_951_920
    trainable = int(sum(p.numel() for p in selection.trainable_parameters))
    total = int(sum(p.numel() for p in model.parameters()))
    if trainable != expected_trainable or total != TOTAL_NUMEL:
        raise RuntimeError(f"reloaded parameter contract mismatch {trainable}/{total}")
    saved = payload["model_state_dict"]
    current = model.state_dict()
    if set(saved) != set(current):
        raise RuntimeError("reloaded state-dict key set differs from the checkpoint")
    differing = [k for k in saved if not torch.equal(saved[k].cpu(), current[k].cpu())]
    if differing:
        raise RuntimeError(f"{len(differing)} tensors differ after strict reload")
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "epoch": int(payload["epoch"]),
        "strict_state_load": True,
        "bitwise_identical": True,
        "adapters_enabled": False,
        "unfreeze_policy": unfreeze,
        "trainable_parameters": trainable,
        "total_parameters": total,
    }


def run_epoch(
    *,
    model: torch.nn.Module,
    core: torch.nn.Module,
    loader: DataLoader,
    sampler: DistributedSampler | None,
    epoch: int,
    loss_fn: torch.nn.Module,
    device: torch.device,
    selection: partial_unfreeze.ParameterSelection,
    optimizer: torch.optim.Optimizer | None,
    amp_dtype: torch.dtype | None,
    accumulation_steps: int = 1,
) -> dict[str, float]:
    training = optimizer is not None
    # Only the training pass goes through the DDP wrapper. Running eval forwards
    # through DDP would perturb its static_graph bookkeeping for no benefit,
    # since no_grad produces no gradients to synchronize.
    net = model if training else core
    net.train(training)
    if sampler is not None and training:
        sampler.set_epoch(epoch)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    dice_sum = torch.zeros((), device=device, dtype=torch.float64)
    count = torch.zeros((), device=device, dtype=torch.float64)
    start = time.perf_counter()
    if training:
        optimizer.zero_grad(set_to_none=True)
    total_steps = len(loader)
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for step, batch in enumerate(loader):
            image = batch["image"].to(device, non_blocking=True).float()
            target = batch["label"].to(device, non_blocking=True).float()
            autocast = (
                torch.amp.autocast("cuda", dtype=amp_dtype)
                if amp_dtype is not None and device.type == "cuda"
                else nullcontext()
            )
            with autocast:
                logits, loss = cumulative_interaction_forward(
                    model=net,
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
                # Accumulate locally; synchronize across ranks only at the step
                # boundary. Every rank has the same len(loader) because the
                # sampler shards evenly, so the collective sequence stays aligned.
                (loss / accumulation_steps).backward()
                if (step + 1) % accumulation_steps == 0 or (step + 1) == total_steps:
                    average_gradients(selection)
                    ensure_finite_gradients(selection)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            scores = generic.dice_scores_from_logits(logits.detach(), target)
            if any(s is None or not math.isfinite(s) for s in scores):
                raise RuntimeError("non-finite or undefined Dice")
            bs = int(target.shape[0])
            loss_sum += float(loss.detach()) * bs
            dice_sum += float(sum(float(s) for s in scores if s is not None))
            count += bs
    if is_dist():
        packed = torch.stack([loss_sum, dice_sum, count])
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        loss_sum, dice_sum, count = packed[0], packed[1], packed[2]
    n = float(count)
    if n <= 0:
        raise RuntimeError("epoch processed zero samples")
    elapsed = time.perf_counter() - start
    result = {
        "loss": float(loss_sum) / n,
        "dice": float(dice_sum) / n,
        "samples": n,
        "elapsed_sec": elapsed,
        "cases_per_sec": n / elapsed,
    }
    if not all(math.isfinite(result[k]) for k in ("loss", "dice")):
        raise RuntimeError("non-finite epoch metric")
    return result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group(backend="nccl", timeout=_dt.timedelta(minutes=30))
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("this arm requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA device does not support BF16")

    run_identity = args.run_identity or RUN_IDENTITY
    out_rel = args.output_dir or f"workspace/BraTS/sammed3d_t1ce_wt/{run_identity}"
    output = ROOT / out_rel if not Path(out_rel).is_absolute() else Path(out_rel)
    if rank() == 0 and output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to reuse non-empty output path: {output}")

    if sha256_file(BASE_CHECKPOINT) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("untouched base checkpoint SHA mismatch")

    cache_root = ROOT / args.cache_root if not Path(args.cache_root).is_absolute() else Path(args.cache_root)
    config = T1ceWtPreprocessConfig(
        spatial_preprocessing=args.spatial_preprocessing,
        train_augmentation=args.train_augmentation,
    )
    train_ds = CachedT1ceWtDataset(
        cache_root / "train", split="train", config=config,
        expected_manifest_sha256=TRAIN_MANIFEST_SHA256, expected_count=TRAIN_COUNT,
    )
    val_ds = CachedT1ceWtDataset(
        cache_root / "validation", split="validation", config=config,
        expected_manifest_sha256=VAL_MANIFEST_SHA256, expected_count=VAL_COUNT,
    )
    if args.mode == "smoke":
        from torch.utils.data import Subset

        train_ds = Subset(train_ds, list(range(4)))
        val_ds = Subset(val_ds, list(range(4)))
        epochs = 1
    else:
        epochs = args.epochs

    generic.seed_everything(TRAIN_SEED)
    adapter_request = peft.make_adapter_config(
        target_token=False, lora_rank=0, model_type=engine.MODEL_TYPE
    )
    model, adapter_config = peft.load_model_with_adapters(
        BASE_CHECKPOINT,
        lambda mt: engine.sam_model_registry3D[mt](checkpoint=None),
        device=device,
        requested_config=adapter_request,
        default_model_type=engine.MODEL_TYPE,
    )
    if peft.adapters_enabled(adapter_config):
        raise RuntimeError("this arm requires all adapters disabled")
    if args.unfreeze == "full":
        selection = configure_full_unfreeze(model)
        expected_trainable = TOTAL_NUMEL
    else:
        selection = partial_unfreeze.configure_vit_b_ori_last_four(model)
        expected_trainable = 35_951_920
    optimizer, optimizer_record = build_optimizer(selection, args)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(args.lr_milestones), gamma=args.lr_gamma
    )
    trainable = int(sum(p.numel() for p in selection.trainable_parameters))
    total = int(sum(p.numel() for p in model.parameters()))
    if trainable != expected_trainable or total != TOTAL_NUMEL:
        raise RuntimeError(f"parameter contract mismatch: {trainable}/{total}")

    core = model
    if is_dist():
        # Broadcast rank 0's weights so every rank provably starts identical,
        # which is what makes plain gradient averaging equivalent to one large
        # batch. See average_gradients() for why DDP itself is not used.
        for tensor in model.state_dict().values():
            if torch.is_tensor(tensor):
                dist.broadcast(tensor, src=0)

    # Reset RNG after model/optimizer construction so augmentation and click
    # sampling start from a defined post-construction state. The per-rank offset
    # keeps the two shards from drawing an identical augmentation/click stream;
    # DistributedSampler seeds its shuffle independently, so ordering is
    # unaffected by this offset.
    generic.seed_everything(TRAIN_SEED + rank())

    train_sampler = DistributedSampler(
        train_ds, shuffle=True, seed=TRAIN_SEED, drop_last=False
    ) if is_dist() else None
    val_sampler = DistributedSampler(
        val_ds, shuffle=False, drop_last=False
    ) if is_dist() else None
    if val_sampler is not None and val_sampler.num_samples * world_size() != len(val_ds):
        raise RuntimeError(
            "validation set does not shard evenly; padding would bias the metric"
        )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers, pin_memory=True,
        drop_last=False, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    amp_dtype = torch.bfloat16
    loss_fn = generic.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")

    run_config = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "run_identity": run_identity,
        "mode": args.mode,
        "world_size": world_size(),
        "unfreeze_policy": args.unfreeze,
        "partial_unfreeze_policy": selection.record,
        "per_gpu_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.accumulation_steps,
        "effective_batch_size": args.batch_size * world_size() * args.accumulation_steps,
        "lr_scheduler": {
            "type": "MultiStepLR",
            "milestones": list(args.lr_milestones),
            "gamma": args.lr_gamma,
            "source": "SAM-Med3D official train.py defaults",
        },
        "gradient_sync": "explicit all-reduce mean over ranks (not DDP; forward calls submodules directly)",
        "per_rank_rng_offset": True,
        "max_epochs": epochs,
        "optimizer": optimizer_record,
        "precision": "bf16",
        "train_seed": TRAIN_SEED,
        "validation_seed": VALIDATION_SEED,
        "num_workers": args.num_workers,
        "min_epochs_before_stop": args.min_epochs_before_stop,
        "patience_complete_epochs": args.patience,
        "selection_metric": "val_dice",
        "trainable_parameters": trainable,
        "total_parameters": total,
        "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "cache_train": train_ds.dataset.provenance() if args.mode == "smoke" else train_ds.provenance(),
        "cache_validation": val_ds.dataset.provenance() if args.mode == "smoke" else val_ds.provenance(),
        "dataset_runtime_preprocessing": config.to_protocol_dict(),
        "test_access": False,
        "matched_comparison_valid": False,
        "attribution_warning": (
            "This independent arm may differ in cached geometry, augmentation, "
            "optimization, epochs, batch, or world size. Attribute results only "
            "to the complete recorded protocol unless an explicit matched control "
            "holds every other field fixed."
        ),
    }
    if args.preflight_only:
        run_config["preflight_passed"] = True
        run_config["output_created"] = False
        log(json.dumps(run_config, indent=2, sort_keys=True))
        # Same NCCL teardown hang as at the end of training; exit directly.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    if rank() == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "train_config.json").write_text(
            json.dumps(run_config, indent=2, sort_keys=True) + "\n"
        )
        if args.mode == "smoke":
            (output / "SMOKE_ONLY.json").write_text(json.dumps({
                "purpose": "finite forward/backward/save/reload and DDP wiring only",
                "resume_allowed": False,
                "scientific_result_eligible": False,
            }, indent=2) + "\n")
    if is_dist():
        dist.barrier()

    history_path = output / "history.csv"
    best = {"epoch": 0, "val_dice": -1.0}
    stale = 0
    for epoch in range(1, epochs + 1):
        tr = run_epoch(
            model=model, core=core, loader=train_loader, sampler=train_sampler,
            epoch=epoch, loss_fn=loss_fn, device=device, selection=selection,
            optimizer=optimizer, amp_dtype=amp_dtype,
            accumulation_steps=args.accumulation_steps,
        )
        va = run_epoch(
            model=model, core=core, loader=val_loader, sampler=val_sampler,
            epoch=epoch, loss_fn=loss_fn, device=device, selection=selection,
            optimizer=None, amp_dtype=amp_dtype,
        )
        scheduler.step()
        strict_new_high = va["dice"] > best["val_dice"]
        if strict_new_high:
            best = {"epoch": epoch, "val_dice": va["dice"]}
            stale = 0
        else:
            stale += 1

        if rank() == 0:
            row = {
                "epoch": epoch,
                "train_loss": tr["loss"], "train_dice": tr["dice"],
                "train_cases_per_sec": tr["cases_per_sec"],
                "val_loss": va["loss"], "val_dice": va["dice"],
                "val_cases_per_sec": va["cases_per_sec"],
                "strict_new_high": strict_new_high,
                "stale_complete_epochs": stale,
                "encoder_lr": optimizer.param_groups[0]["lr"],
                "head_lr": optimizer.param_groups[1]["lr"],
                "cuda_peak_memory_allocated": int(torch.cuda.max_memory_allocated()),
            }
            new_file = not history_path.exists()
            with open(history_path, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
            print(json.dumps(row, sort_keys=True), flush=True)

            state = {
                "epoch": epoch,
                "model_state_dict": core.state_dict(),
                "adapter_config": adapter_config,
                "checkpoint_metadata": {
                    "run_identity": run_identity,
                    "epoch": epoch,
                    "val_dice": va["dice"],
                    "effective_batch_size": args.batch_size * world_size(),
                    "head_lr": args.head_lr,
                    "encoder_lr": args.encoder_lr,
                    "test_access": False,
                },
            }
            tmp = output / "sam_model_latest.pth.tmp"
            torch.save(state, tmp)
            tmp.replace(output / "sam_model_latest.pth")
            if strict_new_high:
                tmp2 = output / "sam_model_val_dice_best.pth.tmp"
                torch.save(state, tmp2)
                tmp2.replace(output / "sam_model_val_dice_best.pth")
                (output / "best.json").write_text(json.dumps({
                    "selection": {
                        "checkpoint": "sam_model_val_dice_best.pth",
                        "epoch": epoch, "metric": "val_dice", "value": va["dice"],
                    }
                }, indent=2) + "\n")

        stop = torch.tensor(
            [1 if (epoch >= args.min_epochs_before_stop and stale >= args.patience) else 0],
            device=device,
        )
        if is_dist():
            dist.broadcast(stop, src=0)
        if int(stop.item()):
            log(f"early stop at epoch {epoch}: {stale} stale complete epochs")
            break

    if rank() == 0:
        reload_record = strict_reload_checkpoint(
            output / "sam_model_val_dice_best.pth", best["epoch"], args.unfreeze
        )
        (output / "selected_checkpoint_reload.json").write_text(
            json.dumps(reload_record, indent=2, sort_keys=True) + "\n"
        )
        (output / "TRAIN_DONE.json").write_text(json.dumps({
            "schema": "aflow.sammed3d.t1ce_wt_dpdsam_aligned_train_done",
            "run_identity": run_identity,
            "mode": args.mode,
            "completed_epochs": epoch,
            "selected_epoch": best["epoch"],
            "selected_val_dice": best["val_dice"],
            "scientific_result_eligible": args.mode == "formal",
            "matched_comparison_valid": False,
            "test_access": False,
        }, indent=2) + "\n")
        print(f"TRAIN_DONE output={output} selected_epoch={best['epoch']} "
              f"selected_val_dice={best['val_dice']:.6f}", flush=True)

    # Shut down cleanly. Two things otherwise leave the job hung after
    # TRAIN_DONE, holding both GPUs indefinitely and never signalling
    # completion (observed: 30 processes alive 45 minutes past the final epoch):
    #   1. persistent_workers keeps dataloader workers alive while the loaders
    #      are still referenced, so the parent never exits;
    #   2. a NCCL barrier without device_ids can hang, which torch itself warns
    #      about at runtime.
    for loader in (train_loader, val_loader):
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
    del train_loader, val_loader

    if is_dist():
        # Barrier first so no rank exits before rank 0 has flushed its artifacts.
        dist.barrier(device_ids=[local_rank])

    # Deliberately NOT calling dist.destroy_process_group(): py-spy showed both
    # ranks wedged inside it (distributed_c10d.py:2135) with every epoch, the
    # reload check and all artifacts already complete -- a NCCL communicator
    # teardown hang. The process is terminating anyway, so a graceful teardown
    # buys nothing and the OS reclaims the GPU context. Exiting directly is what
    # keeps a finished 200-epoch run from holding both GPUs forever and never
    # signalling completion.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
