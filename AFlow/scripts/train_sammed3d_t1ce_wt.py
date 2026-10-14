#!/usr/bin/env python3
"""Dedicated T1ce/WT cumulative-click fine-tuning entry point.

The historical generic trainer is protocol-hash anchored. This runner reuses
its optimizer/checkpoint machinery while isolating the new data, normalization,
and interaction contract from that immutable implementation.
"""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d import engine, peft  # noqa: E402
from sammed3d.t1ce_wt_training import (  # noqa: E402
    SPATIAL_PREPROCESSING_CHOICES,
    TRAIN_AUGMENTATION_CHOICES,
    T1ceWtPreprocessConfig,
    T1ceWtTrainingDataset,
    identity_preprocessed_batch,
)
from scripts import train_sammed3d_brats as generic  # noqa: E402


PROTOCOL_SCHEMA = "aflow.sammed3d.t1ce_wt_cumulative3_training"
PROTOCOL_VERSION = 3
GENERIC_TRAINER_SHA256 = (
    "0c5026e98b2e0f848c16b436610e9b7eca391995c062ac128ce1bee26d1bfa4f"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def torchio_z_normalize_batch(image: torch.Tensor) -> torch.Tensor:
    """Match TorchIO ZNormalization(masking_method=lambda x: x > 0)."""
    out = image.clone()
    for index in range(out.shape[0]):
        mask = out[index] > 0
        if not bool(mask.any()):
            continue
        values = out[index][mask]
        mean = values.mean()
        std = values.std(unbiased=True).clamp_min(1e-6)
        out[index] = (out[index] - mean) / std
    return out


def sample_uniform_error_click(
    logits: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample uniformly from GT/XOR using the frozen raw-logit >0 threshold."""
    prediction = logits > 0
    true = target > 0
    false_negative = torch.logical_and(true, torch.logical_not(prediction))
    false_positive = torch.logical_and(torch.logical_not(true), prediction)
    points = []
    labels = []
    for index in range(logits.shape[0]):
        error = torch.logical_or(
            false_negative[index, 0], false_positive[index, 0]
        )
        candidates = torch.argwhere(error)
        if len(candidates) > 0:
            point = candidates[np.random.randint(len(candidates))]
            positive = bool(
                false_negative[
                    index, 0, point[0], point[1], point[2]
                ]
            )
        else:
            foreground = torch.argwhere(true[index, 0])
            if len(foreground) == 0:
                raise ValueError("WT target is unexpectedly empty")
            point = foreground[np.random.randint(len(foreground))]
            positive = True
        points.append(point.reshape(1, 1, 3))
        labels.append(
            torch.tensor(
                [[int(positive)]],
                device=logits.device,
                dtype=torch.long,
            )
        )
    return (
        torch.cat(points, dim=0).to(logits.device),
        torch.cat(labels, dim=0),
    )


def cumulative_interaction_forward(
    model: torch.nn.Module,
    image: torch.Tensor,
    gt: torch.Tensor,
    loss_fn: torch.nn.Module,
    num_clicks: int,
    click_method: str,
    train_image_encoder: bool,
    target_ids: torch.Tensor | None = None,
    return_per_sample_losses: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]:
    if num_clicks != 3 or click_method != "random":
        raise ValueError("T1ce/WT protocol requires exactly 3 uniform clicks")
    if train_image_encoder:
        image_embedding = model.image_encoder(image)
    else:
        with torch.no_grad():
            image_embedding = model.image_encoder(image)
        image_embedding = image_embedding.detach()

    previous_logits = torch.zeros_like(gt, dtype=torch.float32)
    low_res = F.interpolate(
        previous_logits, size=(engine.CROP_SIZE // 4,) * 3
    )
    total_loss = image.sum() * 0.0
    per_sample_loss = (
        image.new_zeros(image.shape[0]) if return_per_sample_losses else None
    )
    point_history: list[torch.Tensor] = []
    label_history: list[torch.Tensor] = []
    for _ in range(num_clicks):
        points, labels = sample_uniform_error_click(
            previous_logits.detach(), gt.detach()
        )
        point_history.append(points)
        label_history.append(labels)
        prompt_points = torch.cat(point_history, dim=1)
        prompt_labels = torch.cat(label_history, dim=1)
        sparse, dense = model.prompt_encoder(
            points=(prompt_points, prompt_labels),
            boxes=None,
            masks=low_res,
        )
        if target_ids is not None:
            sparse = peft.append_target_token(model, sparse, target_ids)
        low_res, _ = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        previous_logits = F.interpolate(
            low_res,
            size=gt.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
        total_loss = total_loss + loss_fn(previous_logits, gt)
        if per_sample_loss is not None:
            for sample_index in range(image.shape[0]):
                per_sample_loss[sample_index] += loss_fn(
                    previous_logits[sample_index : sample_index + 1],
                    gt[sample_index : sample_index + 1],
                )
    mean_loss = total_loss / num_clicks
    if per_sample_loss is not None:
        return previous_logits, mean_loss, per_sample_loss / num_clicks
    return previous_logits, mean_loss


def add_preprocessing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--spatial-preprocessing",
        choices=SPATIAL_PREPROCESSING_CHOICES,
        default="gt_crop",
    )
    parser.add_argument(
        "--train-augmentation",
        choices=TRAIN_AUGMENTATION_CHOICES,
        default="none",
    )


def strip_wrapper_only_args(argv: list[str]) -> list[str]:
    parser = argparse.ArgumentParser(add_help=False)
    add_preprocessing_args(parser)
    _, generic_argv = parser.parse_known_args(argv)
    return generic_argv


def parse_contract_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--val-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path(engine.DEFAULT_CKPT))
    parser.add_argument("--strategy", default="decoder_only")
    parser.add_argument("--num-clicks", type=int, default=5)
    parser.add_argument("--click-method", default="default")
    parser.add_argument("--target-token", action="store_true")
    parser.add_argument("--encoder-qv-lora-rank", type=int, default=0)
    parser.add_argument(
        "--encoder-qv-lora-alpha",
        type=float,
        default=peft.DEFAULT_LORA_ALPHA,
    )
    parser.add_argument("--overfit", action="store_true")
    add_preprocessing_args(parser)
    args, _ = parser.parse_known_args(argv)
    if args.strategy != "decoder_only":
        raise ValueError("T1ce/WT protocol requires strategy=decoder_only")
    if args.num_clicks != 3 or args.click_method != "random":
        raise ValueError("T1ce/WT protocol requires num_clicks=3 and click_method=random")
    if args.target_token:
        raise ValueError("T1ce/WT protocol does not use target tokens")
    if args.encoder_qv_lora_rank < 0:
        raise ValueError("--encoder-qv-lora-rank must be >= 0")
    if args.encoder_qv_lora_rank > 0 and args.encoder_qv_lora_alpha <= 0:
        raise ValueError("--encoder-qv-lora-alpha must be > 0 when LoRA is enabled")
    if args.overfit and args.train_augmentation != "none":
        raise ValueError(
            "--overfit cannot be combined with stochastic augmentation because "
            "validation must remain unaugmented"
        )
    return args


def protocol_record(args: argparse.Namespace) -> dict:
    generic_path = Path(generic.__file__).resolve()
    actual_generic_sha = sha256_file(generic_path)
    if actual_generic_sha != GENERIC_TRAINER_SHA256:
        raise RuntimeError(
            f"generic trainer hash changed: {actual_generic_sha}"
        )
    return {
        "schema": PROTOCOL_SCHEMA,
        "schema_version": PROTOCOL_VERSION,
        "task": "single-channel T1ce to binary WT",
        "target": {
            "foreground_raw_labels": [1, 2, 3],
            "excluded_raw_labels": [4],
        },
        "preprocessing": T1ceWtPreprocessConfig(
            spatial_preprocessing=args.spatial_preprocessing,
            train_augmentation=args.train_augmentation,
        ).to_protocol_dict(),
        "interaction": {
            "num_clicks": 3,
            "first_click": "uniform GT foreground positive",
            "later_clicks": "uniform prediction XOR GT",
            "corrective_labels": "FN positive; FP negative",
            "prediction_threshold": "raw logit > 0 (sigmoid > 0.5)",
            "points": "all prior points accumulated",
            "mask_prompt": "previous low-resolution mask recurrently supplied",
        },
        "loss": "mean MONAI DiceCELoss over all three click predictions",
        "selection": "validation final-click scan-macro Dice",
        "test_access_during_training": False,
        "optimization": {
            "strategy": args.strategy,
            "prompt_encoder_trainable": True,
            "mask_decoder_trainable": True,
            "image_encoder": {
                "update": (
                    "qv_lora"
                    if args.encoder_qv_lora_rank > 0
                    else "frozen"
                ),
                "rank": args.encoder_qv_lora_rank,
                "alpha": args.encoder_qv_lora_alpha,
                "blocks": (
                    list(peft.DEFAULT_LORA_BLOCKS)
                    if args.encoder_qv_lora_rank > 0
                    else []
                ),
                "projections": (
                    ["q", "v"] if args.encoder_qv_lora_rank > 0 else []
                ),
            },
        },
        "source": {
            "wrapper": str(Path(__file__).resolve()),
            "wrapper_sha256": sha256_file(Path(__file__).resolve()),
            "generic_trainer": str(generic_path),
            "generic_trainer_sha256": actual_generic_sha,
            "train_manifest": str(args.train_jsonl.resolve()),
            "train_manifest_sha256": sha256_file(args.train_jsonl),
            "validation_manifest": str(args.val_jsonl.resolve()),
            "validation_manifest_sha256": sha256_file(args.val_jsonl),
            "base_checkpoint": str(args.checkpoint.resolve()),
            "base_checkpoint_sha256": sha256_file(args.checkpoint),
        },
    }


def install_protocol(record: dict, args: argparse.Namespace) -> None:
    config = T1ceWtPreprocessConfig(
        spatial_preprocessing=args.spatial_preprocessing,
        train_augmentation=args.train_augmentation,
    )
    generic.BratsJsonlDataset = partial(
        T1ceWtTrainingDataset,
        config=config,
    )
    generic.z_normalize_batch = identity_preprocessed_batch
    generic.interaction_forward = cumulative_interaction_forward
    original_prepare = generic.prepare_output_dir

    def prepare_with_contract(path: Path, allow_nonempty: bool) -> None:
        original_prepare(path, allow_nonempty)
        contract_path = path / "t1ce_wt_protocol.json"
        if contract_path.exists() and allow_nonempty:
            existing = json.loads(contract_path.read_text(encoding="utf-8"))
            if existing != record:
                raise RuntimeError("existing T1ce/WT protocol record differs")
        else:
            contract_path.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    generic.prepare_output_dir = prepare_with_contract


def main() -> None:
    original_argv = sys.argv[1:]
    args = parse_contract_args(original_argv)
    record = protocol_record(args)
    install_protocol(record, args)
    sys.argv = [sys.argv[0], *strip_wrapper_only_args(original_argv)]
    generic.main()


if __name__ == "__main__":
    main()
