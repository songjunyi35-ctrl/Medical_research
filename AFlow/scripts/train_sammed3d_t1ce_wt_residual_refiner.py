#!/usr/bin/env python3
"""Train a T1ce + frozen SAM-logit 3D residual refinement network.

The network predicts an additive correction to the frozen stage-2 SAM logits.
Its final convolution is zero-initialized, so epoch zero exactly reproduces the
SAM prediction.  This is a development-only runner and has no test-data path.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monai.losses import DiceCELoss  # noqa: E402
from monai.networks.nets import SegResNet  # noqa: E402
from sammed3d.t1ce_wt_cached import MODEL_SHAPE  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_residual_refiner_training"
SCHEMA_VERSION = 1
INPUT_CACHE_SCHEMA = "aflow.sammed3d.t1ce_wt_refiner_input_cache"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class RefinerDataset(Dataset):
    def __init__(self, refiner_cache: Path) -> None:
        self.refiner_cache = refiner_cache.resolve()
        self.index = json.loads((self.refiner_cache / "index.json").read_text())
        if self.index.get("schema") != INPUT_CACHE_SCHEMA:
            raise ValueError("unexpected refiner input cache schema")
        if self.index.get("test_access") is not False:
            raise ValueError("refiner training cannot consume test data")
        self.count = int(self.index["count"])
        self.case_ids = list(self.index["case_ids"])
        source = Path(self.index["source_cache"]["path"])
        if not source.is_absolute():
            source = ROOT / source
        source = source.resolve()
        source_index = json.loads((source / "index.json").read_text())
        if source_index["images_sha256"] != self.index["source_cache"]["images_sha256"]:
            raise RuntimeError("source image cache hash mismatch")
        if source_index["labels_sha256"] != self.index["source_cache"]["labels_sha256"]:
            raise RuntimeError("source label cache hash mismatch")
        if list(source_index["case_ids"][: self.count]) != self.case_ids:
            raise RuntimeError("source/refiner case ordering mismatch")
        self.source = source
        self._images: np.memmap | None = None
        self._labels: np.memmap | None = None
        self._logits: np.memmap | None = None

    def _open(self) -> None:
        if self._images is None:
            self._images = np.memmap(
                self.source / "images.f32",
                dtype=np.float32,
                mode="r",
                shape=(self.count, *MODEL_SHAPE),
            )
            self._labels = np.memmap(
                self.source / "labels.u8",
                dtype=np.uint8,
                mode="r",
                shape=(self.count, *MODEL_SHAPE),
            )
            self._logits = np.memmap(
                self.refiner_cache / "sam_logits.f16",
                dtype=np.float16,
                mode="r",
                shape=(self.count, *MODEL_SHAPE),
            )

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, slot: int) -> dict[str, torch.Tensor | str]:
        self._open()
        assert self._images is not None and self._labels is not None and self._logits is not None
        image = torch.from_numpy(np.array(self._images[slot], dtype=np.float32))[None]
        coarse = torch.from_numpy(np.array(self._logits[slot], dtype=np.float32))[None]
        label = torch.from_numpy(np.array(self._labels[slot], dtype=np.float32))[None]
        return {"image": image, "coarse": coarse, "label": label, "case_id": self.case_ids[slot]}

    def sampling_weights(self) -> torch.Tensor:
        self._open()
        assert self._labels is not None
        volumes = np.asarray(
            [float(np.count_nonzero(self._labels[i])) for i in range(self.count)],
            dtype=np.float64,
        )
        median = float(np.median(volumes))
        weights = np.sqrt(median / np.maximum(volumes, 1.0))
        weights = np.clip(weights, 0.5, 4.0)
        return torch.as_tensor(weights, dtype=torch.double)


class ResidualRefiner(nn.Module):
    def __init__(self, init_filters: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = SegResNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=1,
            init_filters=init_filters,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=dropout,
        )
        final = self.network.conv_final[2].conv
        nn.init.zeros_(final.weight)
        if final.bias is not None:
            nn.init.zeros_(final.bias)

    def forward(self, image: torch.Tensor, coarse_logits: torch.Tensor) -> torch.Tensor:
        coarse_feature = torch.tanh(coarse_logits / 5.0)
        residual = self.network(torch.cat([image, coarse_feature], dim=1))
        return coarse_logits + residual


def shift_without_wrap(value: torch.Tensor, shifts: tuple[int, int, int]) -> torch.Tensor:
    result = torch.roll(value, shifts=shifts, dims=(-3, -2, -1))
    for axis, shift in enumerate(shifts, start=value.ndim - 3):
        if shift > 0:
            slices = [slice(None)] * value.ndim
            slices[axis] = slice(0, shift)
            result[tuple(slices)] = 0
        elif shift < 0:
            slices = [slice(None)] * value.ndim
            slices[axis] = slice(shift, None)
            result[tuple(slices)] = 0
    return result


def augment_batch(
    image: torch.Tensor,
    coarse: torch.Tensor,
    label: torch.Tensor,
    *,
    max_shift: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    for batch_index in range(image.shape[0]):
        for axis in (-3, -2, -1):
            if bool(torch.rand((), device=image.device) < 0.5):
                image[batch_index] = torch.flip(image[batch_index], dims=(axis,))
                coarse[batch_index] = torch.flip(coarse[batch_index], dims=(axis,))
                label[batch_index] = torch.flip(label[batch_index], dims=(axis,))
        shifts = tuple(
            int(torch.randint(-max_shift, max_shift + 1, (), device=image.device))
            for _ in range(3)
        )
        image[batch_index] = shift_without_wrap(image[batch_index], shifts)
        coarse[batch_index] = shift_without_wrap(coarse[batch_index], shifts)
        label[batch_index] = shift_without_wrap(label[batch_index], shifts)
    scale = torch.empty((image.shape[0], 1, 1, 1, 1), device=image.device).uniform_(0.9, 1.1)
    offset = torch.empty((image.shape[0], 1, 1, 1, 1), device=image.device).uniform_(-0.1, 0.1)
    noise_scale = torch.empty((image.shape[0], 1, 1, 1, 1), device=image.device).uniform_(0.0, 0.05)
    image = image * scale + offset + torch.randn_like(image) * noise_scale
    return image, coarse, label


def dice_per_case(logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    prediction = logits > 0
    target = label > 0.5
    intersection = (prediction & target).flatten(1).sum(dim=1).float()
    denominator = prediction.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
    return torch.where(denominator > 0, 2.0 * intersection / denominator.float(), torch.ones_like(intersection))


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    refined: list[float] = []
    coarse_values: list[float] = []
    case_ids: list[str] = []
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        coarse = batch["coarse"].to(device=device, dtype=torch.float32, non_blocking=True)
        label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(image, coarse)
        refined.extend(dice_per_case(logits.float(), label).cpu().tolist())
        coarse_values.extend(dice_per_case(coarse, label).cpu().tolist())
        case_ids.extend(list(batch["case_id"]))
    return {
        "cases": len(refined),
        "coarse_dsc_mean": float(np.mean(coarse_values)),
        "refined_dsc_mean": float(np.mean(refined)),
        "delta": float(np.mean(np.asarray(refined) - np.asarray(coarse_values))),
        "improved_cases": int(np.sum(np.asarray(refined) > np.asarray(coarse_values))),
        "worsened_cases": int(np.sum(np.asarray(refined) < np.asarray(coarse_values))),
        "per_case": [
            {"case_id": cid, "coarse_dsc": coarse, "refined_dsc": value}
            for cid, coarse, value in zip(case_ids, coarse_values, refined, strict=True)
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("refiner training requires CUDA")

    train_data = RefinerDataset(args.train_cache)
    validation_data = RefinerDataset(args.validation_cache)
    if train_data.index["split"] != "train" or validation_data.index["split"] != "validation":
        raise ValueError("train/validation refiner cache split mismatch")
    if train_data.index["model"]["checkpoint_sha256"] != validation_data.index["model"]["checkpoint_sha256"]:
        raise ValueError("train/validation coarse checkpoint mismatch")

    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        train_data.sampling_weights(),
        num_samples=len(train_data),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    model = ResidualRefiner(args.init_filters, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = DiceCELoss(sigmoid=True, squared_pred=False, lambda_dice=1.0, lambda_ce=1.0)
    scaler = None
    history: list[dict[str, Any]] = []
    best_value = float("-inf")
    best_epoch = 0
    stale = 0
    started = time.time()

    baseline = validate(model, validation_loader, device)
    if abs(baseline["delta"]) > 1e-7:
        raise RuntimeError("zero-initialized refiner does not reproduce coarse baseline")
    print(f"epoch=0 coarse={baseline['coarse_dsc_mean']:.6f} refined={baseline['refined_dsc_mean']:.6f}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        train_dice: list[float] = []
        for batch in train_loader:
            image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            coarse = batch["coarse"].to(device=device, dtype=torch.float32, non_blocking=True)
            label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            image, coarse, label = augment_batch(
                image, coarse, label, max_shift=args.max_shift
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(image, coarse)
                loss = loss_fn(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            train_dice.extend(dice_per_case(logits.detach().float(), label).cpu().tolist())

        validation = validate(model, validation_loader, device)
        value = float(validation["refined_dsc_mean"])
        strict_new_high = value > best_value
        if strict_new_high:
            best_value = value
            best_epoch = epoch
            stale = 0
            checkpoint = {
                "schema": "aflow.sammed3d.t1ce_wt_residual_refiner_checkpoint",
                "schema_version": 1,
                "epoch": epoch,
                "validation": validation,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "architecture": {
                    "class": "ResidualRefiner",
                    "backbone": "MONAI SegResNet",
                    "init_filters": args.init_filters,
                    "dropout": args.dropout,
                    "input_channels": ["T1ce", "tanh(stage2_click3_raw_logit/5)"],
                    "output": "stage2_click3_raw_logit + learned_residual",
                },
            }
            torch.save(checkpoint, output_dir / "refiner_val_dice_best.pth")
        else:
            stale += 1
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_dsc": float(np.mean(train_dice)),
            "validation": validation,
            "strict_new_high": strict_new_high,
            "stale_epochs": stale,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        write_json_atomic(output_dir / "progress.json", {"history": history, "best_epoch": best_epoch, "best_dsc": best_value})
        print(
            f"epoch={epoch} loss={record['train_loss']:.5f} train={record['train_dsc']:.5f} "
            f"val={value:.6f} delta={validation['delta']:+.6f} best={best_value:.6f}",
            flush=True,
        )
        if stale >= args.patience:
            break

    best_path = output_dir / "refiner_val_dice_best.pth"
    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "task": {
            "input": "T1ce single channel plus frozen SAM click-3 raw logits",
            "target": "WT={1,2,3}; raw label 4 excluded",
            "split": "972 train / 108 validation, subject-clean",
            "geometry": "stage-2 native 128^3 crop; validation remains development-only",
        },
        "baseline_validation": baseline,
        "selection": {"metric": "validation scan-macro DSC", "epoch": best_epoch, "value": best_value},
        "best_checkpoint": {"path": str(best_path), "sha256": sha256_file(best_path)},
        "coarse_checkpoint_sha256": train_data.index["model"]["checkpoint_sha256"],
        "train_cache": {"path": str(args.train_cache.resolve()), "index_sha256": sha256_file(args.train_cache / "index.json")},
        "validation_cache": {"path": str(args.validation_cache.resolve()), "index_sha256": sha256_file(args.validation_cache / "index.json")},
        "hyperparameters": vars(args) | {"train_cache": str(args.train_cache), "validation_cache": str(args.validation_cache), "output_dir": str(args.output_dir)},
        "history": history,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": time.time() - started,
    }
    write_json_atomic(output_dir / "TRAIN_DONE.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--init-filters", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-shift", type=int, default=12)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
