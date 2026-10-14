#!/usr/bin/env python3
"""Train a DDP prompt-centred local residual refiner on development data.

The frozen full-resolution refiner remains the global prediction.  A compact
network sees a fixed 64^3 cube around the first positive corrective click and
predicts a residual only inside that cube.  The residual output is zero at
initialization, so epoch zero exactly reproduces the frozen global refiner.
"""
from __future__ import annotations

import argparse
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
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monai.losses import DiceCELoss  # noqa: E402
from monai.networks.nets import SegResNet  # noqa: E402
from sammed3d.t1ce_wt_cached import MODEL_SHAPE  # noqa: E402


CACHE_SCHEMA = "aflow.sammed3d.t1ce_wt_prompt_local_refiner_merged_cache"
TRAIN_SCHEMA = "aflow.sammed3d.t1ce_wt_prompt_local_refiner_training"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_dist() else 0


def log(message: str) -> None:
    if rank() == 0:
        print(message, flush=True)


def gaussian_heatmap(
    point: torch.Tensor,
    label: int,
    patch_origin: tuple[int, int, int],
    patch_size: int,
    sigma: float,
) -> torch.Tensor:
    heatmap = torch.zeros((patch_size, patch_size, patch_size), dtype=torch.float32)
    if label not in {0, 1} or bool(torch.any(point < 0)):
        return heatmap
    local = point.to(torch.float32) - torch.tensor(patch_origin, dtype=torch.float32)
    if bool(torch.any(local < 0)) or bool(torch.any(local >= patch_size)):
        return heatmap
    axes = [torch.arange(patch_size, dtype=torch.float32) for _ in range(3)]
    z, y, x = torch.meshgrid(*axes, indexing="ij")
    distance2 = (z - local[0]) ** 2 + (y - local[1]) ** 2 + (x - local[2]) ** 2
    return torch.exp(-0.5 * distance2 / (sigma * sigma))


def patch_origin(first_positive: torch.Tensor, patch_size: int) -> tuple[int, int, int]:
    half = patch_size // 2
    maximum = MODEL_SHAPE[0] - patch_size
    return tuple(
        max(0, min(maximum, int(value) - half)) for value in first_positive.tolist()
    )


class PromptLocalDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        *,
        patch_size: int,
        click_sigma: float,
        augment: bool,
    ) -> None:
        self.cache_dir = cache_dir.resolve()
        self.index = json.loads((self.cache_dir / "index.json").read_text())
        if self.index.get("schema") != CACHE_SCHEMA:
            raise ValueError("unexpected prompt-local cache schema")
        if self.index.get("test_access") is not False:
            raise ValueError("prompt-local training cannot consume test data")
        if patch_size <= 0 or patch_size > MODEL_SHAPE[0] or patch_size % 16:
            raise ValueError("patch size must be a positive multiple of 16 no larger than 128")
        self.patch_size = patch_size
        self.click_sigma = click_sigma
        self.augment = augment
        self.count = int(self.index["count"])
        self.case_ids = list(self.index["case_ids"])
        source = Path(self.index["source_cache"]["path"])
        source_index = json.loads((source / "index.json").read_text())
        if source_index["images_sha256"] != self.index["source_cache"]["images_sha256"]:
            raise RuntimeError("source image cache hash mismatch")
        if source_index["labels_sha256"] != self.index["source_cache"]["labels_sha256"]:
            raise RuntimeError("source label cache hash mismatch")
        if list(source_index["case_ids"]) != self.case_ids:
            raise RuntimeError("source/prompt-local case ordering mismatch")
        self.source = source
        self._images: np.memmap | None = None
        self._labels: np.memmap | None = None
        self._global: np.memmap | None = None
        self._points: np.ndarray | None = None
        self._point_labels: np.ndarray | None = None
        self._outside_stats = np.full((self.count, 3), -1, dtype=np.int64)

    def _open(self) -> None:
        if self._images is not None:
            return
        shape = (self.count, *MODEL_SHAPE)
        self._images = np.memmap(self.source / "images.f32", dtype=np.float32, mode="r", shape=shape)
        self._labels = np.memmap(self.source / "labels.u8", dtype=np.uint8, mode="r", shape=shape)
        self._global = np.memmap(
            self.cache_dir / "global_refined_logits.f16", dtype=np.float16, mode="r", shape=shape
        )
        self._points = np.load(self.cache_dir / "click_points.i16.npy", mmap_mode="r", allow_pickle=False)
        self._point_labels = np.load(self.cache_dir / "click_labels.i8.npy", mmap_mode="r", allow_pickle=False)

    def __len__(self) -> int:
        return self.count

    def target_volumes(self) -> np.ndarray:
        self._open()
        assert self._labels is not None
        return np.asarray([np.count_nonzero(self._labels[i]) for i in range(self.count)], dtype=np.float64)

    def __getitem__(self, slot: int) -> dict[str, torch.Tensor | str]:
        self._open()
        assert all(value is not None for value in (
            self._images, self._labels, self._global, self._points, self._point_labels
        ))
        points = torch.from_numpy(np.array(self._points[slot], dtype=np.int16)).long()
        labels = torch.from_numpy(np.array(self._point_labels[slot], dtype=np.int8)).long()
        if int(labels[0]) != 1:
            raise RuntimeError("first click is not positive")
        origin = patch_origin(points[0], self.patch_size)
        slices = tuple(slice(start, start + self.patch_size) for start in origin)
        image = torch.from_numpy(np.array(self._images[(slot, *slices)], dtype=np.float32))[None]
        global_logits = torch.from_numpy(np.array(self._global[(slot, *slices)], dtype=np.float32))[None]
        target = torch.from_numpy(np.array(self._labels[(slot, *slices)], dtype=np.float32))[None]
        if self.augment:
            outside_intersection = outside_prediction = outside_target = 0
        else:
            if self._outside_stats[slot, 0] < 0:
                full_global_prediction = np.asarray(self._global[slot]) > 0
                full_target = np.asarray(self._labels[slot]) > 0
                patch_global_prediction = np.asarray(self._global[(slot, *slices)]) > 0
                patch_target = np.asarray(self._labels[(slot, *slices)]) > 0
                self._outside_stats[slot] = (
                    int(np.count_nonzero(full_global_prediction & full_target))
                    - int(np.count_nonzero(patch_global_prediction & patch_target)),
                    int(np.count_nonzero(full_global_prediction))
                    - int(np.count_nonzero(patch_global_prediction)),
                    int(np.count_nonzero(full_target)) - int(np.count_nonzero(patch_target)),
                )
            outside_intersection, outside_prediction, outside_target = (
                int(value) for value in self._outside_stats[slot]
            )
        positive = torch.zeros_like(target)
        negative = torch.zeros_like(target)
        for point, click_label in zip(points, labels, strict=True):
            heatmap = gaussian_heatmap(point, int(click_label), origin, self.patch_size, self.click_sigma)[None]
            if int(click_label) == 1:
                positive = torch.maximum(positive, heatmap)
            elif int(click_label) == 0:
                negative = torch.maximum(negative, heatmap)
        if self.augment:
            for axis in (-3, -2, -1):
                if bool(torch.rand(()) < 0.5):
                    image = torch.flip(image, dims=(axis,))
                    global_logits = torch.flip(global_logits, dims=(axis,))
                    target = torch.flip(target, dims=(axis,))
                    positive = torch.flip(positive, dims=(axis,))
                    negative = torch.flip(negative, dims=(axis,))
            scale = torch.empty(()).uniform_(0.9, 1.1)
            offset = torch.empty(()).uniform_(-0.1, 0.1)
            image = image * scale + offset + torch.randn_like(image) * torch.empty(()).uniform_(0.0, 0.05)
        inputs = torch.cat((image, torch.tanh(global_logits / 5.0), positive, negative), dim=0)
        return {
            "inputs": inputs,
            "global_logits": global_logits,
            "target": target,
            "case_id": self.case_ids[slot],
            "outside_intersection": torch.tensor(outside_intersection, dtype=torch.int64),
            "outside_prediction": torch.tensor(outside_prediction, dtype=torch.int64),
            "outside_target": torch.tensor(outside_target, dtype=torch.int64),
        }


class DistributedWeightedSampler(Sampler[int]):
    """Deterministic replacement sampling, sharded evenly across DDP ranks."""

    def __init__(self, weights: torch.Tensor, replicas: int, rank_index: int, seed: int) -> None:
        self.weights = weights.to(dtype=torch.double, device="cpu")
        self.replicas = replicas
        self.rank_index = rank_index
        self.seed = seed
        self.epoch = 0
        self.samples_per_rank = len(weights) // replicas
        if self.samples_per_rank <= 0:
            raise ValueError("not enough samples for distributed weighted sampling")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        total = self.samples_per_rank * self.replicas
        selected = torch.multinomial(self.weights, total, replacement=True, generator=generator)
        return iter(selected[self.rank_index:total:self.replicas].tolist())

    def __len__(self) -> int:
        return self.samples_per_rank


class PromptLocalResidualRefiner(nn.Module):
    def __init__(self, init_filters: int, dropout: float) -> None:
        super().__init__()
        self.network = SegResNet(
            spatial_dims=3,
            in_channels=4,
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

    def forward(self, inputs: torch.Tensor, global_logits: torch.Tensor) -> torch.Tensor:
        return global_logits + self.network(inputs)


def dice_per_case(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = logits > 0
    truth = target > 0.5
    intersection = (prediction & truth).flatten(1).sum(dim=1).float()
    denominator = prediction.flatten(1).sum(dim=1) + truth.flatten(1).sum(dim=1)
    return torch.where(denominator > 0, 2 * intersection / denominator.float(), torch.ones_like(intersection))


def combined_patch_dice(
    patch_intersection: np.ndarray,
    patch_prediction: np.ndarray,
    patch_target: np.ndarray,
    outside_intersection: np.ndarray,
    outside_prediction: np.ndarray,
    outside_target: np.ndarray,
) -> np.ndarray:
    denominator = patch_prediction + outside_prediction + patch_target + outside_target
    return np.where(
        denominator > 0,
        2 * (patch_intersection + outside_intersection) / denominator,
        1.0,
    )


@torch.inference_mode()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    local_values: list[float] = []
    global_values: list[float] = []
    case_ids: list[str] = []
    for batch in loader:
        inputs = batch["inputs"].to(device=device, dtype=torch.float32, non_blocking=True)
        global_logits = batch["global_logits"].to(device=device, dtype=torch.float32, non_blocking=True)
        target = batch["target"].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            local_logits = model(inputs, global_logits)
        local_prediction = local_logits.float() > 0
        global_prediction = global_logits > 0
        truth = target > 0.5
        local_intersection = (local_prediction & truth).flatten(1).sum(dim=1).cpu().numpy()
        global_intersection = (global_prediction & truth).flatten(1).sum(dim=1).cpu().numpy()
        local_prediction_count = local_prediction.flatten(1).sum(dim=1).cpu().numpy()
        global_prediction_count = global_prediction.flatten(1).sum(dim=1).cpu().numpy()
        target_count = truth.flatten(1).sum(dim=1).cpu().numpy()
        outside_intersection = batch["outside_intersection"].numpy()
        outside_prediction = batch["outside_prediction"].numpy()
        outside_target = batch["outside_target"].numpy()
        local_values.extend(
            combined_patch_dice(
                local_intersection, local_prediction_count, target_count,
                outside_intersection, outside_prediction, outside_target,
            ).tolist()
        )
        global_values.extend(
            combined_patch_dice(
                global_intersection, global_prediction_count, target_count,
                outside_intersection, outside_prediction, outside_target,
            ).tolist()
        )
        case_ids.extend(list(batch["case_id"]))
    return {
        "cases": len(local_values),
        "global_scan_macro_dsc": float(np.mean(global_values)),
        "local_scan_macro_dsc": float(np.mean(local_values)),
        "delta": float(np.mean(np.asarray(local_values) - np.asarray(global_values))),
        "per_case": [
            {"case_id": case_id, "global_dsc": coarse, "local_refined_dsc": refined}
            for case_id, coarse, refined in zip(case_ids, global_values, local_values, strict=True)
        ],
    }


def setup_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 0, 1
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    local_rank, global_rank, size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    output_dir = args.output_dir.resolve()
    if global_rank == 0:
        output_dir.mkdir(parents=True, exist_ok=False)
    if is_dist():
        dist.barrier()
    random.seed(args.seed + global_rank)
    np.random.seed(args.seed + global_rank)
    torch.manual_seed(args.seed + global_rank)
    torch.cuda.manual_seed_all(args.seed + global_rank)
    torch.backends.cudnn.benchmark = False

    train_data = PromptLocalDataset(
        args.train_cache, patch_size=args.patch_size, click_sigma=args.click_sigma, augment=True
    )
    validation_data = PromptLocalDataset(
        args.validation_cache, patch_size=args.patch_size, click_sigma=args.click_sigma, augment=False
    )
    if train_data.index["split"] != "train" or validation_data.index["split"] != "validation":
        raise ValueError("train/validation split mismatch")
    if train_data.index["global_refiner"]["checkpoint_sha256"] != validation_data.index["global_refiner"]["checkpoint_sha256"]:
        raise ValueError("global refiner mismatch")

    volumes = train_data.target_volumes()
    weights = np.clip(np.sqrt(np.median(volumes) / np.maximum(volumes, 1)), 0.5, 4.0)
    train_sampler = DistributedWeightedSampler(
        torch.as_tensor(weights, dtype=torch.double), size, global_rank, args.seed
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=False, drop_last=True,
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.validation_batch_size, shuffle=False,
        num_workers=max(1, args.num_workers // 2), pin_memory=True, persistent_workers=False,
    )
    model: nn.Module = PromptLocalResidualRefiner(args.init_filters, args.dropout).to(device)
    if size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = DiceCELoss(sigmoid=True, squared_pred=False, lambda_dice=1.0, lambda_ce=1.0)
    history: list[dict[str, Any]] = []
    best_value = float("-inf")
    best_epoch = 0
    stale = 0
    started = time.time()

    if global_rank == 0:
        validation_model = model.module if isinstance(model, DistributedDataParallel) else model
        baseline = validate(validation_model, validation_loader, device)
        if abs(baseline["delta"]) > 1e-7:
            raise RuntimeError("zero-initialized local refiner does not reproduce global baseline")
        log(
            f"epoch=0 global={baseline['global_scan_macro_dsc']:.6f} "
            f"local={baseline['local_scan_macro_dsc']:.6f}"
        )
    else:
        baseline = None
    if is_dist():
        dist.barrier()

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_losses: list[float] = []
        epoch_dice: list[float] = []
        for batch in train_loader:
            inputs = batch["inputs"].to(device=device, dtype=torch.float32, non_blocking=True)
            global_logits = batch["global_logits"].to(device=device, dtype=torch.float32, non_blocking=True)
            target = batch["target"].to(device=device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                local_logits = model(inputs, global_logits)
                loss = loss_fn(local_logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
            epoch_dice.extend(dice_per_case(local_logits.detach().float(), target).cpu().tolist())
        if is_dist():
            dist.barrier()
        if global_rank == 0:
            validation_model = model.module if isinstance(model, DistributedDataParallel) else model
            validation = validate(validation_model, validation_loader, device)
            value = float(validation["local_scan_macro_dsc"])
            improved = value > best_value
            if improved:
                best_value = value
                best_epoch = epoch
                stale = 0
                module = model.module if isinstance(model, DistributedDataParallel) else model
                torch.save(
                    {
                        "schema": "aflow.sammed3d.t1ce_wt_prompt_local_refiner_checkpoint",
                        "schema_version": 1,
                        "epoch": epoch,
                        "validation": validation,
                        "model_state_dict": module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "architecture": {
                            "class": "PromptLocalResidualRefiner",
                            "backbone": "MONAI SegResNet",
                            "input_channels": ["T1ce", "tanh(global_refined_logit/5)", "positive_click_heatmap", "negative_click_heatmap"],
                            "patch_size": args.patch_size,
                            "click_sigma": args.click_sigma,
                            "init_filters": args.init_filters,
                            "dropout": args.dropout,
                        },
                    },
                    output_dir / "prompt_local_refiner_val_dice_best.pth",
                )
            else:
                stale += 1
            record = {
                "epoch": epoch,
                "rank0_train_loss": float(np.mean(epoch_losses)),
                "rank0_train_dsc": float(np.mean(epoch_dice)),
                "validation": validation,
                "strict_new_high": improved,
                "stale_epochs": stale,
                "elapsed_seconds": time.time() - started,
            }
            history.append(record)
            write_json_atomic(output_dir / "progress.json", {"history": history, "best_epoch": best_epoch, "best_scan_macro_dsc": best_value})
            log(
                f"epoch={epoch} loss={record['rank0_train_loss']:.5f} "
                f"train={record['rank0_train_dsc']:.5f} val={value:.6f} "
                f"delta={validation['delta']:+.6f} best={best_value:.6f}"
            )
            stop = stale >= args.patience
        else:
            stop = False
        if is_dist():
            stop_tensor = torch.tensor([int(stop)], device=device)
            dist.broadcast(stop_tensor, src=0)
            stop = bool(stop_tensor.item())
        if stop:
            break

    report: dict[str, Any] | None = None
    if global_rank == 0:
        assert baseline is not None
        best_path = output_dir / "prompt_local_refiner_val_dice_best.pth"
        report = {
            "schema": TRAIN_SCHEMA,
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "test_access": False,
            "task": {
                "input": "T1ce + frozen global-refiner logit + frozen corrective-click heatmaps",
                "target": "WT={1,2,3}; raw label 4 excluded",
                "split": "972 train / 108 validation, subject-clean",
                "geometry": f"fixed {args.patch_size}^3 cube centred on frozen first positive click",
            },
            "world_size": size,
            "baseline_validation": baseline,
            "selection": {"metric": "validation full-scan macro DSC", "epoch": best_epoch, "value": best_value},
            "best_checkpoint": {"path": str(best_path), "sha256": sha256_file(best_path)},
            "train_cache": {"path": str(args.train_cache.resolve()), "index_sha256": sha256_file(args.train_cache / "index.json")},
            "validation_cache": {"path": str(args.validation_cache.resolve()), "index_sha256": sha256_file(args.validation_cache / "index.json")},
            "hyperparameters": vars(args) | {
                "train_cache": str(args.train_cache), "validation_cache": str(args.validation_cache), "output_dir": str(args.output_dir)
            },
            "history": history,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "elapsed_seconds": time.time() - started,
        }
        write_json_atomic(output_dir / "TRAIN_DONE.json", report)
    if is_dist():
        # This worker image's NCCL/IB plugin hangs indefinitely in commFree even
        # after both ranks finish.  Synchronize after rank 0 has durably written
        # every artifact, then let the OS reclaim communicator resources with
        # the process.  The preregistered worker-exit amendment records this
        # lifecycle-only workaround.
        dist.barrier()
        os._exit(0)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--click-sigma", type=float, default=3.0)
    parser.add_argument("--init-filters", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
