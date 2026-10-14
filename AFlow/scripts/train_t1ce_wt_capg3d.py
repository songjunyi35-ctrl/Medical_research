#!/usr/bin/env python3
"""Train matched image-only plain/CAPG-3D automatic prompt locators.

This entry point is deliberately validation-only.  It accepts only caches that
declare ``test_access: false`` and has no test mode or test manifest argument.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any

import numpy as np
import torch
from monai.losses import DiceCELoss
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_cached import CACHE_SCHEMA, MODEL_SHAPE  # noqa: E402
from sammed3d.t1ce_wt_capg3d import (  # noqa: E402
    AutomaticPromptLocator,
    automatic_prompt_points,
    center_heatmap_loss,
    gaussian_center_target,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_capg3d_training"
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_capg3d_checkpoint"


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


def subject_id(case_id: str) -> str:
    pieces = case_id.rsplit("-", 1)
    if len(pieces) != 2 or not pieces[1].isdigit():
        raise ValueError(f"cannot derive subject id from {case_id!r}")
    return pieces[0]


def read_case_ids(path: Path | None) -> list[str] | None:
    """Read an ordered JSONL case view without opening any image or label."""

    if path is None:
        return None
    case_ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("case_id"):
                raise ValueError(f"{path}:{line_number} lacks case_id")
            case_ids.append(str(row["case_id"]))
    if not case_ids:
        raise ValueError(f"case view is empty: {path}")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"case view contains duplicate case_id values: {path}")
    return case_ids


def resolve_case_slots(
    available_case_ids: list[str], requested_case_ids: list[str] | None
) -> tuple[list[int], list[str]]:
    """Resolve a frozen case view to source-cache slots, failing on leakage errors."""

    if len(available_case_ids) != len(set(available_case_ids)):
        raise ValueError("source cache contains duplicate case_id values")
    if requested_case_ids is None:
        return list(range(len(available_case_ids))), list(available_case_ids)
    if len(requested_case_ids) != len(set(requested_case_ids)):
        raise ValueError("requested case view contains duplicate case_id values")
    source_slots = {case_id: slot for slot, case_id in enumerate(available_case_ids)}
    missing = [case_id for case_id in requested_case_ids if case_id not in source_slots]
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(f"requested cases are absent from source cache: {preview}")
    return [source_slots[case_id] for case_id in requested_case_ids], list(requested_case_ids)


class BrainCropDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        *,
        expected_split: str,
        limit: int | None = None,
        case_ids: list[str] | None = None,
    ) -> None:
        self.cache_dir = cache_dir.resolve()
        self.index = json.loads((self.cache_dir / "index.json").read_text())
        if self.index.get("schema") != CACHE_SCHEMA:
            raise ValueError("unexpected input cache schema")
        if self.index.get("test_access") is not False:
            raise ValueError("CAPG-3D cannot consume a test-accessible cache")
        if self.index.get("split") != expected_split:
            raise ValueError(
                f"cache split {self.index.get('split')!r} != {expected_split!r}"
            )
        preprocessing = self.index.get("preprocessing", {})
        if preprocessing.get("spatial") != "braincrop_resize":
            raise ValueError("CAPG-3D requires image-only braincrop_resize geometry")
        if preprocessing.get("uses_gt_for_localization") is not False:
            raise ValueError("input cache must not use GT for localization")
        total = int(self.index["count"])
        available_case_ids = [str(value) for value in self.index["case_ids"]]
        if len(available_case_ids) != total:
            raise ValueError("source cache count and case_ids length differ")
        slots, selected_case_ids = resolve_case_slots(available_case_ids, case_ids)
        self.count = min(len(slots), int(limit)) if limit is not None else len(slots)
        self._slots = slots[: self.count]
        self.case_ids = selected_case_ids[: self.count]
        self._images: np.memmap | None = None
        self._labels: np.memmap | None = None
        self._total = total

    def _open(self) -> None:
        if self._images is None:
            self._images = np.memmap(
                self.cache_dir / "images.f32",
                dtype=np.float32,
                mode="r",
                shape=(self._total, *MODEL_SHAPE),
            )
            self._labels = np.memmap(
                self.cache_dir / "labels.u8",
                dtype=np.uint8,
                mode="r",
                shape=(self._total, *MODEL_SHAPE),
            )

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, slot: int) -> dict[str, torch.Tensor | str]:
        self._open()
        assert self._images is not None and self._labels is not None
        source_slot = self._slots[slot]
        image = torch.from_numpy(np.array(self._images[source_slot], dtype=np.float32))[None]
        label = torch.from_numpy(np.array(self._labels[source_slot], dtype=np.float32))[None]
        return {
            "image": image,
            "label": label,
            "case_id": self.case_ids[slot],
            "subject_id": subject_id(self.case_ids[slot]),
        }

    def sampling_weights(self) -> torch.Tensor:
        self._open()
        assert self._labels is not None
        volumes = np.asarray(
            [float(np.count_nonzero(self._labels[i])) for i in self._slots],
            dtype=np.float64,
        )
        median = float(np.median(volumes))
        weights = np.sqrt(median / np.maximum(volumes, 1.0))
        return torch.as_tensor(np.clip(weights, 0.5, 4.0), dtype=torch.double)


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
    image: torch.Tensor, label: torch.Tensor, *, max_shift: int
) -> tuple[torch.Tensor, torch.Tensor]:
    for batch_index in range(image.shape[0]):
        for axis in (-3, -2, -1):
            if bool(torch.rand((), device=image.device) < 0.5):
                image[batch_index] = torch.flip(image[batch_index], dims=(axis,))
                label[batch_index] = torch.flip(label[batch_index], dims=(axis,))
        shifts = tuple(
            int(torch.randint(-max_shift, max_shift + 1, (), device=image.device))
            for _ in range(3)
        )
        image[batch_index] = shift_without_wrap(image[batch_index], shifts)
        label[batch_index] = shift_without_wrap(label[batch_index], shifts)
    scale = torch.empty((image.shape[0], 1, 1, 1, 1), device=image.device).uniform_(0.9, 1.1)
    offset = torch.empty((image.shape[0], 1, 1, 1, 1), device=image.device).uniform_(-0.1, 0.1)
    noise = torch.empty((image.shape[0], 1, 1, 1, 1), device=image.device).uniform_(0.0, 0.05)
    return image * scale + offset + torch.randn_like(image) * noise, label


def dice_per_case(logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    prediction = logits > 0
    target = label > 0.5
    intersection = (prediction & target).flatten(1).sum(dim=1).float()
    denominator = prediction.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
    return torch.where(
        denominator > 0,
        2.0 * intersection / denominator.float(),
        torch.ones_like(intersection),
    )


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(case["locator_dsc"]) for case in cases]
    by_subject: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        by_subject[str(case["subject_id"])].append(float(case["locator_dsc"]))
    subject_values = [statistics.fmean(items) for items in by_subject.values()]
    return {
        "cases": len(cases),
        "subjects": len(subject_values),
        "locator_dsc_mean": statistics.fmean(values),
        "locator_dsc_median": statistics.median(values),
        "locator_dsc_std_population": statistics.pstdev(values),
        "locator_subject_dsc_mean": statistics.fmean(subject_values),
        "automatic_point_hit_rate": statistics.fmean(
            float(case["automatic_point_hits_gt"]) for case in cases
        ),
        "per_case": cases,
    }


@torch.inference_mode()
def validate(
    model: AutomaticPromptLocator,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    cases: list[dict[str, Any]] = []
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(image)
        dsc = dice_per_case(output.segmentation_logits.float(), label)
        points = automatic_prompt_points(output).long()
        for index, case_id in enumerate(batch["case_id"]):
            z, y, x = (int(value) for value in points[index, 0].tolist())
            cases.append(
                {
                    "case_id": str(case_id),
                    "subject_id": str(batch["subject_id"][index]),
                    "locator_dsc": float(dsc[index]),
                    "automatic_point": [z, y, x],
                    "automatic_point_hits_gt": bool(label[index, 0, z, y, x] > 0.5),
                }
            )
    return summarize_cases(cases)


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def case_view_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    if args.arm not in {"plain", "capg"}:
        raise ValueError("arm must be plain or capg")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CAPG-3D training requires CUDA")
    torch.cuda.set_device(device)

    train_data = BrainCropDataset(
        args.train_cache,
        expected_split=args.train_cache_expected_split,
        limit=args.limit_train_cases,
        case_ids=read_case_ids(args.train_case_ids),
    )
    validation_data = BrainCropDataset(
        args.validation_cache,
        expected_split=args.validation_cache_expected_split,
        limit=args.limit_validation_cases,
        case_ids=read_case_ids(args.validation_case_ids),
    )
    if set(train_data.case_ids) & set(validation_data.case_ids):
        raise ValueError("train and validation case views overlap")
    train_subjects = {subject_id(case_id) for case_id in train_data.case_ids}
    validation_subjects = {subject_id(case_id) for case_id in validation_data.case_ids}
    if train_subjects & validation_subjects:
        raise ValueError("train and validation case views have subject leakage")
    data_roles = {
        "train_case_ids": case_view_metadata(args.train_case_ids),
        "validation_case_ids": case_view_metadata(args.validation_case_ids),
        "case_overlap": 0,
        "subject_overlap": 0,
    }
    effective_train_batch_size = min(args.batch_size, len(train_data))
    effective_validation_batch_size = min(
        args.validation_batch_size, len(validation_data)
    )
    if effective_train_batch_size <= 0 or effective_validation_batch_size <= 0:
        raise ValueError("train and validation datasets must be non-empty")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=effective_train_batch_size,
        sampler=WeightedRandomSampler(
            train_data.sampling_weights(),
            num_samples=len(train_data),
            replacement=True,
            generator=generator,
        ),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=effective_validation_batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    if len(train_loader) == 0:
        raise RuntimeError("training loader has zero optimizer steps")
    center_aware = args.arm == "capg"
    model = AutomaticPromptLocator(
        center_aware=center_aware,
        init_filters=args.init_filters,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    segmentation_loss = DiceCELoss(
        sigmoid=True,
        squared_pred=False,
        lambda_dice=1.0,
        lambda_ce=args.ce_weight,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    history: list[dict[str, Any]] = []
    best_value = float("-inf")
    best_epoch = 0
    stale = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        segmentation_losses: list[float] = []
        center_losses: list[float] = []
        train_dsc: list[float] = []
        for batch in train_loader:
            image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            image, label = augment_batch(image, label, max_shift=args.max_shift)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(image)
                seg_loss = segmentation_loss(output.segmentation_logits, label)
                if output.center_logits is None:
                    aux_loss = torch.zeros((), device=device)
                else:
                    center_target, _ = gaussian_center_target(label, sigma=args.center_sigma)
                    aux_loss = center_heatmap_loss(output.center_logits, center_target)
                loss = seg_loss + args.center_weight * aux_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            segmentation_losses.append(float(seg_loss.detach()))
            center_losses.append(float(aux_loss.detach()))
            train_dsc.extend(
                dice_per_case(output.segmentation_logits.detach().float(), label).cpu().tolist()
            )

        validation = validate(model, validation_loader, device)
        value = float(validation["locator_dsc_mean"])
        strict_new_high = value > best_value
        if strict_new_high:
            best_value = value
            best_epoch = epoch
            stale = 0
            checkpoint = {
                "schema": CHECKPOINT_SCHEMA,
                "schema_version": 1,
                "epoch": epoch,
                "arm": args.arm,
                "validation": validation,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "architecture": {
                    "class": "AutomaticPromptLocator",
                    "backbone": "MONAI SegResNet",
                    "center_aware": center_aware,
                    "init_filters": args.init_filters,
                    "dropout": args.dropout,
                    "input_channels": ["T1ce"],
                    "outputs": ["WT segmentation logit"]
                    + (["positive-click center heatmap"] if center_aware else []),
                },
                "hyperparameters": serializable_args(args),
                "data_roles": data_roles,
                "test_access": False,
            }
            torch.save(checkpoint, output_dir / "locator_val_dice_best.pth")
        else:
            stale += 1
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_segmentation_loss": float(np.mean(segmentation_losses)),
            "train_center_loss": float(np.mean(center_losses)),
            "train_dsc": float(np.mean(train_dsc)),
            "validation": validation,
            "strict_new_high": strict_new_high,
            "stale_epochs": stale,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        write_json_atomic(
            output_dir / "progress.json",
            {"history": history, "best_epoch": best_epoch, "best_dsc": best_value},
        )
        print(
            f"arm={args.arm} epoch={epoch} loss={record['train_loss']:.5f} "
            f"train={record['train_dsc']:.5f} val={value:.6f} "
            f"point_hit={validation['automatic_point_hit_rate']:.4f} "
            f"best={best_value:.6f}@{best_epoch}",
            flush=True,
        )
        if epoch >= args.min_epochs and stale >= args.patience:
            break

    best_path = output_dir / "locator_val_dice_best.pth"
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "arm": args.arm,
        "task": {
            "input": "T1ce only",
            "target": "WT={1,2,3}; raw label 4 excluded",
            "split": f"{len(train_data)} train / {len(validation_data)} validation, subject-clean",
            "geometry": "image-nonzero brain crop resized to 128^3; no GT localization",
            "inference_prompt": (
                "argmax learned center heatmap" if center_aware else "argmax WT segmentation logit"
            ),
        },
        "selection": {
            "metric": "validation scan-macro locator DSC at fixed logit threshold 0",
            "epoch": best_epoch,
            "value": best_value,
        },
        "best_checkpoint": {"path": str(best_path), "sha256": sha256_file(best_path)},
        "train_cache": {
            "path": str(args.train_cache.resolve()),
            "index_sha256": sha256_file(args.train_cache / "index.json"),
            "images_sha256": train_data.index["images_sha256"],
            "labels_sha256": train_data.index["labels_sha256"],
        },
        "validation_cache": {
            "path": str(args.validation_cache.resolve()),
            "index_sha256": sha256_file(args.validation_cache / "index.json"),
            "images_sha256": validation_data.index["images_sha256"],
            "labels_sha256": validation_data.index["labels_sha256"],
        },
        "parameter_count": parameter_count,
        "effective_train_batch_size": effective_train_batch_size,
        "effective_validation_batch_size": effective_validation_batch_size,
        "hyperparameters": serializable_args(args),
        "data_roles": data_roles,
        "history": history,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
    }
    write_json_atomic(output_dir / "TRAIN_DONE.json", report)
    print(
        f"TRAIN_DONE arm={args.arm} best={best_value:.9f} epoch={best_epoch} "
        f"checkpoint={best_path}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--train-case-ids", type=Path, default=None)
    parser.add_argument("--validation-case-ids", type=Path, default=None)
    parser.add_argument("--train-cache-expected-split", default="train")
    parser.add_argument("--validation-cache-expected-split", default="validation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("plain", "capg"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--init-filters", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-shift", type=int, default=8)
    parser.add_argument("--center-weight", type=float, default=0.1)
    parser.add_argument("--center-sigma", type=float, default=6.0)
    parser.add_argument("--ce-weight", type=float, default=0.1)
    parser.add_argument("--limit-train-cases", type=int, default=None)
    parser.add_argument("--limit-validation-cases", type=int, default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
