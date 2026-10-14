#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sammed3d import engine, peft  # noqa: E402


def resolve_amp_dtype(args: argparse.Namespace, device: torch.device) -> torch.dtype | None:
    requested = getattr(args, "amp_dtype", "fp16")
    if getattr(args, "no_amp", False) or requested == "none" or device.type != "cuda":
        return None
    if requested == "fp16":
        return torch.float16
    if requested == "bf16":
        return torch.bfloat16
    raise ValueError(f"unknown AMP dtype: {requested}")


def amp_dtype_name(dtype: torch.dtype | None) -> str:
    if dtype is None:
        return "none"
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    raise ValueError(f"unsupported AMP dtype: {dtype}")


def make_grad_scaler(
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> torch.amp.GradScaler | None:
    if device.type == "cuda" and amp_dtype == torch.float16:
        return torch.amp.GradScaler("cuda")
    return None


def resolve_val_seed(args: argparse.Namespace) -> int:
    val_seed = getattr(args, "val_seed", None)
    return int(args.seed if val_seed is None else val_seed)


@contextmanager
def fixed_rng_seed(seed: int, device: torch.device):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def prepare_output_dir(path: Path, allow_nonempty: bool) -> None:
    if path.exists():
        if not path.is_dir():
            raise RuntimeError(f"output path is not a directory: {path}")
        if any(path.iterdir()) and not allow_nonempty:
            raise RuntimeError(
                f"output directory is not empty: {path}; choose a new directory or "
                "pass --allow-nonempty-output"
            )
        return
    path.mkdir(parents=True, exist_ok=False)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _row_sort_key(row: dict) -> tuple[str, str, str, str]:
    return (
        str(row.get("target", "")),
        str(row["case_id"]),
        str(row.get("image_path", "")),
        str(row.get("label_path", "")),
    )


def _stable_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def target_balanced_subset(
    rows: list[dict],
    max_rows: int | None,
    rows_per_target: int | None,
    seed: int,
) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("target", ""))].append(row)
    if len(grouped) <= 1:
        ordered = sorted(rows, key=_row_sort_key)
        if rows_per_target:
            ordered = ordered[:rows_per_target]
        return ordered[:max_rows] if max_rows else ordered

    candidates: dict[str, list[dict]] = {}
    for target, target_rows in grouped.items():
        target_rows = sorted(target_rows, key=_row_sort_key)
        rng = random.Random(_stable_seed(seed, target))
        rng.shuffle(target_rows)
        if rows_per_target:
            target_rows = target_rows[:rows_per_target]
        candidates[target] = target_rows

    target_order = sorted(candidates)
    random.Random(_stable_seed(seed, "target-order")).shuffle(target_order)
    limit = min(max_rows, sum(map(len, candidates.values()))) if max_rows else None
    selected = []
    offsets = {target: 0 for target in target_order}
    while limit is None or len(selected) < limit:
        added = False
        for target in target_order:
            offset = offsets[target]
            if offset >= len(candidates[target]):
                continue
            selected.append(candidates[target][offset])
            offsets[target] += 1
            added = True
            if limit is not None and len(selected) >= limit:
                break
        if not added:
            break
    return sorted(selected, key=_row_sort_key)


def filter_rows(
    rows: list[dict],
    targets: set[str] | None,
    max_cases: int | None,
    *,
    balance_targets: bool = False,
    rows_per_target: int | None = None,
    seed: int = 2026,
) -> list[dict]:
    if targets:
        rows = [row for row in rows if row.get("target") in targets]
    rows = sorted(rows, key=_row_sort_key)
    if balance_targets and len({row.get("target", "") for row in rows}) > 1:
        return target_balanced_subset(
            rows,
            max_rows=max_cases,
            rows_per_target=rows_per_target,
            seed=seed,
        )
    if rows_per_target:
        counts: dict[str, int] = defaultdict(int)
        limited = []
        for row in rows:
            target = str(row.get("target", ""))
            if counts[target] >= rows_per_target:
                continue
            limited.append(row)
            counts[target] += 1
        rows = limited
    if max_cases:
        rows = rows[:max_cases]
    return rows


def validate_target_patient_rows(rows: list[dict]) -> list[tuple[str, str]]:
    target_patients = []
    case_patients: dict[str, str] = {}
    seen_target_cases: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        values = {}
        for field in ("target", "patient_id", "case_id"):
            value = row.get(field)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(
                    f"row {index} requires a non-empty, trimmed string {field}"
                )
            values[field] = value

        target = values["target"]
        patient_id = values["patient_id"]
        case_id = values["case_id"]
        previous_patient = case_patients.setdefault(case_id, patient_id)
        if previous_patient != patient_id:
            raise ValueError(
                f"case_id {case_id!r} maps to conflicting patient_id values: "
                f"{previous_patient!r} and {patient_id!r}"
            )
        target_case = (target, case_id)
        if target_case in seen_target_cases:
            raise ValueError(f"duplicate (target, case_id): {target_case!r}")
        seen_target_cases.add(target_case)
        target_patients.append((target, patient_id))
    return target_patients


def target_sampling_weights(rows: list[dict], mode: str) -> torch.Tensor:
    if not rows:
        raise ValueError("cannot build sampling weights for an empty dataset")
    counts = Counter(str(row.get("target", "")) for row in rows)
    if mode == "natural":
        return torch.ones(len(rows), dtype=torch.double)
    if mode == "balanced":
        return torch.tensor(
            [1.0 / counts[str(row.get("target", ""))] for row in rows],
            dtype=torch.double,
        )
    if mode == "target_patient_balanced":
        target_patients = validate_target_patient_rows(rows)
        cell_counts = Counter(target_patients)
        patients_by_target: dict[str, set[str]] = defaultdict(set)
        for target, patient_id in target_patients:
            patients_by_target[target].add(patient_id)
        return torch.tensor(
            [
                1.0
                / (
                    len(patients_by_target[target])
                    * cell_counts[(target, patient_id)]
                )
                for target, patient_id in target_patients
            ],
            dtype=torch.double,
        )
    raise ValueError(f"unknown target sampling mode: {mode}")


def resolve_exact_samples_per_epoch(
    row_count: int,
    target_count: int,
    requested_samples: int | None,
) -> int:
    if row_count <= 0 or target_count <= 0:
        raise ValueError("exact sampling requires non-empty rows and targets")
    if requested_samples is None:
        resolved = (row_count // target_count) * target_count
    else:
        if requested_samples <= 0:
            raise ValueError("explicit exact samples per epoch must be > 0")
        if requested_samples % target_count:
            raise ValueError(
                "--train-samples-per-epoch must be divisible by the number of "
                f"targets ({target_count}) for target_patient_balanced sampling"
            )
        resolved = requested_samples
    if resolved < target_count:
        raise ValueError(
            "exact sampling needs at least one sample per target; "
            f"resolved {resolved} samples for {target_count} targets"
        )
    return resolved


class ExactTargetPatientSampler(Sampler[int]):
    SCHEMA_VERSION = 1
    ALGORITHM = "exact_target_patient_round_robin_v1"

    def __init__(
        self,
        rows: list[dict],
        seed: int,
        samples_per_epoch: int | None = None,
    ):
        target_patients = validate_target_patient_rows(rows)
        self.rows = rows
        self.seed = int(seed)
        grouped: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for index, (target, patient_id) in enumerate(target_patients):
            grouped[target][patient_id].append(index)

        self.targets = tuple(sorted(grouped))
        self.num_samples = resolve_exact_samples_per_epoch(
            len(rows),
            len(self.targets),
            samples_per_epoch,
        )
        self.samples_per_target = self.num_samples // len(self.targets)
        self.epoch = 0
        self._patient_orders: dict[str, tuple[str, ...]] = {}
        self._timepoint_orders: dict[tuple[str, str], tuple[int, ...]] = {}
        self._schedule_cache: dict[int, tuple[int, ...]] = {}
        self._metadata_cache: dict[int, dict] = {}

        for target in self.targets:
            patient_order = sorted(grouped[target])
            random.Random(
                _stable_seed(self.seed, f"{self.ALGORITHM}:patients:{target}")
            ).shuffle(patient_order)
            self._patient_orders[target] = tuple(patient_order)
            for patient_id, row_indices in grouped[target].items():
                timepoint_order = sorted(
                    row_indices,
                    key=lambda index: (_row_sort_key(rows[index]), index),
                )
                random.Random(
                    _stable_seed(
                        self.seed,
                        f"{self.ALGORITHM}:timepoints:{target}:{patient_id}",
                    )
                ).shuffle(timepoint_order)
                self._timepoint_orders[(target, patient_id)] = tuple(timepoint_order)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be >= 0")
        self.epoch = int(epoch)

    def _build_epoch(self, epoch: int) -> tuple[tuple[int, ...], dict]:
        if epoch < 0:
            raise ValueError("epoch must be >= 0")
        cached = self._schedule_cache.get(epoch)
        if cached is not None:
            return cached, self._metadata_cache[epoch]

        schedule = []
        patient_exposures: dict[str, Counter] = {}
        for target in self.targets:
            patient_order = self._patient_orders[target]
            patient_count = len(patient_order)
            exposures = Counter({patient_id: 0 for patient_id in patient_order})
            patient_exposures[target] = exposures
            for offset in range(self.samples_per_target):
                global_position = epoch * self.samples_per_target + offset
                patient_id = patient_order[global_position % patient_count]
                patient_occurrence = global_position // patient_count
                timepoints = self._timepoint_orders[(target, patient_id)]
                schedule.append(timepoints[patient_occurrence % len(timepoints)])
                exposures[patient_id] += 1

        random.Random(
            _stable_seed(self.seed, f"{self.ALGORITHM}:schedule:{epoch}")
        ).shuffle(schedule)
        target_quota = Counter(str(self.rows[index]["target"]) for index in schedule)
        expected_quota = {target: self.samples_per_target for target in self.targets}
        if dict(target_quota) != expected_quota:
            raise RuntimeError(
                f"exact sampler target quota mismatch: {target_quota} != {expected_quota}"
            )

        digest = hashlib.sha256()
        for index in schedule:
            row = self.rows[index]
            identity = [
                row["target"],
                row["patient_id"],
                row["case_id"],
                str(row.get("image_path", "")),
                str(row.get("label_path", "")),
            ]
            digest.update(
                json.dumps(identity, separators=(",", ":"), ensure_ascii=True).encode(
                    "utf-8"
                )
            )
            digest.update(b"\n")

        metadata = {
            "schema_version": self.SCHEMA_VERSION,
            "algorithm": self.ALGORITHM,
            "epoch": epoch + 1,
            "samples": len(schedule),
            "schedule_sha256": digest.hexdigest(),
            "target_quota": dict(sorted(target_quota.items())),
            "target_patient_exposure": {
                target: {
                    "patients": len(exposures),
                    "min": min(exposures.values()),
                    "max": max(exposures.values()),
                }
                for target, exposures in sorted(patient_exposures.items())
            },
        }
        cached = tuple(schedule)
        self._schedule_cache[epoch] = cached
        self._metadata_cache[epoch] = metadata
        return cached, metadata

    def indices_for_epoch(self, epoch: int) -> list[int]:
        schedule, _ = self._build_epoch(epoch)
        return list(schedule)

    def metadata_for_epoch(self, epoch: int) -> dict:
        _, metadata = self._build_epoch(epoch)
        return metadata

    def __iter__(self):
        return iter(self.indices_for_epoch(self.epoch))


def make_train_sampler(
    rows: list[dict],
    mode: str,
    seed: int,
    samples_per_epoch: int | None,
) -> Sampler[int] | None:
    if mode == "natural" and samples_per_epoch is None:
        return None
    if mode == "target_patient_balanced":
        return ExactTargetPatientSampler(
            rows,
            seed=seed,
            samples_per_epoch=samples_per_epoch,
        )
    generator = make_seeded_generator(seed)
    return WeightedRandomSampler(
        weights=target_sampling_weights(rows, mode),
        num_samples=samples_per_epoch or len(rows),
        replacement=True,
        generator=generator,
    )


def make_seeded_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


class BratsJsonlDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image3d, gt3d = engine.load_case(row["image_path"], row["label_path"])
        return {
            "image": image3d.squeeze(0).float(),
            "label": gt3d.squeeze(0).float(),
            "case_id": row["case_id"],
            "target": row.get("target", ""),
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def macro_dice_from_per_target(metrics: dict[str, dict]) -> float:
    values = [
        float(target_metrics["dice"])
        for target_metrics in metrics.values()
        if int(target_metrics.get("dice_samples", 0)) > 0
    ]
    return float(np.mean(values)) if values else 0.0


def selection_checkpoint_name(metric: str) -> str:
    names = {
        "val_macro_dice": "sam_model_val_macro_dice_best.pth",
        "val_dice": "sam_model_val_dice_best.pth",
        "val_loss": "sam_model_val_loss_best.pth",
    }
    if metric not in names:
        raise ValueError(f"unsupported selection metric: {metric}")
    return names[metric]


def metric_is_better(metric: str, value: float, previous: float) -> bool:
    if metric == "val_loss":
        return value < previous
    if metric in {"val_macro_dice", "val_dice"}:
        return value > previous
    raise ValueError(f"unsupported selection metric: {metric}")


def z_normalize_batch(image: torch.Tensor) -> torch.Tensor:
    out = image.clone()
    for idx in range(out.shape[0]):
        mask = out[idx] > 0
        if not bool(mask.any()):
            continue
        values = out[idx][mask]
        mean = values.mean()
        std = values.std(unbiased=False).clamp_min(1e-6)
        out[idx] = (out[idx] - mean) / std
    return out


def fallback_click(fn_mask: torch.Tensor, gt_mask: torch.Tensor) -> tuple[torch.Tensor, bool]:
    positives = torch.argwhere(gt_mask)
    if len(positives) > 0:
        point = positives[np.random.randint(len(positives))]
        return point, True
    shape = fn_mask.shape
    point = torch.tensor(
        [np.random.randint(size) for size in shape],
        device=fn_mask.device,
        dtype=torch.long,
    )
    return point, False


def safe_random_click(prev_seg: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred = prev_seg > 0.5
    true = gt > 0
    fn = torch.logical_and(true, torch.logical_not(pred))
    fp = torch.logical_and(torch.logical_not(true), pred)

    points = []
    labels = []
    for idx in range(prev_seg.shape[0]):
        error = torch.logical_or(fn[idx, 0], fp[idx, 0])
        candidates = torch.argwhere(error)
        if len(candidates) > 0:
            point = candidates[np.random.randint(len(candidates))]
            is_positive = bool(fn[idx, 0, point[0], point[1], point[2]])
        else:
            point, is_positive = fallback_click(fn[idx, 0], true[idx, 0])
        points.append(point.reshape(1, 1, 3))
        labels.append(torch.tensor([[int(is_positive)]], device=prev_seg.device, dtype=torch.long))
    return torch.cat(points, dim=0).to(prev_seg.device), torch.cat(labels, dim=0)


def safe_ritm_click(prev_seg: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    import edt

    pred = prev_seg > 0.5
    true = gt > 0
    fn = torch.logical_and(true, torch.logical_not(pred))
    fp = torch.logical_and(torch.logical_not(true), pred)

    points = []
    labels = []
    for idx in range(prev_seg.shape[0]):
        fn_single = F.pad(fn[idx, 0], (1, 1, 1, 1, 1, 1), "constant", value=0).to(torch.uint8)
        fp_single = F.pad(fp[idx, 0], (1, 1, 1, 1, 1, 1), "constant", value=0).to(torch.uint8)
        fn_dt = torch.tensor(
            edt.edt(fn_single.cpu().numpy(), black_border=True, parallel=4),
            device=prev_seg.device,
        )[1:-1, 1:-1, 1:-1]
        fp_dt = torch.tensor(
            edt.edt(fp_single.cpu().numpy(), black_border=True, parallel=4),
            device=prev_seg.device,
        )[1:-1, 1:-1, 1:-1]
        fn_max = torch.max(fn_dt)
        fp_max = torch.max(fp_dt)
        max_dist = torch.maximum(fn_max, fp_max)
        if float(max_dist) <= 0.0:
            point, is_positive = fallback_click(fn[idx, 0], true[idx, 0])
        else:
            dt = fn_dt if bool(fn_max > fp_max) else fp_dt
            candidates = torch.argwhere(dt > (max_dist / 2.0))
            if len(candidates) == 0:
                candidates = torch.argwhere(dt == max_dist)
            point = candidates[np.random.randint(len(candidates))]
            is_positive = bool(fn[idx, 0, point[0], point[1], point[2]])
        points.append(point.reshape(1, 1, 3))
        labels.append(torch.tensor([[int(is_positive)]], device=prev_seg.device, dtype=torch.long))
    return torch.cat(points, dim=0).to(prev_seg.device), torch.cat(labels, dim=0)


def sample_click(prev_seg: torch.Tensor, gt: torch.Tensor, method: str) -> tuple[torch.Tensor, torch.Tensor]:
    if method in {"default", "ritm"}:
        return safe_ritm_click(prev_seg, gt)
    if method == "random":
        return safe_random_click(prev_seg, gt)
    raise ValueError(f"unknown click method: {method}")


def configure_trainable(model: torch.nn.Module, strategy: str) -> list[torch.nn.Parameter]:
    for param in model.parameters():
        param.requires_grad = False
    adapters = list(peft.iter_qv_lora_modules(model))
    if strategy == "decoder_only":
        modules = [model.prompt_encoder, model.mask_decoder]
    elif strategy == "lora":
        if not adapters:
            raise RuntimeError("strategy=lora requires injected image-encoder Q/V LoRA")
        modules = [model.prompt_encoder, model.mask_decoder]
    elif strategy == "mask_decoder_only":
        modules = [model.mask_decoder]
    elif strategy == "full":
        modules = [model]
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    for module in modules:
        for param in module.parameters():
            param.requires_grad = True
    for adapter in adapters:
        for param in adapter.lora_parameters():
            param.requires_grad = True
    target_embed = getattr(model, "target_embed", None)
    if isinstance(target_embed, peft.TargetEmbedding):
        for param in target_embed.parameters():
            param.requires_grad = True
    return [param for param in model.parameters() if param.requires_grad]


def dice_scores_from_logits(logits: torch.Tensor, gt: torch.Tensor) -> list[float | None]:
    pred = torch.sigmoid(logits) > 0.5
    true = gt > 0
    scores = []
    for idx in range(pred.shape[0]):
        denom = pred[idx].sum() + true[idx].sum()
        if int(denom) == 0:
            scores.append(None)
        else:
            scores.append(
                (2.0 * torch.logical_and(pred[idx], true[idx]).sum() / denom).item()
            )
    return scores


def dice_from_logits(logits: torch.Tensor, gt: torch.Tensor) -> float:
    scores = [score for score in dice_scores_from_logits(logits, gt) if score is not None]
    return float(np.mean(scores)) if scores else 0.0


def interaction_forward(
    model: torch.nn.Module,
    image: torch.Tensor,
    gt: torch.Tensor,
    loss_fn: DiceCELoss,
    num_clicks: int,
    click_method: str,
    train_image_encoder: bool,
    target_ids: torch.Tensor | None = None,
    return_per_sample_losses: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if train_image_encoder:
        image_embedding = model.image_encoder(image)
    else:
        with torch.no_grad():
            image_embedding = model.image_encoder(image)
        image_embedding = image_embedding.detach()

    prev_masks = torch.zeros_like(gt, dtype=torch.float32)
    low_res = F.interpolate(prev_masks, size=(engine.CROP_SIZE // 4,) * 3)
    total_loss = image.sum() * 0.0
    per_sample_loss = image.new_zeros(image.shape[0]) if return_per_sample_losses else None
    for click_idx in range(num_clicks):
        method = "random" if click_idx > 1 else click_method
        points, labels = sample_click(prev_masks.detach(), gt.detach(), method)
        sparse_emb, dense_emb = model.prompt_encoder(
            points=[points, labels],
            boxes=None,
            masks=low_res,
        )
        if target_ids is not None:
            sparse_emb = peft.append_target_token(model, sparse_emb, target_ids)
        low_res, _ = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        prev_masks = F.interpolate(
            low_res,
            size=gt.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
        total_loss = total_loss + loss_fn(prev_masks, gt)
        if per_sample_loss is not None:
            for sample_idx in range(image.shape[0]):
                per_sample_loss[sample_idx] += loss_fn(
                    prev_masks[sample_idx : sample_idx + 1],
                    gt[sample_idx : sample_idx + 1],
                )
    mean_loss = total_loss / max(num_clicks, 1)
    if per_sample_loss is not None:
        return prev_masks, mean_loss, per_sample_loss / max(num_clicks, 1)
    return prev_masks, mean_loss


def batch_targets(value, batch_size: int) -> list[str]:
    if isinstance(value, str):
        targets = [value]
    else:
        targets = [str(target) for target in value]
    if len(targets) != batch_size:
        raise ValueError(
            f"target batch has {len(targets)} entries for batch_size={batch_size}"
        )
    return targets


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: DiceCELoss,
    device: torch.device,
    num_clicks: int,
    click_method: str,
    train_image_encoder: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    accumulation_steps: int = 1,
    amp_dtype: torch.dtype | None = None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    if not train_image_encoder:
        model.image_encoder.eval()

    loss_total = 0.0
    loss_sample_count = 0
    dice_total = 0.0
    dice_sample_count = 0
    target_totals: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "loss_total": 0.0,
            "samples": 0,
            "dice_total": 0.0,
            "dice_samples": 0,
        }
    )
    if training:
        optimizer.zero_grad(set_to_none=True)

    iterator = tqdm(loader, desc="train" if training else "val", leave=False)
    grad_context = torch.enable_grad if training else torch.no_grad
    with grad_context():
        for step, batch in enumerate(iterator):
            image = z_normalize_batch(batch["image"].to(device, non_blocking=True).float())
            gt = batch["label"].to(device, non_blocking=True).float()
            target_ids = None
            if isinstance(getattr(model, "target_embed", None), peft.TargetEmbedding):
                target_ids = peft.target_ids_for_model(model, batch["target"], device=device)
            with torch.amp.autocast(
                "cuda",
                enabled=amp_dtype is not None and device.type == "cuda",
                dtype=amp_dtype or torch.float16,
            ):
                forward_result = interaction_forward(
                    model=model,
                    image=image,
                    gt=gt,
                    loss_fn=loss_fn,
                    num_clicks=num_clicks,
                    click_method=click_method,
                    train_image_encoder=train_image_encoder,
                    target_ids=target_ids,
                    return_per_sample_losses=not training,
                )
            if training:
                logits, loss = forward_result
                per_sample_losses = None
                scaled_loss = loss / max(accumulation_steps, 1)
                if scaler is None:
                    scaled_loss.backward()
                else:
                    scaler.scale(scaled_loss).backward()
                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
                    if scaler is None:
                        optimizer.step()
                    else:
                        scaler.step(optimizer)
                        scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                logits, loss, per_sample_losses = forward_result

            batch_size = int(gt.shape[0])
            if per_sample_losses is None:
                loss_total += float(loss.detach().cpu()) * batch_size
            else:
                loss_total += float(per_sample_losses.detach().sum().cpu())
            loss_sample_count += batch_size

            dice_scores = dice_scores_from_logits(logits.detach(), gt.detach())
            for score in dice_scores:
                if score is not None:
                    dice_total += score
                    dice_sample_count += 1

            if per_sample_losses is not None:
                target_names = batch_targets(batch["target"], batch_size)
                sample_loss_values = per_sample_losses.detach().float().cpu().tolist()
                for target, sample_loss, score in zip(
                    target_names,
                    sample_loss_values,
                    dice_scores,
                ):
                    totals = target_totals[target]
                    totals["loss_total"] += float(sample_loss)
                    totals["samples"] += 1
                    if score is not None:
                        totals["dice_total"] += score
                        totals["dice_samples"] += 1

            iterator.set_postfix(
                loss=f"{loss_total / max(loss_sample_count, 1):.4f}",
                dice=f"{dice_total / max(dice_sample_count, 1):.4f}",
            )

    per_target = {
        target: {
            "loss": float(totals["loss_total"]) / max(int(totals["samples"]), 1),
            "dice": float(totals["dice_total"]) / max(int(totals["dice_samples"]), 1),
            "samples": int(totals["samples"]),
            "dice_samples": int(totals["dice_samples"]),
        }
        for target, totals in sorted(target_totals.items())
    }
    return {
        "loss": loss_total / max(loss_sample_count, 1),
        "dice": dice_total / max(dice_sample_count, 1),
        "macro_dice": macro_dice_from_per_target(per_target),
        "samples": loss_sample_count,
        "dice_samples": dice_sample_count,
        "per_target": per_target,
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    args: argparse.Namespace,
    history: list[dict],
    best: dict,
    adapter_config: dict | None = None,
    checkpoint_metadata: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
        "history": history,
        "best": best,
        "checkpoint_metadata": checkpoint_metadata or {},
    }
    if adapter_config is not None:
        payload["adapter_config"] = peft.normalize_adapter_config(adapter_config)
    tmp_path = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def append_history_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_dice",
                "train_schedule_sha256",
                "train_target_quota",
                "train_target_patient_exposure",
                "val_loss",
                "val_dice",
                "val_macro_dice",
                "val_by_target",
                "elapsed_sec",
            ],
        )
        if not exists:
            writer.writeheader()
        csv_row = dict(row)
        for field in (
            "train_target_quota",
            "train_target_patient_exposure",
            "val_by_target",
        ):
            csv_row[field] = json.dumps(
                csv_row.get(field, {}),
                sort_keys=True,
                separators=(",", ":"),
            )
        writer.writerow(csv_row)


def append_target_history_csv(path: Path, epoch: int, metrics: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "target", "val_loss", "val_dice", "samples", "dice_samples"],
        )
        if not exists:
            writer.writeheader()
        for target, values in sorted(metrics.items()):
            writer.writerow(
                {
                    "epoch": epoch,
                    "target": target,
                    "val_loss": values["loss"],
                    "val_dice": values["dice"],
                    "samples": values["samples"],
                    "dice_samples": values["dice_samples"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/fine-tune SAM-Med3D on AFlow BraTS JSONL splits."
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--targets", default="", help="comma-separated target filter")
    parser.add_argument("--strategy", choices=["decoder_only", "mask_decoder_only", "lora", "full"], default="decoder_only")
    parser.add_argument("--target-token", action="store_true")
    parser.add_argument("--encoder-qv-lora-rank", type=int, default=0)
    parser.add_argument("--encoder-qv-lora-alpha", type=float, default=peft.DEFAULT_LORA_ALPHA)
    parser.add_argument("--checkpoint", default=engine.DEFAULT_CKPT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-clicks", type=int, default=5)
    parser.add_argument("--click-method", choices=["default", "ritm", "random"], default="default")
    parser.add_argument("--max-train-cases", type=int, default=0)
    parser.add_argument("--max-val-cases", type=int, default=0)
    parser.add_argument(
        "--subset-mode",
        choices=["balanced", "sorted"],
        default="balanced",
        help="multi-target subset selection; single-target ordering is unchanged",
    )
    parser.add_argument("--train-rows-per-target", type=int, default=0)
    parser.add_argument("--val-rows-per-target", type=int, default=0)
    parser.add_argument(
        "--target-sampling",
        choices=["natural", "balanced", "target_patient_balanced"],
        default="natural",
        help=(
            "training row exposure; target_patient_balanced uses exact per-target "
            "quotas and patient/timepoint cycles"
        ),
    )
    parser.add_argument(
        "--train-samples-per-epoch",
        type=int,
        default=0,
        help=(
            "0 uses the legacy mode default; target_patient_balanced rounds the row "
            "count down to a target-divisible total, while an explicit value must be "
            "divisible by the number of targets"
        ),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--val-seed",
        type=int,
        default=None,
        help="fixed validation click seed; defaults to --seed",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["val_macro_dice", "val_dice", "val_loss"],
        default="val_dice",
        help="metric used for the single selected-best checkpoint",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp-dtype",
        choices=["fp16", "bf16", "none"],
        default="fp16",
        help="CUDA autocast dtype; use bf16 on B200",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="compatibility alias that overrides --amp-dtype and disables AMP",
    )
    parser.add_argument("--overfit", action="store_true", help="use the train rows as validation for M0 sanity")
    parser.add_argument(
        "--allow-nonempty-output",
        action="store_true",
        help="allow appending to or overwriting files in a non-empty output directory",
    )
    args = parser.parse_args()

    for name in (
        "max_train_cases",
        "max_val_cases",
        "train_rows_per_target",
        "val_rows_per_target",
        "train_samples_per_epoch",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")

    if args.strategy == "lora" and args.encoder_qv_lora_rank == 0:
        args.encoder_qv_lora_rank = peft.DEFAULT_LORA_RANK
        print(
            "WARNING: --strategy lora is a legacy alias; enabling rank-4 Q/V LoRA. "
            "Prefer --strategy decoder_only --encoder-qv-lora-rank 4.",
            flush=True,
        )

    if args.batch_size != 1:
        print(
            "WARNING: batch_size > 1 uses this script's batch-safe click sampler; "
            "eval remains case-by-case for comparability.",
            flush=True,
        )

    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, allow_nonempty=args.allow_nonempty_output)
    seed_everything(args.seed)
    targets = {target.strip() for target in args.targets.split(",") if target.strip()} or None

    train_rows = filter_rows(
        read_jsonl(Path(args.train_jsonl)),
        targets,
        args.max_train_cases or None,
        balance_targets=args.subset_mode == "balanced",
        rows_per_target=args.train_rows_per_target or None,
        seed=args.seed,
    )
    if args.overfit:
        val_rows = list(train_rows)
    else:
        val_rows = filter_rows(
            read_jsonl(Path(args.val_jsonl)),
            targets,
            args.max_val_cases or None,
            balance_targets=args.subset_mode == "balanced",
            rows_per_target=args.val_rows_per_target or None,
            seed=resolve_val_seed(args),
        )
    if not train_rows:
        raise RuntimeError("empty train set after filtering")
    if not val_rows:
        raise RuntimeError("empty val set after filtering")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    amp_dtype = resolve_amp_dtype(args, device)
    requested_adapter_config = peft.make_adapter_config(
        target_token=args.target_token,
        lora_rank=args.encoder_qv_lora_rank,
        lora_alpha=args.encoder_qv_lora_alpha,
        model_type=engine.MODEL_TYPE,
    )
    model, adapter_config = peft.load_model_with_adapters(
        args.checkpoint,
        lambda model_type: engine.sam_model_registry3D[model_type](checkpoint=None),
        device=device,
        requested_config=requested_adapter_config,
        default_model_type=engine.MODEL_TYPE,
    )
    trainable = configure_trainable(model, args.strategy)
    train_image_encoder = any(
        param.requires_grad for param in model.image_encoder.parameters()
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_grad_scaler(device, amp_dtype)
    loss_fn = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")

    # Adapter/token injection consumes RNG; start every factorial arm from the same state.
    seed_everything(args.seed)
    train_sampler = make_train_sampler(
        train_rows,
        mode=args.target_sampling,
        seed=args.seed,
        samples_per_epoch=args.train_samples_per_epoch or None,
    )
    train_loader_generator = make_seeded_generator(args.seed + 1)
    train_loader = DataLoader(
        BratsJsonlDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        generator=train_loader_generator,
    )
    val_loader = DataLoader(
        BratsJsonlDataset(val_rows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    train_target_counts = {
        target: sum(1 for row in train_rows if row["target"] == target)
        for target in sorted({row["target"] for row in train_rows})
    }
    val_target_counts = {
        target: sum(1 for row in val_rows if row["target"] == target)
        for target in sorted({row["target"] for row in val_rows})
    }
    sampling_weights = target_sampling_weights(train_rows, args.target_sampling)
    sampling_mass: dict[str, float] = defaultdict(float)
    for row_item, weight in zip(train_rows, sampling_weights.tolist()):
        sampling_mass[str(row_item.get("target", ""))] += float(weight)
    total_sampling_mass = sum(sampling_mass.values())
    target_sampling_probability = {
        target: mass / total_sampling_mass
        for target, mass in sorted(sampling_mass.items())
    }
    exact_schedule_config = None
    if isinstance(train_sampler, ExactTargetPatientSampler):
        exact_schedule_config = {
            "schema_version": train_sampler.SCHEMA_VERSION,
            "algorithm": train_sampler.ALGORITHM,
            "requested_samples_per_epoch": args.train_samples_per_epoch,
            "effective_samples_per_epoch": len(train_sampler),
            "schedules": [
                train_sampler.metadata_for_epoch(epoch)
                for epoch in range(args.epochs)
            ],
        }
    run_config = {
        "args": vars(args),
        "adapter_config": adapter_config,
        "effective_amp_dtype": amp_dtype_name(amp_dtype),
        "effective_train_seed": args.seed,
        "train_loader_seed": args.seed + 1,
        "train_sampler_seed": args.seed,
        "effective_val_seed": resolve_val_seed(args),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_targets": train_target_counts,
        "val_targets": val_target_counts,
        "train_samples_per_epoch": len(train_loader.sampler),
        "target_sampling_probability": target_sampling_probability,
        "exact_train_schedule": exact_schedule_config,
        "selection_metric": args.selection_metric,
        "selection_checkpoint": selection_checkpoint_name(args.selection_metric),
        "trainable_parameters": sum(param.numel() for param in trainable),
        "total_parameters": sum(param.numel() for param in model.parameters()),
    }
    (output_dir / "train_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(run_config, indent=2, sort_keys=True), flush=True)

    history = []
    best = {
        "val_dice": {"value": -1.0, "epoch": -1},
        "val_macro_dice": {"value": -1.0, "epoch": -1},
        "val_loss": {"value": float("inf"), "epoch": -1},
        "selection": {
            "metric": args.selection_metric,
            "checkpoint": selection_checkpoint_name(args.selection_metric),
            "value": -1.0 if args.selection_metric != "val_loss" else float("inf"),
            "epoch": -1,
        },
    }
    start = time.time()
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}", flush=True)
        train_schedule = None
        if isinstance(train_sampler, ExactTargetPatientSampler):
            train_sampler.set_epoch(epoch)
            train_schedule = train_sampler.metadata_for_epoch(epoch)
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            num_clicks=args.num_clicks,
            click_method=args.click_method,
            train_image_encoder=train_image_encoder,
            optimizer=optimizer,
            scaler=scaler,
            accumulation_steps=args.accumulation_steps,
            amp_dtype=amp_dtype,
        )
        with fixed_rng_seed(resolve_val_seed(args), device):
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                num_clicks=args.num_clicks,
                click_method=args.click_method,
                train_image_encoder=train_image_encoder,
                optimizer=None,
                scaler=None,
                accumulation_steps=1,
                amp_dtype=amp_dtype,
            )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_schedule_sha256": (
                train_schedule["schedule_sha256"] if train_schedule else ""
            ),
            "train_target_quota": (
                train_schedule["target_quota"] if train_schedule else {}
            ),
            "train_target_patient_exposure": (
                train_schedule["target_patient_exposure"] if train_schedule else {}
            ),
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_macro_dice": val_metrics["macro_dice"],
            "val_by_target": val_metrics["per_target"],
            "elapsed_sec": time.time() - start,
        }
        history.append(row)
        append_history_csv(output_dir / "history.csv", row)
        append_target_history_csv(
            output_dir / "val_by_target.csv",
            epoch=epoch + 1,
            metrics=val_metrics["per_target"],
        )
        print(json.dumps(row, sort_keys=True), flush=True)

        improved_metrics = set()
        for metric in ("val_loss", "val_dice", "val_macro_dice"):
            value = float(row[metric])
            if metric_is_better(metric, value, float(best[metric]["value"])):
                best[metric] = {"value": value, "epoch": epoch + 1}
                improved_metrics.add(metric)
        selection_improved = args.selection_metric in improved_metrics
        if selection_improved:
            best["selection"].update(
                value=float(row[args.selection_metric]),
                epoch=epoch + 1,
            )

        save_checkpoint(
            output_dir / "sam_model_latest.pth",
            model,
            optimizer,
            scaler,
            epoch,
            args,
            history,
            best,
            adapter_config,
            checkpoint_metadata={
                "kind": "latest",
                "metric": args.selection_metric,
                "metric_value": float(row[args.selection_metric]),
                "is_selected_best": selection_improved,
            },
        )
        if selection_improved:
            save_checkpoint(
                output_dir / selection_checkpoint_name(args.selection_metric),
                model,
                optimizer,
                scaler,
                epoch,
                args,
                history,
                best,
                adapter_config,
                checkpoint_metadata={
                    "kind": "selected_best",
                    "metric": args.selection_metric,
                    "metric_value": float(row[args.selection_metric]),
                    "metric_epoch": epoch + 1,
                },
            )

    (output_dir / "best.json").write_text(
        json.dumps(best, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"TRAIN_DONE output_dir={output_dir} best={best}", flush=True)


if __name__ == "__main__":
    main()
