#!/usr/bin/env python3
"""Paired Full972 anchor-vs-GT-best-teacher routing distillation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterator

import numpy as np
import torch
from monai.losses import DiceCELoss
from torch import nn
from torch.utils.data import DataLoader, Sampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator  # noqa: E402
from scripts.build_t1ce_wt_native_volume_artifact import native_sampling_weights  # noqa: E402
from scripts.train_t1ce_wt_capg3d import (  # noqa: E402
    BrainCropDataset,
    augment_batch,
    case_view_metadata,
    dice_per_case,
    read_case_ids,
    sha256_file,
    subject_id,
    write_json_atomic,
)
from scripts.train_t1ce_wt_plain_long_schedule import learning_rate_for_epoch  # noqa: E402
from scripts.train_t1ce_wt_plain_voxel_kd_continuation import (  # noqa: E402
    KD_WEIGHT,
    TEMPERATURE,
    binary_voxel_kl,
)
from scripts.train_t1ce_wt_voxel_kd_972_draw_budget import (  # noqa: E402
    state_dict_sha256,
    tensor_sha256,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_gt_best_teacher_routing"
CHECKPOINT_SCHEMA = SCHEMA + ".checkpoint"
ARMS = ("anchor_repeat3_control", "gt_best_of3_candidate")
IDEA_SHA256 = "eafc1c829017e35915c2730ffe58c125e900a05c344417cbf7cede4a3d45c23a"
PREREG_SHA256 = "36287d80e63cc0cc6ad1e965cd4bf6046b8e5ea02e8caf0343217101f85687f7"
FULL972_MANIFEST_SHA256 = "6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65"
CACHE_INDEX_SHA256 = "7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd"
NATIVE_ARTIFACT_SHA256 = "32be09856d71e78f512792c8bcf8f5f9ee137e6ba1e4358862ff2a61ba828c31"
TEACHER_FILE_SHA256 = (
    "f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b",
    "48c366ef9f5d3ba5c667482deda6df90fa7cd4bf03c314017e15c13f161dfd38",
    "00db7860b7951d1f4b0affde6cad558f679421ba93dfe69157c31bd497b19ddb",
)
TEACHER_STATE_SHA256 = (
    "f24201eecc7bf4aaff92b8e0a50c85a31e6913f5c9701cca6ad7df3439618ea7",
    "445435848bfaf9e0afae1d62d4f4e2a1cd187f2b09a2ec55f128bbf35fa097b3",
    "3b60ba0907f0db39c191c7a675d0632d05202af56388201c5407caea6865de11",
)
STUDENT_SEED = 20261040
STOP_EPOCH = 168
SCHEDULE_TOTAL_EPOCHS = 200
PARAMETER_COUNT = 4_700_897
SAMPLER_DRAWS = 972
CONSUMED_DRAWS = 968
STEPS_PER_EPOCH = 121
MAX_PEAK_BYTES = 24 * 1024**3
MAX_FORMAL_SECONDS = 6 * 60 * 60


def per_case_soft_dice_loss(
    teacher_logits: torch.Tensor,
    label: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    if teacher_logits.shape != label.shape:
        raise ValueError("teacher logits/label shape mismatch")
    if not bool(torch.isfinite(teacher_logits).all()) or not bool(torch.isfinite(label).all()):
        raise FloatingPointError("non-finite routing input")
    probability = torch.sigmoid(teacher_logits.float())
    target = label.float()
    intersection = (probability * target).flatten(1).sum(dim=1)
    denominator = probability.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
    return 1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)


def select_gt_best_teacher_logits(
    teacher_logits: list[torch.Tensor],
    label: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(teacher_logits) != 3:
        raise ValueError("routing requires exactly three teacher logits")
    if not all(value.shape == label.shape for value in teacher_logits):
        raise ValueError("teacher logit shapes differ from label")
    losses = torch.stack([per_case_soft_dice_loss(value, label) for value in teacher_logits], dim=1)
    indices = losses.argmin(dim=1)
    stacked = torch.stack([value.detach() for value in teacher_logits], dim=1)
    batch_indices = torch.arange(label.shape[0], device=label.device)
    selected = stacked[batch_indices, indices].detach()
    return selected, indices.detach(), losses.detach()


class NativeFullDataset(BrainCropDataset):
    def __init__(self, *args: Any, native_volumes: dict[str, int], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.native_volumes = native_volumes

    def sampling_weights(self) -> torch.Tensor:
        return native_sampling_weights(self.case_ids, self.native_volumes)


class AuditedWeightedSampler(Sampler[int]):
    def __init__(self, weights: torch.Tensor, num_samples: int, seed: int) -> None:
        self.weights = weights.to(dtype=torch.float64, device="cpu")
        self.num_samples = int(num_samples)
        self.generator = torch.Generator().manual_seed(seed)
        self.last_indices: list[int] | None = None

    def __iter__(self) -> Iterator[int]:
        values = torch.multinomial(self.weights, self.num_samples, replacement=True, generator=self.generator).tolist()
        self.last_indices = [int(value) for value in values]
        return iter(self.last_indices)

    def __len__(self) -> int:
        return self.num_samples


def index_sha256(indices: list[int]) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def load_teacher(path: Path, expected_file_sha: str, expected_state_sha: str, device: torch.device) -> torch.nn.Module:
    if sha256_file(path) != expected_file_sha:
        raise ValueError("frozen teacher file SHA mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != 168 or checkpoint.get("model_state_sha256") != expected_state_sha:
        raise ValueError("frozen teacher metadata mismatch")
    if state_dict_sha256(checkpoint["model_state_dict"]) != expected_state_sha:
        raise ValueError("frozen teacher model-state mismatch")
    for key in ("validation_access", "fixed_validation108_access", "outer_selector_access", "confirmation_access", "test_access"):
        if checkpoint.get(key) is not False:
            raise ValueError("teacher declares prior evaluation access")
    model = AutomaticPromptLocator(center_aware=False, init_filters=16, dropout=0.1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval().requires_grad_(False)


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.arm not in ARMS:
        raise ValueError("unexpected routing arm")
    if args.seed != STUDENT_SEED or args.batch_size != 8:
        raise ValueError("student seed/batch differ from preregistration")
    if args.stop_epoch != STOP_EPOCH or args.schedule_total_epochs != SCHEDULE_TOTAL_EPOCHS:
        raise ValueError("epoch/schedule differ from preregistration")
    frozen = {
        args.train_manifest: FULL972_MANIFEST_SHA256,
        args.idea: IDEA_SHA256,
        args.prereg: PREREG_SHA256,
        args.native_artifact: NATIVE_ARTIFACT_SHA256,
    }
    for path, expected in frozen.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen input SHA mismatch: {path}")
    if sha256_file(args.cache / "index.json") != CACHE_INDEX_SHA256:
        raise ValueError("train cache index SHA mismatch")
    teacher_paths = [args.teacher_seed32, args.teacher_seed40, args.teacher_seed41]
    for path, expected in zip(teacher_paths, TEACHER_FILE_SHA256):
        if sha256_file(path) != expected:
            raise ValueError("teacher checkpoint file SHA mismatch")
    artifact = json.loads(args.native_artifact.read_text())
    if artifact.get("status") != "passed" or artifact.get("audit", {}).get("all_gates_passed") is not True:
        raise ValueError("native-volume artifact did not pass")
    native_volumes = {str(key): int(value) for key, value in artifact["native_wt_voxels"].items()}

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GT-best teacher routing training requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    limit = args.limit_train_cases if args.smoke else None
    train_data = NativeFullDataset(
        args.cache,
        expected_split="train",
        case_ids=read_case_ids(args.train_manifest),
        limit=limit,
        native_volumes=native_volumes,
    )
    subjects = {subject_id(case_id) for case_id in train_data.case_ids}
    if not args.smoke and (len(train_data) != 972 or len(subjects) != 444):
        raise ValueError("formal Full972 case/subject count mismatch")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    data_role = {"train": case_view_metadata(args.train_manifest), "cases": len(train_data), "subjects": len(subjects)}
    write_json_atomic(output_dir / "STARTED.json", {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "started_before_optimizer_step",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "data_role": data_role,
        "hyperparameters": serializable_args(args),
        "access": {"train_only_inner": False, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False},
    })

    weights = train_data.sampling_weights()
    draws = args.smoke_steps * args.batch_size if args.smoke else SAMPLER_DRAWS
    sampler = AuditedWeightedSampler(weights, draws, args.seed)
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    expected_steps = args.smoke_steps if args.smoke else STEPS_PER_EPOCH
    if len(loader) != expected_steps:
        raise RuntimeError("student optimizer-step count mismatch")
    teachers = [
        load_teacher(path, file_sha, state_sha, device)
        for path, file_sha, state_sha in zip(teacher_paths, TEACHER_FILE_SHA256, TEACHER_STATE_SHA256)
    ]
    model = AutomaticPromptLocator(center_aware=False, init_filters=16, dropout=0.1).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise ValueError("student parameter-count mismatch")
    initial_state_sha = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay, foreach=None, fused=None)
    loss_function = DiceCELoss(sigmoid=True, squared_pred=False, lambda_dice=1.0, lambda_ce=0.1)

    history: list[dict[str, Any]] = []
    total_routing = torch.zeros(3, dtype=torch.int64)
    started = time.time()
    epochs_to_run = 1 if args.smoke else args.stop_epoch
    for epoch in range(1, epochs_to_run + 1):
        lr = learning_rate_for_epoch(epoch, base_lr=args.base_lr, min_lr=args.min_lr, constant_epochs=100, total_epochs=args.schedule_total_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        losses: list[float] = []
        supervised_losses: list[float] = []
        kd_losses: list[float] = []
        train_dice: list[float] = []
        routing_counts = torch.zeros(3, dtype=torch.int64)
        first_batch_sha: dict[str, str] | None = None
        first_student_loss: float | None = None
        teacher_forward_count = 0
        for step, batch in enumerate(loader):
            image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            image, label = augment_batch(image, label, max_shift=8)
            if epoch == 1 and step == 0:
                first_batch_sha = {"image": tensor_sha256(image), "label": tensor_sha256(label)}
            teacher_logits: list[torch.Tensor] = []
            with torch.no_grad():
                if args.arm == "anchor_repeat3_control":
                    for _ in range(3):
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            teacher_logits.append(teachers[0](image).segmentation_logits.float())
                        teacher_forward_count += 1
                    if any(float((value - teacher_logits[0]).abs().max()) != 0.0 for value in teacher_logits[1:]):
                        raise RuntimeError("repeated anchor logits are not exact")
                    target_logits = teacher_logits[0].detach()
                    selected_indices = torch.zeros(label.shape[0], device=device, dtype=torch.long)
                else:
                    for teacher in teachers:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            teacher_logits.append(teacher(image).segmentation_logits.float())
                        teacher_forward_count += 1
                    target_logits, selected_indices, _ = select_gt_best_teacher_logits(teacher_logits, label)
            routing_counts += torch.bincount(selected_indices.cpu(), minlength=3)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(image)
                supervised = loss_function(output.segmentation_logits, label)
            kd = binary_voxel_kl(output.segmentation_logits.float(), target_logits, temperature=TEMPERATURE)
            loss = supervised + KD_WEIGHT * kd
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite routing student loss")
            if epoch == 1 and step == 0:
                first_student_loss = float(loss.detach())
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
            if float(grad_norm) <= 0:
                raise RuntimeError("non-positive routing student gradient")
            if any(parameter.grad is not None for teacher in teachers for parameter in teacher.parameters()):
                raise RuntimeError("frozen routing teacher received gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
            supervised_losses.append(float(supervised.detach()))
            kd_losses.append(float(kd.detach()))
            train_dice.extend(dice_per_case(output.segmentation_logits.detach().float(), label).cpu().tolist())
        if sampler.last_indices is None:
            raise RuntimeError("sampler did not expose epoch indices")
        if int(routing_counts.sum()) != expected_steps * args.batch_size:
            raise RuntimeError("routing counts do not sum to consumed cases")
        if teacher_forward_count != expected_steps * 3:
            raise RuntimeError("teacher forward count mismatch")
        total_routing += routing_counts
        row = {
            "epoch": epoch,
            "learning_rate": lr,
            "train_loss": float(np.mean(losses)),
            "train_supervised_loss": float(np.mean(supervised_losses)),
            "train_kd_loss": float(np.mean(kd_losses)),
            "train_dice": float(np.mean(train_dice)),
            "optimizer_steps": len(losses),
            "consumed_draws": len(losses) * args.batch_size,
            "teacher_forward_count": teacher_forward_count,
            "routing_counts": [int(value) for value in routing_counts.tolist()],
            "sampler_index_sha256": index_sha256(sampler.last_indices[: expected_steps * args.batch_size]),
            "first_batch_sha256": first_batch_sha,
            "first_student_loss": first_student_loss,
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        write_json_atomic(output_dir / "progress.json", {"history": history, "total_routing_counts": total_routing.tolist()})
        print(
            f"arm={args.arm} epoch={epoch}/{args.stop_epoch} loss={row['train_loss']:.6f} "
            f"train={row['train_dice']:.6f} routing={row['routing_counts']}", flush=True,
        )

    completed_epoch = int(history[-1]["epoch"])
    checkpoint_path = output_dir / f"plain_{args.arm}_epoch{completed_epoch}.pth"
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "arm": args.arm,
        "epoch": completed_epoch,
        "model_state_dict": model.state_dict(),
        "model_state_sha256": state_dict_sha256(model.state_dict()),
        "initial_state_sha256": initial_state_sha,
        "optimizer_state_dict": optimizer.state_dict(),
        "architecture": {"class": "AutomaticPromptLocator", "init_filters": 16, "dropout": 0.1, "input_channels": ["T1ce"], "outputs": ["WT segmentation logit"]},
        "routing": {"method": args.arm, "teacher_file_sha256": list(TEACHER_FILE_SHA256), "total_counts": total_routing.tolist(), "teacher_forwards_per_step": 3},
        "distillation": {"temperature": TEMPERATURE, "kd_weight": KD_WEIGHT},
        "data_role": data_role,
        "documents": {"idea_sha256": IDEA_SHA256, "prereg_sha256": PREREG_SHA256},
        "selection": "fixed epoch168; no validation/inner access during training",
        "access": {"train_only_inner": False, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False},
    }
    torch.save(checkpoint, checkpoint_path)
    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reload_exact = reloaded["model_state_sha256"] == state_dict_sha256(reloaded["model_state_dict"])
    elapsed = time.time() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    gates = {
        "epoch_count_exact": len(history) == (1 if args.smoke else STOP_EPOCH),
        "all_steps_exact": all(row["optimizer_steps"] == expected_steps for row in history),
        "all_teacher_forwards_exact": all(row["teacher_forward_count"] == expected_steps * 3 for row in history),
        "all_losses_finite": all(np.isfinite(row[key]) for row in history for key in ("train_loss", "train_supervised_loss", "train_kd_loss", "train_dice")),
        "checkpoint_reload_exact": reload_exact,
        "peak_at_most_24gib": peak <= MAX_PEAK_BYTES,
        "elapsed_within_6h": elapsed <= MAX_FORMAL_SECONDS,
    }
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete" if all(gates.values()) else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm,
        "all_gates_passed": all(gates.values()),
        "gates": gates,
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
        "model_state_sha256": checkpoint["model_state_sha256"],
        "initial_state_sha256": initial_state_sha,
        "routing": checkpoint["routing"],
        "data_role": data_role,
        "sampling_weight_sha256": tensor_sha256(weights),
        "history": history,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": peak,
        "access": checkpoint["access"],
    }
    write_json_atomic(output_dir / "TRAIN_DONE.json", report)
    print(f"TRAIN_DONE arm={args.arm} checkpoint={checkpoint_path} sha256={report['checkpoint']['sha256']}", flush=True)
    if not report["all_gates_passed"]:
        raise RuntimeError("GT-best teacher routing engineering gates failed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    record = ROOT / "archive/research_records/2026-09-01_gt_best_teacher_routing_distillation"
    prior = ROOT / "archive/research_records/2026-08-21_voxel_kd_data972_native_sampling"
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--teacher-seed32", type=Path, required=True)
    parser.add_argument("--teacher-seed40", type=Path, required=True)
    parser.add_argument("--teacher-seed41", type=Path, required=True)
    parser.add_argument("--idea", type=Path, default=record / "IDEA.md")
    parser.add_argument("--prereg", type=Path, default=record / "PREREG.md")
    parser.add_argument("--native-artifact", type=Path, default=prior / "NATIVE_WT_VOLUMES.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=STUDENT_SEED)
    parser.add_argument("--stop-epoch", type=int, default=STOP_EPOCH)
    parser.add_argument("--schedule-total-epochs", type=int, default=SCHEDULE_TOTAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--limit-train-cases", type=int, default=8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

