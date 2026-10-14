#!/usr/bin/env python3
"""Paired Full972 AdamW/GCSAM training on the frozen train-only inner split."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
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
from scripts.gcsam_optimizer import PaperGCSAM  # noqa: E402
from scripts.train_t1ce_wt_capg3d import (  # noqa: E402
    BrainCropDataset,
    augment_batch,
    case_view_metadata,
    dice_per_case,
    read_case_ids,
    sha256_file,
    subject_id,
    validate,
    write_json_atomic,
)
from scripts.train_t1ce_wt_plain_voxel_kd_continuation import (  # noqa: E402
    EMA168_SHA256,
    HISTORICAL_SHA256,
    KD_WEIGHT,
    TEACHER_EMA_WEIGHT,
    TEMPERATURE,
    _model,
    binary_voxel_kl,
)
from scripts.train_t1ce_wt_voxel_kd_972_draw_budget import (  # noqa: E402
    state_dict_sha256,
    tensor_sha256,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_gcsam_inner"
CHECKPOINT_SCHEMA = SCHEMA + ".checkpoint"
ARMS = ("adamw_control", "gcsam_candidate")
FIT_SHA256 = "587dcd66828b4ad44cf23db224852511158af2fa11d00f66e269e99887ba8e6d"
TUNE_SHA256 = "aa091a3a5259745b6987fafe9d6915db2576e10478a991917d34106f972b2c11"
SPLIT_SHA256 = "716484f91b503d3890807db6e3fd28353c30cc86a788699cccca9a56bee2a8d8"
IDEA_SHA256 = "0c3516c686e13d6cb86b947f2a4fff50a45a936ee8c1ebc939bd9ac349b4250a"
PREREG_SHA256 = "7fe9d8039ede538ff9c47cba001d2ed44ef4769d3aa0142568692797c37ef603"
CACHE_INDEX_SHA256 = "7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd"
NATIVE_ARTIFACT_SHA256 = "32be09856d71e78f512792c8bcf8f5f9ee137e6ba1e4358862ff2a61ba828c31"
SAMPLER_DRAWS = 972
CONSUMED_DRAWS = 968
STEPS_PER_EPOCH = 121
PARAMETER_COUNT = 4_700_897
FROZEN_SEED = 20261002
RHO = 0.05
MAX_PEAK_BYTES = 24 * 1024**3
MAX_SECONDS = {"adamw_control": 6 * 60 * 60, "gcsam_candidate": 9 * 60 * 60}
FORBIDDEN_ROLE_TOKENS = ("selector", "confirmation", "sealed", "test270", "validation")


class NativeFitDataset(BrainCropDataset):
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
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=True,
            generator=self.generator,
        ).tolist()
        self.last_indices = [int(value) for value in indices]
        return iter(self.last_indices)

    def __len__(self) -> int:
        return self.num_samples


def index_sha256(indices: list[int]) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def assert_train_only_path(path: Path) -> None:
    lowered = str(path.resolve()).lower()
    if any(token in lowered for token in FORBIDDEN_ROLE_TOKENS):
        raise ValueError(f"forbidden evaluation-role token in training path: {path}")


def learning_rate_for_epoch(epoch: int, base_lr: float, min_lr: float) -> float:
    if not 1 <= epoch <= 200:
        raise ValueError("epoch outside frozen200-epoch schedule")
    if epoch <= 100:
        return float(base_lr)
    progress = (epoch - 101) / 99
    return float(min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def student_loss(
    model: nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    teacher_logits: torch.Tensor,
    loss_function: DiceCELoss,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(image)
        supervised = loss_function(output.segmentation_logits, label)
    kd = binary_voxel_kl(output.segmentation_logits.float(), teacher_logits, temperature=TEMPERATURE)
    loss = supervised + KD_WEIGHT * kd
    return loss, supervised, kd, output.segmentation_logits


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.arm not in ARMS:
        raise ValueError("unexpected GCSAM inner arm")
    if args.seed != FROZEN_SEED or args.batch_size != 8:
        raise ValueError("seed/batch differ from preregistration")
    if not args.smoke and args.epochs != 200:
        raise ValueError("formal inner requires exactly200 epochs")
    for path in (args.fit_case_ids, args.inner_tune_case_ids, args.split_manifest):
        assert_train_only_path(path)
    frozen = {
        args.fit_case_ids: FIT_SHA256,
        args.inner_tune_case_ids: TUNE_SHA256,
        args.split_manifest: SPLIT_SHA256,
        args.idea: IDEA_SHA256,
        args.prereg: PREREG_SHA256,
        args.native_artifact: NATIVE_ARTIFACT_SHA256,
    }
    for path, expected in frozen.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen input SHA mismatch: {path}")
    if sha256_file(args.cache / "index.json") != CACHE_INDEX_SHA256:
        raise ValueError("train cache index SHA mismatch")
    if sha256_file(args.historical_teacher_checkpoint) != HISTORICAL_SHA256:
        raise ValueError("historical teacher SHA mismatch")
    if sha256_file(args.ema_teacher_checkpoint) != EMA168_SHA256:
        raise ValueError("EMA teacher SHA mismatch")
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
        raise RuntimeError("GCSAM inner training requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    fit_limit = args.limit_fit_cases if args.smoke else None
    tune_limit = args.limit_inner_tune_cases if args.smoke else None
    fit_data = NativeFitDataset(
        args.cache,
        expected_split="train",
        case_ids=read_case_ids(args.fit_case_ids),
        limit=fit_limit,
        native_volumes=native_volumes,
    )
    tune_data = BrainCropDataset(
        args.cache,
        expected_split="train",
        case_ids=read_case_ids(args.inner_tune_case_ids),
        limit=tune_limit,
    )
    if not args.smoke and (len(fit_data) != 864 or len(tune_data) != 108):
        raise ValueError("formal fit/tune count mismatch")
    fit_subjects = {subject_id(case_id) for case_id in fit_data.case_ids}
    tune_subjects = {subject_id(case_id) for case_id in tune_data.case_ids}
    if set(fit_data.case_ids) & set(tune_data.case_ids) or fit_subjects & tune_subjects:
        raise ValueError("fit/tune leakage")
    if not args.smoke and (len(fit_subjects) != 394 or len(tune_subjects) != 50):
        raise ValueError("formal subject count mismatch")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    roles = {
        "fit": case_view_metadata(args.fit_case_ids),
        "inner_tune": case_view_metadata(args.inner_tune_case_ids),
        "fit_cases": len(fit_data),
        "fit_subjects": len(fit_subjects),
        "inner_tune_cases": len(tune_data),
        "inner_tune_subjects": len(tune_subjects),
        "case_overlap": 0,
        "subject_overlap": 0,
    }
    write_json_atomic(output_dir / "STARTED.json", {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started_before_optimizer_step",
        "arm": args.arm,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "data_roles": roles,
        "hyperparameters": serializable_args(args),
        "train_only_inner_access": True,
        "fixed_validation108_access": False,
        "outer_access": False,
        "confirmation_access": False,
        "test270_access": False,
    })

    weights = fit_data.sampling_weights()
    weight_sha = tensor_sha256(weights)
    draws = args.smoke_steps * args.batch_size if args.smoke else SAMPLER_DRAWS
    sampler = AuditedWeightedSampler(weights, draws, args.seed)
    fit_loader = DataLoader(
        fit_data,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    tune_loader = DataLoader(
        tune_data,
        batch_size=min(args.validation_batch_size, len(tune_data)),
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    expected_steps = args.smoke_steps if args.smoke else STEPS_PER_EPOCH
    if len(fit_loader) != expected_steps:
        raise RuntimeError("optimizer-step count mismatch")

    history_checkpoint = torch.load(args.historical_teacher_checkpoint, map_location="cpu", weights_only=False)
    ema_checkpoint = torch.load(args.ema_teacher_checkpoint, map_location="cpu", weights_only=False)
    historical_teacher = _model(history_checkpoint, device).eval().requires_grad_(False)
    ema_teacher = _model(ema_checkpoint, device).eval().requires_grad_(False)
    model = AutomaticPromptLocator(center_aware=False, init_filters=16, dropout=0.1).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise ValueError("student parameter-count mismatch")
    initial_state_sha = state_dict_sha256(model.state_dict())
    base_optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay, foreach=None, fused=None
    )
    optimizer: torch.optim.AdamW | PaperGCSAM
    optimizer = (
        PaperGCSAM(model.parameters(), base_optimizer, rho=RHO)
        if args.arm == "gcsam_candidate"
        else base_optimizer
    )
    loss_function = DiceCELoss(sigmoid=True, squared_pred=False, lambda_dice=1.0, lambda_ce=0.1)

    history: list[dict[str, Any]] = []
    best = {
        "pre100": {"scan": float("-inf"), "subject": float("-inf"), "epoch": 0},
        "post100": {"scan": float("-inf"), "subject": float("-inf"), "epoch": 0},
    }
    started = time.time()
    epochs_to_run = 1 if args.smoke else args.epochs
    for epoch in range(1, epochs_to_run + 1):
        lr = learning_rate_for_epoch(epoch, args.base_lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        losses: list[float] = []
        supervised_losses: list[float] = []
        kd_losses: list[float] = []
        train_dice: list[float] = []
        perturbation_norms: list[float] = []
        first_batch_sha: dict[str, str] | None = None
        first_batch_loss: float | None = None
        for step, batch in enumerate(fit_loader):
            image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            image, label = augment_batch(image, label, max_shift=8)
            if epoch == 1 and step == 0:
                first_batch_sha = {"image": tensor_sha256(image), "label": tensor_sha256(label)}
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                history_logits = historical_teacher(image).segmentation_logits.float()
                ema_logits = ema_teacher(image).segmentation_logits.float()
                teacher_logits = history_logits * 0.75 + ema_logits * 0.25

            optimizer.zero_grad(set_to_none=True)
            cpu_rng = torch.random.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(device)
            loss, supervised, kd, logits = student_loss(model, image, label, teacher_logits, loss_function)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite first-pass loss")
            if epoch == 1 and step == 0:
                first_batch_loss = float(loss.detach())
            loss.backward()
            if args.arm == "gcsam_candidate":
                assert isinstance(optimizer, PaperGCSAM)
                diagnostics = optimizer.first_step(zero_grad=True)
                perturbation_norms.append(float(diagnostics["perturbation_norm"]))
                torch.random.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng, device)
                try:
                    second_loss, _, _, _ = student_loss(model, image, label, teacher_logits, loss_function)
                    if not bool(torch.isfinite(second_loss)):
                        raise FloatingPointError("non-finite second-pass loss")
                    second_loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=5.0, error_if_nonfinite=True
                    )
                    if float(grad_norm) <= 0:
                        raise RuntimeError("non-positive second-pass gradient")
                    optimizer.second_step(zero_grad=True)
                except BaseException:
                    if optimizer.perturbation_active:
                        optimizer.restore()
                    raise
            else:
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=5.0, error_if_nonfinite=True
                )
                if float(grad_norm) <= 0:
                    raise RuntimeError("non-positive control gradient")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if any(
                parameter.grad is not None
                for teacher in (historical_teacher, ema_teacher)
                for parameter in teacher.parameters()
            ):
                raise RuntimeError("teacher received gradient")
            losses.append(float(loss.detach()))
            supervised_losses.append(float(supervised.detach()))
            kd_losses.append(float(kd.detach()))
            train_dice.extend(dice_per_case(logits.detach().float(), label).cpu().tolist())

        if sampler.last_indices is None:
            raise RuntimeError("sampler did not expose indices")
        sampler_sha = index_sha256(sampler.last_indices[: expected_steps * args.batch_size])
        with torch.no_grad():
            inner = validate(model, tune_loader, device)
        window = "pre100" if epoch <= 100 else "post100"
        key = (float(inner["locator_dsc_mean"]), float(inner["locator_subject_dsc_mean"]), -epoch)
        old = (float(best[window]["scan"]), float(best[window]["subject"]), -int(best[window]["epoch"]))
        new_high = key > old
        if new_high:
            best[window] = {"scan": key[0], "subject": key[1], "epoch": epoch}
            checkpoint_path = output_dir / f"{window}_best.pth"
            checkpoint = {
                "schema": CHECKPOINT_SCHEMA,
                "schema_version": 1,
                "arm": args.arm,
                "epoch": epoch,
                "window": window,
                "model_state_dict": model.state_dict(),
                "model_state_sha256": state_dict_sha256(model.state_dict()),
                "initial_state_sha256": initial_state_sha,
                "optimizer_state_dict": optimizer.state_dict(),
                "inner": inner,
                "method": {"name": args.arm, "rho": RHO if args.arm == "gcsam_candidate" else None},
                "documents": {"idea_sha256": IDEA_SHA256, "prereg_sha256": PREREG_SHA256},
                "access": {"train_only_inner": True, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False},
            }
            torch.save(checkpoint, checkpoint_path)
        row = {
            "epoch": epoch,
            "window": window,
            "learning_rate": lr,
            "train_loss": float(np.mean(losses)),
            "train_supervised_loss": float(np.mean(supervised_losses)),
            "train_kd_loss": float(np.mean(kd_losses)),
            "train_dice": float(np.mean(train_dice)),
            "optimizer_steps": len(losses),
            "consumed_draws": len(losses) * args.batch_size,
            "sampler_index_sha256": sampler_sha,
            "first_batch_sha256": first_batch_sha,
            "first_batch_loss": first_batch_loss,
            "mean_perturbation_norm": float(np.mean(perturbation_norms)) if perturbation_norms else None,
            "inner": inner,
            "strict_new_high": new_high,
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        write_json_atomic(output_dir / "progress.json", {"history": history, "best": best})
        print(
            f"arm={args.arm} epoch={epoch} loss={row['train_loss']:.6f} "
            f"inner={inner['locator_dsc_mean']:.6f}/{inner['locator_subject_dsc_mean']:.6f}/"
            f"{inner['automatic_point_hit_rate']:.6f} best={best[window]['scan']:.6f}@{best[window]['epoch']}",
            flush=True,
        )

    selections: dict[str, Any] = {}
    for window, value in best.items():
        if int(value["epoch"]) == 0:
            continue
        path = output_dir / f"{window}_best.pth"
        selections[window] = {**value, "checkpoint": {"path": str(path), "sha256": sha256_file(path)}}
    elapsed = time.time() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    expected_epochs = 1 if args.smoke else 200
    gates = {
        "epoch_count_exact": len(history) == expected_epochs,
        "all_steps_exact": all(row["optimizer_steps"] == expected_steps for row in history),
        "all_losses_finite": all(
            np.isfinite(row[key]) for row in history
            for key in ("train_loss", "train_supervised_loss", "train_kd_loss", "train_dice")
        ),
        "both_windows_saved": args.smoke or set(selections) == {"pre100", "post100"},
        "gcsam_rho_exact": args.arm == "adamw_control" or all(
            abs(float(row["mean_perturbation_norm"]) - RHO) <= 2e-6 for row in history
        ),
        "peak_at_most_24gib": peak <= MAX_PEAK_BYTES,
        "elapsed_within_arm_limit": elapsed <= MAX_SECONDS[args.arm],
    }
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "arm": args.arm,
        "all_gates_passed": all(gates.values()),
        "gates": gates,
        "initial_state_sha256": initial_state_sha,
        "sampling_weight_sha256": weight_sha,
        "selections": selections,
        "history": history,
        "data_roles": roles,
        "parameter_count": PARAMETER_COUNT,
        "method": {
            "optimizer": "AdamW" if args.arm == "adamw_control" else "paper-form GCSAM-wrapped AdamW",
            "rho": None if args.arm == "adamw_control" else RHO,
            "adaptive": False,
            "centralization_dims": "1..N-1 for ndim>1",
            "teacher_reused_between_student_passes": args.arm == "gcsam_candidate",
            "dropout_rng_replayed_between_student_passes": args.arm == "gcsam_candidate",
        },
        "schedule": {"epochs": args.epochs, "constant_epochs": 100, "base_lr": args.base_lr, "min_lr": args.min_lr},
        "documents": {"idea_sha256": IDEA_SHA256, "prereg_sha256": PREREG_SHA256, "split_sha256": SPLIT_SHA256},
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": peak,
        "access": {"train_only_inner": True, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False},
    }
    write_json_atomic(output_dir / "TRAIN_DONE.json", report)
    if not report["all_gates_passed"]:
        raise RuntimeError("GCSAM inner engineering gates failed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    record = ROOT / "archive/research_records/2026-08-31_gcsam_full972_stability"
    split = ROOT / "data/datasets/brats2024_t1ce_gcsam_inner_seed20261001_v1"
    prior = ROOT / "archive/research_records/2026-08-21_voxel_kd_data972_native_sampling"
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--fit-case-ids", type=Path, default=split / "fit_cases.jsonl")
    parser.add_argument("--inner-tune-case-ids", type=Path, default=split / "inner_tune_cases.jsonl")
    parser.add_argument("--split-manifest", type=Path, default=split / "split_manifest.json")
    parser.add_argument("--idea", type=Path, default=record / "IDEA.md")
    parser.add_argument("--prereg", type=Path, default=record / "PREREG.md")
    parser.add_argument("--native-artifact", type=Path, default=prior / "NATIVE_WT_VOLUMES.json")
    parser.add_argument("--historical-teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--ema-teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=FROZEN_SEED)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--limit-fit-cases", type=int, default=8)
    parser.add_argument("--limit-inner-tune-cases", type=int, default=8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
