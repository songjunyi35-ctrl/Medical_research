#!/usr/bin/env python3
"""Train fixed-epoch168 control/percentile-clip replications without inner access."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from monai.losses import DiceCELoss
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_t1ce_wt_gcsam_inner as base  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_percentile_clip_replication_train"
CHECKPOINT_SCHEMA = SCHEMA + ".checkpoint"
IDEA_SHA256 = "7504085184ea0e17d97268e9233ff4eb857f9f80c1b49e301aa1534416a24b0a"
PREREG_SHA256 = "cf4706a98e6053799f698d83dcd59efbfd87d7ff3880b6bea09624b4ed99e031"
SOURCE_SHA256 = "6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65"
CONTROL_CACHE_INDEX_SHA256 = "7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd"
CONTROL_LABELS_SHA256 = "bf470d48a5f115f09407ad351bb6dee11ee36bdd0c2a4285ae0bd9cc22125724"
NATIVE_ARTIFACT_SHA256 = "32be09856d71e78f512792c8bcf8f5f9ee137e6ba1e4358862ff2a61ba828c31"
REPLICATES = {"a": 20261112, "b": 20261114}
ARMS = ("control", "percentile_clip")
FORMAL_EPOCHS = 168
SAMPLER_DRAWS = 972
CONSUMED_DRAWS = 968
STEPS_PER_EPOCH = 121
PARAMETER_COUNT = 4_700_897
MAX_PEAK_BYTES = 24 * 1024**3
MAX_SECONDS = 6 * 60 * 60
FORBIDDEN_TOKENS = ("validation", "test270", "external75", "confirmation", "sealed")


class NativeFitDataset(base.NativeFitDataset):
    pass


def assert_train_only_path(path: Path) -> None:
    lowered = str(path.resolve()).lower()
    if any(token in lowered for token in FORBIDDEN_TOKENS):
        raise ValueError(f"forbidden evaluation-role token: {path}")


def _rng_bytes(device: torch.device) -> bytes:
    cpu = torch.random.get_rng_state().numpy().tobytes()
    cuda = torch.cuda.get_rng_state(device).cpu().numpy().tobytes()
    return cpu + cuda


def _cache_contract(cache: Path, arm: str, expected_index_sha256: str) -> dict[str, Any]:
    if base.sha256_file(cache / "index.json") != expected_index_sha256:
        raise ValueError("input cache index SHA-256 mismatch")
    index = json.loads((cache / "index.json").read_text())
    if index.get("count") != 972 or index.get("split") != "train" or index.get("test_access") is not False:
        raise ValueError("input cache role/count mismatch")
    if index.get("manifest_sha256") != SOURCE_SHA256:
        raise ValueError("input cache source manifest mismatch")
    if index.get("labels_sha256") != CONTROL_LABELS_SHA256:
        raise ValueError("input cache labels are not canonical")
    preprocessing = index.get("preprocessing", {})
    if preprocessing.get("spatial") != "braincrop_resize" or preprocessing.get("uses_gt_for_localization") is not False:
        raise ValueError("input cache geometry contract mismatch")
    if arm == "control":
        if expected_index_sha256 != CONTROL_CACHE_INDEX_SHA256:
            raise ValueError("control cache index is not frozen canonical cache")
    else:
        if preprocessing.get("low_percentile") != 1.0 or preprocessing.get("high_percentile") != 99.0:
            raise ValueError("candidate percentile contract mismatch")
        if preprocessing.get("percentile_method") != "numpy.percentile linear":
            raise ValueError("candidate percentile method mismatch")
        if index.get("labels_byte_exact_to_control") is not True:
            raise ValueError("candidate labels exactness gate failed")
    return index


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.replicate not in REPLICATES or args.arm not in ARMS:
        raise ValueError("unknown replicate or arm")
    if args.seed != REPLICATES[args.replicate] or args.batch_size != 8:
        raise ValueError("seed/batch differ from preregistration")
    if not args.smoke and args.epochs != FORMAL_EPOCHS:
        raise ValueError("formal training must stop at fixed epoch168")
    for path in (args.cache, args.fit_case_ids, args.split_manifest, args.idea, args.prereg, args.output_dir):
        assert_train_only_path(path)
    for path, expected in (
        (args.idea, IDEA_SHA256),
        (args.prereg, PREREG_SHA256),
        (args.native_artifact, NATIVE_ARTIFACT_SHA256),
    ):
        if base.sha256_file(path) != expected:
            raise ValueError(f"frozen input SHA-256 mismatch: {path}")
    split = json.loads(args.split_manifest.read_text())
    if split.get("schema") != "aflow.brats2024.t1ce_wt_percentile_clip_replication_splits":
        raise ValueError("split schema mismatch")
    if split.get("documents") != {"idea_sha256": IDEA_SHA256, "prereg_sha256": PREREG_SHA256}:
        raise ValueError("split document binding mismatch")
    if split.get("access") != {"train_only_inner_metrics": False, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False, "ucsd_external75": False}:
        raise ValueError("split access boundary changed")
    record = split["replications"][args.replicate]
    if record.get("train_seed") != args.seed or base.sha256_file(args.fit_case_ids) != record["fit"]["sha256"]:
        raise ValueError("replicate fit manifest/seed mismatch")
    cache_index = _cache_contract(args.cache.resolve(), args.arm, args.expected_cache_index_sha256)
    if base.sha256_file(args.historical_teacher_checkpoint) != base.HISTORICAL_SHA256:
        raise ValueError("historical teacher SHA-256 mismatch")
    if base.sha256_file(args.ema_teacher_checkpoint) != base.EMA168_SHA256:
        raise ValueError("EMA teacher SHA-256 mismatch")
    artifact = json.loads(args.native_artifact.read_text())
    if artifact.get("status") != "passed" or artifact.get("audit", {}).get("all_gates_passed") is not True:
        raise ValueError("native-volume artifact gate failed")
    native_volumes = {str(key): int(value) for key, value in artifact["native_wt_voxels"].items()}

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal training requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    fit_ids = base.read_case_ids(args.fit_case_ids)
    fit_data = NativeFitDataset(
        args.cache,
        expected_split="train",
        case_ids=fit_ids,
        limit=args.limit_fit_cases if args.smoke else None,
        native_volumes=native_volumes,
    )
    fit_subjects = {base.subject_id(case_id) for case_id in fit_data.case_ids}
    if not args.smoke and (len(fit_data) != 864 or len(fit_subjects) != int(record["fit"]["subjects"])):
        raise ValueError("formal fit case/subject count mismatch")
    weights = fit_data.sampling_weights()
    weight_sha256 = base.tensor_sha256(weights)
    draws = args.smoke_steps * args.batch_size if args.smoke else SAMPLER_DRAWS
    sampler = base.AuditedWeightedSampler(weights, draws, args.seed)
    loader = DataLoader(
        fit_data,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    expected_steps = args.smoke_steps if args.smoke else STEPS_PER_EPOCH
    if len(loader) != expected_steps:
        raise RuntimeError("optimizer step count mismatch")

    history_payload = torch.load(args.historical_teacher_checkpoint, map_location="cpu", weights_only=False)
    ema_payload = torch.load(args.ema_teacher_checkpoint, map_location="cpu", weights_only=False)
    historical_teacher = base._model(history_payload, device).eval().requires_grad_(False)
    ema_teacher = base._model(ema_payload, device).eval().requires_grad_(False)
    model = base.AutomaticPromptLocator(center_aware=False, init_filters=16, dropout=0.1).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise ValueError("student parameter count mismatch")
    initial_state_sha256 = base.state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay, foreach=None, fused=None)
    loss_function = DiceCELoss(sigmoid=True, squared_pred=False, lambda_dice=1.0, lambda_ce=0.1)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    roles = {
        "fit": {"path": str(args.fit_case_ids.resolve()), "sha256": record["fit"]["sha256"], "cases": len(fit_data), "subjects": len(fit_subjects)},
        "input_cache": {"path": str(args.cache.resolve()), "index_sha256": args.expected_cache_index_sha256, "images_sha256": cache_index["images_sha256"], "labels_sha256": cache_index["labels_sha256"]},
        "inner_dataset_constructed": False,
        "inner_metrics_accessed": False,
    }
    access = {"fit": True, "inner_metrics": False, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False, "ucsd_external75": False}
    base.write_json_atomic(output / "STARTED.json", {
        "schema": SCHEMA + ".started", "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started_before_optimizer_step", "replicate": args.replicate,
        "arm": args.arm, "fixed_stop_epoch": FORMAL_EPOCHS,
        "data_roles": roles, "runner_sha256": base.sha256_file(Path(__file__).resolve()),
        "access": access,
    })

    history: list[dict[str, Any]] = []
    started = time.time()
    epochs = 1 if args.smoke else args.epochs
    for epoch in range(1, epochs + 1):
        lr = base.learning_rate_for_epoch(epoch, args.base_lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        losses: list[float] = []
        supervised_losses: list[float] = []
        kd_losses: list[float] = []
        train_dice: list[float] = []
        rng_before = hashlib.sha256()
        rng_after = hashlib.sha256()
        first_batch_sha256: dict[str, str] | None = None
        first_batch_loss: float | None = None
        for step, batch in enumerate(loader):
            image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            rng_before.update(_rng_bytes(device))
            image, label = base.augment_batch(image, label, max_shift=8)
            rng_after.update(_rng_bytes(device))
            if epoch == 1 and step == 0:
                first_batch_sha256 = {"image": base.tensor_sha256(image), "label": base.tensor_sha256(label)}
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                history_logits = historical_teacher(image).segmentation_logits.float()
                ema_logits = ema_teacher(image).segmentation_logits.float()
                teacher_logits = 0.75 * history_logits + 0.25 * ema_logits
            optimizer.zero_grad(set_to_none=True)
            loss, supervised, kd, logits = base.student_loss(model, image, label, teacher_logits, loss_function)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite loss")
            if epoch == 1 and step == 0:
                first_batch_loss = float(loss.detach())
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
            if float(grad_norm) <= 0:
                raise RuntimeError("non-positive gradient")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if any(parameter.grad is not None for teacher in (historical_teacher, ema_teacher) for parameter in teacher.parameters()):
                raise RuntimeError("teacher received gradient")
            losses.append(float(loss.detach()))
            supervised_losses.append(float(supervised.detach()))
            kd_losses.append(float(kd.detach()))
            train_dice.extend(base.dice_per_case(logits.detach().float(), label).cpu().tolist())
        if sampler.last_indices is None:
            raise RuntimeError("sampler indices missing")
        history.append({
            "epoch": epoch, "learning_rate": lr,
            "train_loss": float(np.mean(losses)),
            "train_supervised_loss": float(np.mean(supervised_losses)),
            "train_kd_loss": float(np.mean(kd_losses)),
            "train_dice": float(np.mean(train_dice)),
            "optimizer_steps": len(losses),
            "consumed_draws": len(losses) * args.batch_size,
            "sampler_index_sha256": base.index_sha256(sampler.last_indices[:expected_steps * args.batch_size]),
            "augmentation_rng_before_sha256": rng_before.hexdigest(),
            "augmentation_rng_after_sha256": rng_after.hexdigest(),
            "first_batch_sha256": first_batch_sha256,
            "first_batch_loss": first_batch_loss,
            "elapsed_seconds": time.time() - started,
            "inner_metrics": None,
        })
        base.write_json_atomic(output / "progress.json", {"history": history, "fixed_stop_epoch": FORMAL_EPOCHS, "inner_metrics_accessed": False})
        print(f"rep={args.replicate} arm={args.arm} epoch={epoch} loss={history[-1]['train_loss']:.6f} train={history[-1]['train_dice']:.6f}", flush=True)

    epoch = 1 if args.smoke else FORMAL_EPOCHS
    model_state_sha256 = base.state_dict_sha256(model.state_dict())
    checkpoint_path = output / ("smoke_epoch1.pth" if args.smoke else "fixed_epoch168.pth")
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA, "schema_version": 1,
        "replicate": args.replicate, "arm": args.arm, "epoch": epoch,
        "model_state_dict": model.state_dict(), "model_state_sha256": model_state_sha256,
        "initial_state_sha256": initial_state_sha256,
        "architecture": {"class": "AutomaticPromptLocator", "init_filters": 16, "dropout": 0.1, "center_aware": False},
        "method": {"source_percentile_clip": None if args.arm == "control" else [1.0, 99.0], "cache_index_sha256": args.expected_cache_index_sha256},
        "documents": {"idea_sha256": IDEA_SHA256, "prereg_sha256": PREREG_SHA256, "split_manifest_sha256": base.sha256_file(args.split_manifest)},
        "access": access,
    }
    torch.save(checkpoint, checkpoint_path)
    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reload_exact = reloaded["model_state_sha256"] == base.state_dict_sha256(reloaded["model_state_dict"]) == model_state_sha256
    elapsed = time.time() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    gates = {
        "cache_contract": True,
        "epoch_count_exact": len(history) == epochs,
        "all_steps_exact": all(row["optimizer_steps"] == expected_steps for row in history),
        "all_consumed_draws_exact": all(row["consumed_draws"] == expected_steps * args.batch_size for row in history),
        "all_losses_finite": all(np.isfinite(row[key]) for row in history for key in ("train_loss", "train_supervised_loss", "train_kd_loss", "train_dice")),
        "no_inner_metrics": all(row["inner_metrics"] is None for row in history),
        "checkpoint_reload_exact": reload_exact,
        "peak_at_most_24gib": peak <= MAX_PEAK_BYTES,
        "elapsed_within_arm_limit": args.smoke or elapsed <= MAX_SECONDS,
    }
    report = {
        "schema": SCHEMA, "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete", "replicate": args.replicate, "arm": args.arm,
        "fixed_stop_epoch": epoch, "all_gates_passed": all(gates.values()), "gates": gates,
        "initial_state_sha256": initial_state_sha256,
        "sampling_weight_sha256": weight_sha256,
        "first_batch_sha256": history[0]["first_batch_sha256"],
        "first_batch_loss": history[0]["first_batch_loss"],
        "checkpoint": {"path": str(checkpoint_path), "sha256": base.sha256_file(checkpoint_path), "model_state_sha256": model_state_sha256},
        "history": history, "data_roles": roles, "parameter_count": PARAMETER_COUNT,
        "runner_sha256": base.sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": elapsed, "peak_gpu_memory_bytes": peak, "access": access,
    }
    base.write_json_atomic(output / "TRAIN_DONE.json", report)
    if not report["all_gates_passed"]:
        raise RuntimeError("training engineering gate failed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    record = ROOT / "archive/research_records/2026-09-03_t1ce_percentile_clip_stability"
    parser.add_argument("--replicate", choices=tuple(REPLICATES), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--expected-cache-index-sha256", required=True)
    parser.add_argument("--fit-case-ids", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--idea", type=Path, default=record / "IDEA.md")
    parser.add_argument("--prereg", type=Path, default=record / "PREREG.md")
    parser.add_argument("--native-artifact", type=Path, default=ROOT / "archive/research_records/2026-08-21_voxel_kd_data972_native_sampling/NATIVE_WT_VOLUMES.json")
    parser.add_argument("--historical-teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--ema-teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--limit-fit-cases", type=int, default=8)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
