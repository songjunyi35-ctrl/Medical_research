"""Train one conditional Full972 teacher/control/candidate; no evaluation loader."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from monai.losses import DiceCELoss
from torch import nn
from torch.utils.data import DataLoader

from scripts import n4_family_full972_protocol as p
from scripts import train_t1ce_wt_gcsam_inner as base

SCHEMA = "aflow.n4_family_full972.train"


@torch.no_grad()
def update_ema(shadow: dict[str, torch.Tensor], model: nn.Module) -> None:
    for key, value in model.state_dict().items():
        if value.is_floating_point():
            shadow[key].mul_(0.999).add_(value.detach(), alpha=0.001)
        else:
            shadow[key].copy_(value.detach())


def rng_bytes(device: torch.device) -> bytes:
    return (
        torch.random.get_rng_state().numpy().tobytes()
        + torch.cuda.get_rng_state(device).cpu().numpy().tobytes()
    )


def verify_teacher(payload: dict, report: dict, role: str, smoke: bool) -> None:
    if payload.get("schema") != SCHEMA + ".checkpoint" or payload.get("role") != role:
        raise ValueError("wrong Full972 teacher identity")
    if bool(payload.get("smoke")) != smoke or payload.get("fit_sha256") != p.SOURCE_SHA:
        raise ValueError("Full972 teacher ownership/smoke mismatch")
    if payload.get("input_cache_index_sha256") != p.CONTROL_INDEX_SHA:
        raise ValueError("Full972 teacher did not use canonical N4")
    if payload.get("stage_a_inner_metrics_used_for_training") is not False:
        raise ValueError("teacher used Stage A metrics during optimization")
    if not report.get("all_gates_passed"):
        raise ValueError("teacher training gates failed")
    if not smoke and payload.get("epoch") != 168:
        raise ValueError("teacher is not fixed epoch168")


def run(args: argparse.Namespace) -> dict:
    p.docs()
    p.checked(p.SOURCE, p.SOURCE_SHA)
    p.checked(p.NATIVE, p.NATIVE_SHA)
    from scripts.freeze_n4_family_full972 import check_implementation
    check_implementation()
    # The eight-hour condition is a stage-start gate checked once by the supervisor.
    # Children may legitimately finish after less than eight hours remain.
    eligible, admission = p.verify_admission(require_time=False)
    if args.role in p.CANDIDATES and args.role not in eligible:
        raise RuntimeError(f"{args.role} did not pass Stage A")
    if not args.smoke:
        smoke = json.loads((p.RECORD / "FULL972_GPU_SMOKE_RECEIPT.json").read_text())
        if smoke.get("status") != "passed" or smoke.get("eligible_candidates") != eligible:
            raise ValueError("Full972 formal launch requires matching smoke gates")

    cache, cache_sha = p.verify_cache(args.role)
    rows = p.read_rows(p.SOURCE)
    case_ids = [row["case_id"] for row in rows]
    subjects = {row["subject_id"] for row in rows}
    if len(case_ids) != 972 or len(set(case_ids)) != 972 or len(subjects) != 444:
        raise ValueError("Full972 source cardinality mismatch")

    seed = p.SEEDS[args.role]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Full972 training requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(1)

    native = json.loads(p.NATIVE.read_text())
    volumes = {str(key): int(value) for key, value in native["native_wt_voxels"].items()}
    data = base.NativeFitDataset(
        cache,
        expected_split="train",
        case_ids=case_ids,
        limit=8 if args.smoke else None,
        native_volumes=volumes,
    )
    if not args.smoke and len(data) != 972:
        raise ValueError("Full972 dataset length mismatch")
    weights = data.sampling_weights()
    weight_sha = base.tensor_sha256(weights)
    sampler = base.AuditedWeightedSampler(weights, 16 if args.smoke else p.DRAWS, seed)
    loader = DataLoader(
        data,
        batch_size=8,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=True,
    )
    if len(loader) != (2 if args.smoke else p.STEPS):
        raise ValueError("Full972 step count mismatch")

    teacher_role = args.role in ("history", "ema")
    teachers = []
    teacher_info = []
    if not teacher_role:
        for role in ("history", "ema"):
            directory = p.WORK / ("smoke" if args.smoke else "formal") / role
            report = json.loads((directory / "TRAIN_DONE.json").read_text())
            path = Path(report["checkpoint"]["path"])
            p.checked(path, report["checkpoint"]["sha256"])
            payload = torch.load(path, map_location="cpu", weights_only=False)
            verify_teacher(payload, report, role, args.smoke)
            teacher = base.AutomaticPromptLocator(
                center_aware=False, init_filters=16, dropout=0.1
            ).to(device)
            teacher.load_state_dict(payload["model_state_dict"], strict=True)
            teacher.eval().requires_grad_(False)
            teachers.append(teacher)
            teacher_info.append({"role": role, "path": str(path), "sha256": p.sha(path)})

    model = base.AutomaticPromptLocator(
        center_aware=False, init_filters=16, dropout=0.1
    ).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != p.PARAMETERS:
        raise ValueError("Full972 model parameter count mismatch")
    initial_sha = base.state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-4, weight_decay=1e-5, foreach=None, fused=None
    )
    loss_fn = DiceCELoss(
        sigmoid=True, squared_pred=False, lambda_dice=1.0, lambda_ce=0.1
    )
    phase = "smoke" if args.smoke else "formal"
    output = p.WORK / phase / args.role
    output.mkdir(parents=True, exist_ok=False)
    access = {
        "full972_train": True,
        "stage_a_inner_decision_receipt": True,
        "stage_a_inner_arrays_loaded_by_trainer": False,
        "new_validation": False,
        "fixed_validation": False,
        "outer": False,
        "confirmation": False,
        "test270": False,
        "ucsd75": False,
        "rhuh": False,
    }
    p.write_new(
        output / "STARTED.json",
        {
            "schema": SCHEMA + ".started",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "role": args.role,
            "smoke": args.smoke,
            "seed": seed,
            "fit_sha256": p.SOURCE_SHA,
            "cache_index_sha256": cache_sha,
            "initial_state_sha256": initial_sha,
            "teacher_checkpoints": teacher_info,
            "eligible_candidates": eligible,
            "admission_sha256": p.sha(p.RECORD / "FULL972_ADMISSION_CHECK.json"),
            "access": access,
            "runner_sha256": p.sha(Path(__file__)),
        },
    )

    history = []
    shadow = None
    ema_updates = 0
    start = time.monotonic()
    epochs = 1 if args.smoke else p.EPOCHS
    for epoch in range(1, epochs + 1):
        lr = base.learning_rate_for_epoch(epoch, 2e-4, 1e-5)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        losses, supervised_losses, kd_losses, dices = [], [], [], []
        before, after, first = hashlib.sha256(), hashlib.sha256(), None
        for step, batch in enumerate(loader):
            image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            before.update(rng_bytes(device))
            image, label = base.augment_batch(image, label, max_shift=8)
            after.update(rng_bytes(device))
            if epoch == 1 and step == 0:
                first = {"image": base.tensor_sha256(image), "label": base.tensor_sha256(label)}
            optimizer.zero_grad(set_to_none=True)
            if teacher_role:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(image).segmentation_logits
                    supervised = loss_fn(logits, label)
                kd = torch.zeros((), device=device)
                loss = supervised
            else:
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    target = (
                        0.75 * teachers[0](image).segmentation_logits.float()
                        + 0.25 * teachers[1](image).segmentation_logits.float()
                    )
                loss, supervised, kd, logits = base.student_loss(
                    model, image, label, target, loss_fn
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite Full972 loss")
            if first is not None and step == 0:
                first["loss"] = float(loss.detach())
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            if float(norm) <= 0:
                raise RuntimeError("zero Full972 gradient")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if args.role == "ema" and shadow is not None:
                update_ema(shadow, model)
                ema_updates += 1
            if args.smoke and args.role == "ema" and shadow is None:
                shadow = {key: value.detach().clone() for key, value in model.state_dict().items()}
            if any(parameter.grad is not None for teacher in teachers for parameter in teacher.parameters()):
                raise RuntimeError("Full972 teacher gradient leak")
            losses.append(float(loss.detach()))
            supervised_losses.append(float(supervised.detach()))
            kd_losses.append(float(kd.detach()))
            dices.extend(base.dice_per_case(logits.detach().float(), label).cpu().tolist())
        if args.role == "ema" and epoch == 100:
            shadow = {key: value.detach().clone() for key, value in model.state_dict().items()}
        item = {
            "epoch": epoch,
            "lr": lr,
            "loss": float(np.mean(losses)),
            "supervised": float(np.mean(supervised_losses)),
            "kd": float(np.mean(kd_losses)),
            "train_dice": float(np.mean(dices)),
            "steps": len(losses),
            "draws": len(losses) * 8,
            "sampler_sha256": base.index_sha256(sampler.last_indices[: len(losses) * 8]),
            "augmentation_rng_before_sha256": before.hexdigest(),
            "augmentation_rng_after_sha256": after.hexdigest(),
            "first_batch": first,
            "ema_updates": ema_updates,
            "elapsed_seconds": time.monotonic() - start,
            "stage_a_inner_metrics_used_for_training": False,
        }
        history.append(item)
        base.write_json_atomic(
            output / "progress.json",
            {"history": history, "stage_a_inner_metrics_used_for_training": False},
        )
        print(
            f"full972/{args.role} epoch={epoch} loss={item['loss']:.6f} "
            f"fit_dice={item['train_dice']:.6f}",
            flush=True,
        )
        if not args.smoke and time.monotonic() - start > 6 * 3600:
            raise RuntimeError("six-hour Full972 arm time gate")
        if torch.cuda.max_memory_allocated(device) > 24 * 1024**3:
            raise RuntimeError("24GiB Full972 peak gate")

    selected_state = shadow if args.role == "ema" else model.state_dict()
    if selected_state is None:
        raise RuntimeError("Full972 EMA state was not initialized")
    state_sha = base.state_dict_sha256(selected_state)
    checkpoint = {
        "schema": SCHEMA + ".checkpoint",
        "role": args.role,
        "seed": seed,
        "epoch": epochs,
        "smoke": args.smoke,
        "fit_sha256": p.SOURCE_SHA,
        "input_cache_index_sha256": cache_sha,
        "model_state_dict": selected_state,
        "state_sha256": state_sha,
        "initial_state_sha256": initial_sha,
        "teacher_checkpoints": teacher_info,
        "stage_a_inner_metrics_used_for_training": False,
        "ema_updates": ema_updates,
        "architecture": {"init_filters": 16, "dropout": 0.1, "center_aware": False},
        "admission_sha256": p.sha(p.RECORD / "FULL972_ADMISSION_CHECK.json"),
    }
    checkpoint_path = output / "fixed.pth"
    torch.save(checkpoint, checkpoint_path)
    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    gates = {
        "fixed_epochs": len(history) == epochs,
        "fixed_steps": all(item["steps"] == len(loader) for item in history),
        "reload_exact": base.state_dict_sha256(reloaded["model_state_dict"]) == state_sha,
        "ema_updates_exact": (
            ema_updates == (1 if args.smoke else 8228)
            if args.role == "ema"
            else ema_updates == 0
        ),
        "finite": all(
            np.isfinite(item[key])
            for item in history
            for key in ("loss", "supervised", "kd", "train_dice")
        ),
        "no_new_evaluation": all(
            item["stage_a_inner_metrics_used_for_training"] is False for item in history
        ),
        "peak": torch.cuda.max_memory_allocated(device) <= 24 * 1024**3,
    }
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "role": args.role,
        "smoke": args.smoke,
        "seed": seed,
        "fit_sha256": p.SOURCE_SHA,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": p.sha(checkpoint_path),
            "state_sha256": state_sha,
        },
        "initial_state_sha256": initial_sha,
        "sampling_weight_sha256": weight_sha,
        "history": history,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "teacher_checkpoints": teacher_info,
        "cache_index_sha256": cache_sha,
        "elapsed_seconds": time.monotonic() - start,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "access": access,
        "eligible_candidates": eligible,
        "admission_created_utc": admission["created_utc"],
        "runner_sha256": p.sha(Path(__file__)),
    }
    p.write_new(output / "TRAIN_DONE.json", report)
    if not report["all_gates_passed"]:
        raise RuntimeError("Full972 training engineering gates failed")
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=p.ROLES, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
