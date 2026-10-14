#!/usr/bin/env python3
"""Matched plain SAGE versus latent-domain conditioned Domain-SAGE training."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from monai.losses import DiceCELoss

from sammed3d.t1ce_wt_domain_sage import DomainSAGESafeAdapter
from sammed3d.t1ce_wt_safe_adapter import FrozenBaseSafeAdapter
from scripts.train_sammed3d_t1ce_wt_residual_refiner import (
    RefinerDataset,
    ResidualRefiner,
    augment_batch,
    dice_per_case,
    sha256_file,
    write_json_atomic,
)
from scripts.train_t1ce_wt_residual_refiner_graymean_robust import corrupt_coarse_batch
from scripts.train_t1ce_wt_safe_adapter import (
    SOURCE_CHECKPOINT_SCHEMA,
    serializable_args,
    subject_macro,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_domain_sage_training"
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_domain_sage_checkpoint"
REGISTRY_SCHEMA = "aflow.sammed3d.t1ce_wt_domain_sage_registry"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RegistrySplit:
    descriptors: np.ndarray
    domain_ids: np.ndarray
    row_by_case_id: dict[str, int]

    def batch(
        self, case_ids: list[str] | tuple[str, ...], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = [self.row_by_case_id[case_id] for case_id in case_ids]
        descriptors = torch.from_numpy(self.descriptors[rows]).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        domains = torch.from_numpy(self.domain_ids[rows]).to(
            device=device, dtype=torch.long, non_blocking=True
        )
        return descriptors, domains


@dataclass(frozen=True)
class DomainRegistry:
    train: RegistrySplit
    validation: RegistrySplit
    descriptor_dim: int
    domain_count: int
    manifest: dict[str, Any]
    manifest_path: Path
    npz_path: Path


def load_registry(path: Path) -> DomainRegistry:
    root = path.resolve()
    manifest_path = root / "registry.json"
    npz_path = root / "registry.npz"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != REGISTRY_SCHEMA or manifest.get("test_access") is not False:
        raise ValueError("invalid or test-accessible Domain-SAGE registry")
    if manifest["registry_npz"]["sha256"] != sha256_file(npz_path):
        raise RuntimeError("Domain-SAGE registry hash mismatch")
    with np.load(npz_path, allow_pickle=False) as payload:
        train_ids = payload["train_case_ids"].astype(str).tolist()
        validation_ids = payload["validation_case_ids"].astype(str).tolist()
        train_descriptors = payload["train_descriptors"].astype(np.float32)
        validation_descriptors = payload["validation_descriptors"].astype(np.float32)
        train_domains = payload["train_domain_ids"].astype(np.int64)
        validation_domains = payload["validation_domain_ids"].astype(np.int64)
        centroids = payload["centroids"].astype(np.float32)
    if len(set(train_ids)) != len(train_ids) or len(set(validation_ids)) != len(validation_ids):
        raise ValueError("registry case IDs are not unique")
    if train_descriptors.shape[0] != len(train_ids):
        raise ValueError("training registry rows do not match case IDs")
    if validation_descriptors.shape[0] != len(validation_ids):
        raise ValueError("validation registry rows do not match case IDs")
    if train_descriptors.shape[1] != validation_descriptors.shape[1]:
        raise ValueError("registry descriptor dimensions differ by split")
    if not np.isfinite(train_descriptors).all() or not np.isfinite(validation_descriptors).all():
        raise ValueError("registry contains non-finite descriptors")
    domain_count = int(centroids.shape[0])
    return DomainRegistry(
        train=RegistrySplit(
            train_descriptors,
            train_domains,
            {case_id: index for index, case_id in enumerate(train_ids)},
        ),
        validation=RegistrySplit(
            validation_descriptors,
            validation_domains,
            {case_id: index for index, case_id in enumerate(validation_ids)},
        ),
        descriptor_dim=int(train_descriptors.shape[1]),
        domain_count=domain_count,
        manifest=manifest,
        manifest_path=manifest_path,
        npz_path=npz_path,
    )


def _forward(
    model: FrozenBaseSafeAdapter | DomainSAGESafeAdapter,
    *,
    arm: str,
    image: torch.Tensor,
    coarse: torch.Tensor,
    descriptor: torch.Tensor | None,
) -> dict[str, torch.Tensor | None]:
    if arm == "plain":
        output = model.forward_with_aux(image, coarse)  # type: ignore[call-arg]
        return {
            "logits": output.logits,
            "base_logits": output.base_logits,
            "residual": output.residual,
            "domain_logits": None,
            "domain_affinities": None,
            "film_abs_mean": torch.zeros(image.shape[0], device=image.device),
        }
    if descriptor is None:
        raise RuntimeError("Domain-SAGE requires a descriptor")
    output = model.forward_with_aux(image, coarse, descriptor)  # type: ignore[call-arg]
    return {
        "logits": output.logits,
        "base_logits": output.base_logits,
        "residual": output.residual,
        "domain_logits": output.domain_logits,
        "domain_affinities": output.domain_affinities,
        "film_abs_mean": output.film_abs_mean,
    }


@torch.inference_mode()
def validate(
    model: FrozenBaseSafeAdapter | DomainSAGESafeAdapter,
    loader: DataLoader,
    device: torch.device,
    *,
    arm: str,
    registry: DomainRegistry | None,
) -> dict[str, Any]:
    model.eval()
    final_values: list[float] = []
    base_values: list[float] = []
    coarse_values: list[float] = []
    residual_magnitudes: list[float] = []
    film_magnitudes: list[float] = []
    domain_targets: list[int] = []
    domain_predictions: list[int] = []
    domain_entropies: list[float] = []
    case_ids: list[str] = []
    for batch in loader:
        ids = list(batch["case_id"])
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        coarse = batch["coarse"].to(device=device, dtype=torch.float32, non_blocking=True)
        label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
        descriptor: torch.Tensor | None = None
        domains: torch.Tensor | None = None
        if arm == "domain":
            assert registry is not None
            descriptor, domains = registry.validation.batch(ids, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = _forward(
                model, arm=arm, image=image, coarse=coarse, descriptor=descriptor
            )
        logits = output["logits"].float()  # type: ignore[union-attr]
        base_logits = output["base_logits"].float()  # type: ignore[union-attr]
        residual = output["residual"].float()  # type: ignore[union-attr]
        final_values.extend(dice_per_case(logits, label).cpu().tolist())
        base_values.extend(dice_per_case(base_logits, label).cpu().tolist())
        coarse_values.extend(dice_per_case(coarse, label).cpu().tolist())
        residual_magnitudes.extend(residual.abs().flatten(1).mean(1).cpu().tolist())
        film_magnitudes.extend(output["film_abs_mean"].float().cpu().tolist())  # type: ignore[union-attr]
        if arm == "domain":
            domain_logits = output["domain_logits"].float()  # type: ignore[union-attr]
            affinities = output["domain_affinities"].float()  # type: ignore[union-attr]
            assert domains is not None
            domain_targets.extend(domains.cpu().tolist())
            domain_predictions.extend(domain_logits.argmax(1).cpu().tolist())
            entropy = -(affinities * affinities.clamp_min(1e-8).log()).sum(1)
            domain_entropies.extend(entropy.cpu().tolist())
        case_ids.extend(ids)
    final_array = np.asarray(final_values)
    base_array = np.asarray(base_values)
    result: dict[str, Any] = {
        "cases": len(case_ids),
        "subjects": len({case_id.rsplit("-", 1)[0] for case_id in case_ids}),
        "coarse_dsc_mean": float(np.mean(coarse_values)),
        "base_dsc_mean": float(np.mean(base_array)),
        "base_subject_dsc_mean": subject_macro(case_ids, base_values),
        "final_dsc_mean": float(np.mean(final_array)),
        "final_subject_dsc_mean": subject_macro(case_ids, final_values),
        "delta_from_base": float(np.mean(final_array - base_array)),
        "improved_vs_base_cases": int(np.sum(final_array > base_array)),
        "worsened_vs_base_cases": int(np.sum(final_array < base_array)),
        "adapter_residual_abs_mean": float(np.mean(residual_magnitudes)),
        "film_abs_mean": float(np.mean(film_magnitudes)),
    }
    if arm == "domain":
        targets = np.asarray(domain_targets)
        predictions = np.asarray(domain_predictions)
        result["domain_accuracy"] = float(np.mean(targets == predictions))
        result["domain_entropy_mean"] = float(np.mean(domain_entropies))
        result["domain_target_counts"] = np.bincount(
            targets, minlength=registry.domain_count  # type: ignore[union-attr]
        ).tolist()
        result["domain_prediction_counts"] = np.bincount(
            predictions, minlength=registry.domain_count  # type: ignore[union-attr]
        ).tolist()
    result["per_case"] = [
        {
            "case_id": case_id,
            "coarse_dsc": coarse,
            "base_dsc": base,
            "final_dsc": final,
            "delta_from_base": final - base,
            **(
                {
                    "pseudo_domain": domain_targets[index],
                    "predicted_domain": domain_predictions[index],
                }
                if arm == "domain"
                else {}
            ),
        }
        for index, (case_id, coarse, base, final) in enumerate(
            zip(case_ids, coarse_values, base_values, final_values, strict=True)
        )
    ]
    return result


def checkpoint_payload(
    *,
    model: FrozenBaseSafeAdapter | DomainSAGESafeAdapter,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation: dict[str, Any],
    args: argparse.Namespace,
    source_checkpoint: Path,
    source_sha256: str,
    registry: DomainRegistry | None,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "epoch": epoch,
        "arm": args.arm,
        "validation": validation,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "initialization": {
            "path": str(source_checkpoint),
            "sha256": source_sha256,
            "base_frozen": True,
            "adapter_zero_initialized": True,
            "domain_film_identity_initialized": args.arm == "domain",
        },
        "architecture": {
            "name": "plain SAGE-3D matched control" if args.arm == "plain" else "Domain-SAGE 3D",
            "base": "frozen robust ResidualRefiner",
            "adapter": "MONAI SegResNet with the existing five SAGE inputs",
            "adapter_init_filters": args.adapter_init_filters,
            "adapter_dropout": args.adapter_dropout,
            "domain_count": args.domain_count if args.arm == "domain" else 0,
            "descriptor_dim": registry.descriptor_dim if registry else 0,
            "domain_hidden_dim": args.domain_hidden_dim if args.arm == "domain" else 0,
            "film_rank": args.film_rank if args.arm == "domain" else 0,
            "film_sites": ["bottleneck", "decoder_32", "decoder_16", "decoder_8"]
            if args.arm == "domain"
            else [],
            "output": "frozen_base_logit + adapter_residual",
            "inference_label_input": False,
        },
        "objective": {
            "segmentation": "MONAI DiceCELoss(sigmoid=True)",
            "domain": "fit-only pseudo-domain cross entropy" if args.arm == "domain" else None,
            "domain_weight": args.domain_weight if args.arm == "domain" else 0.0,
        },
        "registry": (
            {
                "manifest_path": str(registry.manifest_path),
                "manifest_sha256": sha256_file(registry.manifest_path),
                "npz_path": str(registry.npz_path),
                "npz_sha256": sha256_file(registry.npz_path),
            }
            if registry
            else None
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.arm not in {"plain", "domain"}:
        raise ValueError("arm must be plain or domain")
    if args.arm == "domain" and args.domain_weight <= 0:
        raise ValueError("Domain-SAGE requires a positive domain loss weight")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Domain-SAGE paired training requires CUDA")
    torch.cuda.set_device(device)

    train_source = RefinerDataset(args.train_cache)
    validation_source = RefinerDataset(args.validation_cache)
    if train_source.index.get("split") != "train" or validation_source.index.get("split") != "validation":
        raise ValueError("train/validation cache split mismatch")
    if train_source.index.get("test_access") is not False or validation_source.index.get("test_access") is not False:
        raise ValueError("test-accessible cache is forbidden")
    if train_source.index["model"]["checkpoint_sha256"] != validation_source.index["model"]["checkpoint_sha256"]:
        raise ValueError("train/validation coarse checkpoint mismatch")

    registry = load_registry(args.domain_registry) if args.arm == "domain" else None
    if registry is not None:
        if registry.domain_count != args.domain_count:
            raise ValueError("registry domain count mismatch")
        if list(registry.train.row_by_case_id) != train_source.case_ids:
            raise ValueError("registry training membership/order differs from cache")
        if list(registry.validation.row_by_case_id) != validation_source.case_ids:
            raise ValueError("registry validation membership/order differs from cache")

    source_checkpoint = args.init_checkpoint.resolve()
    source_sha256 = sha256_file(source_checkpoint)
    if args.expected_init_sha256 and source_sha256 != args.expected_init_sha256:
        raise RuntimeError("robust base checkpoint hash mismatch")
    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if source.get("schema") != SOURCE_CHECKPOINT_SCHEMA:
        raise ValueError("unexpected robust base checkpoint schema")
    architecture = source.get("architecture", {})
    if int(architecture.get("init_filters", -1)) != args.base_init_filters:
        raise ValueError("base init_filters mismatch")
    if float(architecture.get("dropout", -1)) != args.base_dropout:
        raise ValueError("base dropout mismatch")

    train_limit = args.limit_train_cases or len(train_source)
    validation_limit = args.limit_validation_cases or len(validation_source)
    if not 0 < train_limit <= len(train_source) or not 0 < validation_limit <= len(validation_source):
        raise ValueError("invalid case limit")
    train_data = train_source if train_limit == len(train_source) else Subset(train_source, range(train_limit))
    validation_data = validation_source if validation_limit == len(validation_source) else Subset(validation_source, range(validation_limit))
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        train_source.sampling_weights()[:train_limit],
        num_samples=train_limit,
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
    validation_workers = max(0, args.num_workers // 2)
    validation_loader = DataLoader(
        validation_data,
        batch_size=1,
        shuffle=False,
        num_workers=validation_workers,
        pin_memory=True,
        persistent_workers=validation_workers > 0,
    )

    base = ResidualRefiner(args.base_init_filters, args.base_dropout)
    base.load_state_dict(source["model_state_dict"], strict=True)
    if args.arm == "plain":
        model: FrozenBaseSafeAdapter | DomainSAGESafeAdapter = FrozenBaseSafeAdapter(
            base,
            mode="plain",
            init_filters=args.adapter_init_filters,
            dropout=args.adapter_dropout,
        )
    else:
        assert registry is not None
        model = DomainSAGESafeAdapter(
            base,
            descriptor_dim=registry.descriptor_dim,
            domain_count=args.domain_count,
            domain_hidden_dim=args.domain_hidden_dim,
            film_rank=args.film_rank,
            init_filters=args.adapter_init_filters,
            dropout=args.adapter_dropout,
        )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    segmentation_loss_fn = DiceCELoss(
        sigmoid=True, squared_pred=False, lambda_dice=1.0, lambda_ce=1.0
    )

    started = time.time()
    baseline = validate(
        model, validation_loader, device, arm=args.arm, registry=registry
    )
    if abs(baseline["final_dsc_mean"] - baseline["base_dsc_mean"]) > 1e-9:
        raise RuntimeError("identity-initialized arm does not reproduce frozen base")
    if args.expected_baseline_dsc is not None and abs(
        baseline["base_dsc_mean"] - args.expected_baseline_dsc
    ) > args.baseline_tolerance:
        raise RuntimeError("frozen base DSC differs from preregistered anchor")
    best_value = float(baseline["final_dsc_mean"])
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    best_path = output_dir / "domain_sage_val_dice_best.pth"
    torch.save(
        checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=0,
            validation=baseline,
            args=args,
            source_checkpoint=source_checkpoint,
            source_sha256=source_sha256,
            registry=registry,
        ),
        best_path,
    )
    print(
        f"arm={args.arm} epoch=0 base={baseline['base_dsc_mean']:.6f} "
        f"final={baseline['final_dsc_mean']:.6f} subject={baseline['final_subject_dsc_mean']:.6f}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_losses: list[float] = []
        segmentation_losses: list[float] = []
        domain_losses: list[float] = []
        train_dice: list[float] = []
        domain_correct = 0
        domain_total = 0
        for batch in train_loader:
            ids = list(batch["case_id"])
            image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            coarse = batch["coarse"].to(device=device, dtype=torch.float32, non_blocking=True)
            label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            descriptor: torch.Tensor | None = None
            domains: torch.Tensor | None = None
            if args.arm == "domain":
                assert registry is not None
                descriptor, domains = registry.train.batch(ids, device)
            image, coarse, label = augment_batch(
                image, coarse, label, max_shift=args.max_shift
            )
            coarse = corrupt_coarse_batch(
                coarse,
                max_shift=args.coarse_shift_max,
                probability=args.coarse_shift_probability,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = _forward(
                    model, arm=args.arm, image=image, coarse=coarse, descriptor=descriptor
                )
                segmentation_loss = segmentation_loss_fn(output["logits"], label)
                if args.arm == "domain":
                    assert domains is not None
                    domain_loss = F.cross_entropy(output["domain_logits"], domains)
                    loss = segmentation_loss + args.domain_weight * domain_loss
                    domain_correct += int(
                        (output["domain_logits"].argmax(1) == domains).sum().detach()
                    )
                    domain_total += int(domains.numel())
                else:
                    domain_loss = segmentation_loss.new_zeros(())
                    loss = segmentation_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=5.0)
            optimizer.step()
            total_losses.append(float(loss.detach()))
            segmentation_losses.append(float(segmentation_loss.detach()))
            domain_losses.append(float(domain_loss.detach()))
            train_dice.extend(
                dice_per_case(output["logits"].detach().float(), label).cpu().tolist()
            )

        validation = validate(
            model, validation_loader, device, arm=args.arm, registry=registry
        )
        value = float(validation["final_dsc_mean"])
        strict_new_high = value > best_value
        if strict_new_high:
            best_value = value
            best_epoch = epoch
            stale = 0
            torch.save(
                checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    validation=validation,
                    args=args,
                    source_checkpoint=source_checkpoint,
                    source_sha256=source_sha256,
                    registry=registry,
                ),
                best_path,
            )
        else:
            stale += 1
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(total_losses)),
            "train_segmentation_loss": float(np.mean(segmentation_losses)),
            "train_domain_loss": float(np.mean(domain_losses)),
            "train_domain_accuracy": float(domain_correct / domain_total) if domain_total else None,
            "train_dsc": float(np.mean(train_dice)),
            "validation": validation,
            "strict_new_high": strict_new_high,
            "stale_epochs": stale,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        write_json_atomic(
            output_dir / "progress.json",
            {
                "schema": SCHEMA + ".progress",
                "schema_version": SCHEMA_VERSION,
                "arm_name": args.arm_name,
                "arm": args.arm,
                "history": history,
                "best_epoch": best_epoch,
                "best_dsc": best_value,
                "test_access": False,
            },
        )
        print(
            f"arm={args.arm} epoch={epoch} loss={record['train_loss']:.5f} "
            f"train={record['train_dsc']:.5f} val={value:.6f} "
            f"subject={validation['final_subject_dsc_mean']:.6f} "
            f"delta={validation['delta_from_base']:+.6f} best={best_value:.6f} "
            f"domain_acc={validation.get('domain_accuracy')}",
            flush=True,
        )
        if stale >= args.patience:
            break

    selected_validation = baseline if best_epoch == 0 else history[best_epoch - 1]["validation"]
    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "arm_name": args.arm_name,
        "arm": args.arm,
        "hypothesis": (
            "fit-only latent-domain conditioning of an identity-initialized safe adapter "
            "can improve subject-clean development Dice over the matched plain adapter"
        ),
        "task": {
            "input": "T1ce plus frozen SAM click-3 raw logits",
            "target": "WT={1,2,3}; raw label 4 excluded",
            "split": "972 train / 108 validation, subject-clean",
            "geometry": "GT-centered N4 128^3 refiner cache; oracle development-only",
            "known_limitation": "validation is development-exposed and localization is GT-derived",
        },
        "initialization": {
            "path": str(source_checkpoint),
            "sha256": source_sha256,
            "base_frozen": True,
            "optimizer_state_restored": False,
        },
        "baseline_validation": baseline,
        "selection": {
            "metric": "validation scan-macro DSC",
            "epoch": best_epoch,
            "value": best_value,
            "subject_macro_value": selected_validation["final_subject_dsc_mean"],
            "delta_from_frozen_base": best_value - baseline["base_dsc_mean"],
        },
        "best_checkpoint": {"path": str(best_path), "sha256": sha256_file(best_path)},
        "train_cache": {
            "path": str(args.train_cache.resolve()),
            "index_sha256": sha256_file(args.train_cache / "index.json"),
            "cases_used": train_limit,
        },
        "validation_cache": {
            "path": str(args.validation_cache.resolve()),
            "index_sha256": sha256_file(args.validation_cache / "index.json"),
            "cases_used": validation_limit,
        },
        "registry": (
            {
                "manifest": str(registry.manifest_path),
                "manifest_sha256": sha256_file(registry.manifest_path),
                "npz": str(registry.npz_path),
                "npz_sha256": sha256_file(registry.npz_path),
            }
            if registry
            else None
        ),
        "parameter_counts": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        },
        "hyperparameters": serializable_args(args),
        "history": history,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "architecture_module_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / "sammed3d/t1ce_wt_domain_sage.py"
        ),
        "elapsed_seconds": time.time() - started,
    }
    write_json_atomic(output_dir / "TRAIN_DONE.json", report)
    print(
        f"TRAIN_DONE arm={args.arm} best_epoch={best_epoch} best={best_value:.9f}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--domain-registry", type=Path)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-init-sha256", default="")
    parser.add_argument("--expected-baseline-dsc", type=float)
    parser.add_argument("--baseline-tolerance", type=float, default=1e-5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm-name", required=True)
    parser.add_argument("--arm", choices=["plain", "domain"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-init-filters", type=int, default=16)
    parser.add_argument("--base-dropout", type=float, default=0.1)
    parser.add_argument("--adapter-init-filters", type=int, default=8)
    parser.add_argument("--adapter-dropout", type=float, default=0.1)
    parser.add_argument("--domain-count", type=int, default=4)
    parser.add_argument("--domain-hidden-dim", type=int, default=64)
    parser.add_argument("--film-rank", type=int, default=4)
    parser.add_argument("--domain-weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-shift", type=int, default=12)
    parser.add_argument("--coarse-shift-max", type=int, default=4)
    parser.add_argument("--coarse-shift-probability", type=float, default=1.0)
    parser.add_argument("--limit-train-cases", type=int, default=0)
    parser.add_argument("--limit-validation-cases", type=int, default=0)
    args = parser.parse_args()
    if args.arm == "domain" and args.domain_registry is None:
        parser.error("--domain-registry is required for the domain arm")
    run(args)


if __name__ == "__main__":
    main()
