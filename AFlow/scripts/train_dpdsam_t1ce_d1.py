#!/usr/bin/env python3
"""Train the fit-only latent-domain D1 T1ce binary reconstruction."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import __version__ as sklearn_version
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d import dpdsam_binary_model as binary_model  # noqa: E402
from sammed3d import engine  # noqa: E402
from sammed3d.t1ce_binary import (  # noqa: E402
    T1ceInferenceCase,
    T1ceSupervisedCase,
    load_t1ce_inference_view,
    load_t1ce_supervised_view,
    score_t1ce_dual_grid,
)
from scripts import train_dpdsam_t1ce_b0 as b0_runner  # noqa: E402


TRAINING_SCHEMA = "aflow.sammed3d.dpdsam_t1ce_binary_d1_training"
TRAINING_SCHEMA_VERSION = 2
BEST_CHECKPOINT_NAME = "best_model_grid_dsc.pth"
DOMAIN_SCHEMA = "aflow.sammed3d.dpdsam_t1ce_binary_d1_latent_domains"
DOMAIN_SCHEMA_VERSION = 1


class T1ceDescriptorDataset(Dataset):
    """Image-only dataset used before any supervised training starts."""

    def __init__(self, rows: list[dict]) -> None:
        self.cases = [
            T1ceInferenceCase(
                case_id=str(row["case_id"]),
                t1ce_path=Path(str(row["image_path"])),
            )
            for row in rows
        ]

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        view = load_t1ce_inference_view(self.cases[index])
        return {"case_id": view.case_id, "image": view.image}


@torch.no_grad()
def compute_fit_descriptors(
    model: binary_model.DynamicDomainPromptBinarySAM,
    rows: list[dict],
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    batch_size: int,
    num_workers: int,
) -> tuple[list[str], np.ndarray]:
    model.sam_model.image_encoder.eval()
    loader = DataLoader(
        T1ceDescriptorDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    case_ids: list[str] = []
    descriptors: list[np.ndarray] = []
    for batch in tqdm(loader, desc="fit-only descriptors", leave=False):
        image = batch["image"].to(device, non_blocking=True).float()
        with torch.amp.autocast(
            "cuda",
            enabled=amp_dtype is not None and device.type == "cuda",
            dtype=amp_dtype or torch.float16,
        ):
            embedding = model.sam_model.image_encoder(image)
        descriptor = embedding.float().mean(dim=(-3, -2, -1)).cpu().numpy()
        if not np.isfinite(descriptor).all():
            raise RuntimeError("fit descriptor contains non-finite values")
        case_ids.extend(str(value) for value in batch["case_id"])
        descriptors.append(descriptor.astype(np.float32, copy=False))
    values = np.concatenate(descriptors, axis=0)
    if values.shape != (len(rows), model.domain_adapter.descriptor_dim):
        raise RuntimeError(f"unexpected descriptor matrix shape: {values.shape}")
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("fit descriptor cache contains duplicate case IDs")
    return case_ids, values


def fit_latent_domains(
    descriptors: np.ndarray,
    *,
    domain_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    values = np.asarray(descriptors, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < domain_count:
        raise ValueError("descriptor matrix cannot support requested domain count")
    if not np.isfinite(values).all():
        raise ValueError("descriptors must be finite")
    estimator = KMeans(
        n_clusters=domain_count,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=seed,
        algorithm="lloyd",
    )
    labels = estimator.fit_predict(values).astype(np.int64, copy=False)
    centroids = estimator.cluster_centers_.astype(np.float32, copy=False)
    nearest = np.square(values[:, None, :] - centroids[None, :, :]).sum(axis=-1).argmin(1)
    if not np.array_equal(labels, nearest):
        raise RuntimeError("sklearn labels disagree with frozen nearest-centroid assignment")
    metadata = {
        "library": "scikit-learn",
        "version": sklearn_version,
        "class": "sklearn.cluster.KMeans",
        "n_clusters": domain_count,
        "init": "k-means++",
        "n_init": 10,
        "max_iter": 300,
        "tol": 1e-4,
        "random_state": seed,
        "algorithm": "lloyd",
        "inertia": float(estimator.inertia_),
        "n_iter": int(estimator.n_iter_),
    }
    return centroids, labels, metadata


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_domain_artifact(
    output_dir: Path,
    *,
    case_ids: list[str],
    descriptors: np.ndarray,
    centroids: np.ndarray,
    pseudo_labels: np.ndarray,
    kmeans: Mapping[str, Any],
    sources: Mapping[str, Any],
) -> dict[str, Any]:
    array_path = output_dir / "fit_only_latent_domains.npz"
    metadata_path = output_dir / "fit_only_latent_domains.json"
    atomic_npz(
        array_path,
        case_ids=np.asarray(case_ids, dtype="U32"),
        descriptors=np.asarray(descriptors, dtype=np.float32),
        centroids=np.asarray(centroids, dtype=np.float32),
        pseudo_labels=np.asarray(pseudo_labels, dtype=np.int64),
    )
    metadata = {
        "schema": DOMAIN_SCHEMA,
        "schema_version": DOMAIN_SCHEMA_VERSION,
        "fit_only": True,
        "reads_validation_images": False,
        "reads_any_labels": False,
        "case_count": len(case_ids),
        "descriptor_shape": list(descriptors.shape),
        "centroid_shape": list(centroids.shape),
        "domain_counts": np.bincount(
            pseudo_labels, minlength=centroids.shape[0]
        ).astype(int).tolist(),
        "kmeans": dict(kmeans),
        "sources": dict(sources),
        "array_path": str(array_path.resolve()),
        "array_sha256": b0_runner.sha256_file(array_path),
    }
    b0_runner.atomic_json(metadata, metadata_path)
    return {
        "array_path": str(array_path.resolve()),
        "array_sha256": metadata["array_sha256"],
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": b0_runner.sha256_file(metadata_path),
        "metadata": metadata,
    }


def load_domain_artifact(
    output_dir: Path, *, expected_sources: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, int], dict[str, Any]]:
    array_path = output_dir / "fit_only_latent_domains.npz"
    metadata_path = output_dir / "fit_only_latent_domains.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != DOMAIN_SCHEMA or metadata.get("sources") != expected_sources:
        raise ValueError("latent-domain artifact source/configuration mismatch")
    if b0_runner.sha256_file(array_path) != metadata.get("array_sha256"):
        raise ValueError("latent-domain array hash mismatch")
    with np.load(array_path, allow_pickle=False) as arrays:
        centroids = np.asarray(arrays["centroids"], dtype=np.float32)
        case_ids = arrays["case_ids"].astype(str).tolist()
        labels = np.asarray(arrays["pseudo_labels"], dtype=np.int64)
        descriptors = np.asarray(arrays["descriptors"], dtype=np.float32)
    if len(case_ids) != int(metadata["case_count"]) or labels.shape != (len(case_ids),):
        raise ValueError("latent-domain artifact case contract mismatch")
    nearest = np.square(descriptors[:, None, :] - centroids[None, :, :]).sum(-1).argmin(1)
    if not np.array_equal(nearest, labels):
        raise ValueError("latent-domain artifact pseudo-label mismatch")
    pseudo_domain_by_case_id = {
        case_id: int(label) for case_id, label in zip(case_ids, labels.tolist())
    }
    return centroids, pseudo_domain_by_case_id, {
        "array_path": str(array_path.resolve()),
        "array_sha256": metadata["array_sha256"],
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": b0_runner.sha256_file(metadata_path),
        "metadata": metadata,
    }


def d1_loss(
    output: binary_model.DynamicBinarySAMOutput,
    target: torch.Tensor,
    *,
    epoch: int,
    warmup_epochs: int,
    adapter_beta: float,
    pseudo_domain_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    segmentation, parts = b0_runner.dice_bce_loss(output.logits, target)
    adapter_ce = F.cross_entropy(
        output.domain_logits.float(),
        (
            output.pseudo_domain_ids
            if pseudo_domain_ids is None
            else pseudo_domain_ids
        ).detach().long(),
    )
    if epoch <= 0 or warmup_epochs < 0:
        raise ValueError("epoch must be positive and warmup_epochs non-negative")
    # Phase 1 freezes diffusion/CDR, not the domain adapter. D1 contains no CDR,
    # so adapter supervision is active for every training epoch.
    effective_beta = float(adapter_beta)
    total = segmentation + effective_beta * adapter_ce
    return total, {
        **parts,
        "segmentation": float(segmentation.detach()),
        "adapter_ce": float(adapter_ce.detach()),
        "effective_adapter_beta": effective_beta,
    }


def train_epoch(
    model: binary_model.DynamicDomainPromptBinarySAM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    *,
    epoch: int,
    warmup_epochs: int,
    adapter_beta: float,
    pseudo_domain_by_case_id: Mapping[str, int],
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "bce": 0.0,
        "dice_loss": 0.0,
        "segmentation": 0.0,
        "adapter_ce": 0.0,
        "samples": 0.0,
    }
    effective_beta = float(adapter_beta)
    for batch in tqdm(loader, desc="train D1", leave=False):
        image = batch["image"].to(device, non_blocking=True).float()
        target = batch["target"].to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            "cuda",
            enabled=amp_dtype is not None and device.type == "cuda",
            dtype=amp_dtype or torch.float16,
        ):
            output = model(image)
            pseudo_ids = torch.tensor(
                [pseudo_domain_by_case_id[str(case_id)] for case_id in batch["case_id"]],
                dtype=torch.long,
                device=device,
            )
            loss, parts = d1_loss(
                output,
                target,
                epoch=epoch,
                warmup_epochs=warmup_epochs,
                adapter_beta=adapter_beta,
                pseudo_domain_ids=pseudo_ids,
            )
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        size = int(image.shape[0])
        totals["loss"] += float(loss.detach()) * size
        for key in ("bce", "dice_loss", "segmentation", "adapter_ce"):
            totals[key] += parts[key] * size
        totals["samples"] += size
        effective_beta = parts["effective_adapter_beta"]
    samples = max(totals.pop("samples"), 1.0)
    result = {key: value / samples for key, value in totals.items()}
    result["effective_adapter_beta"] = effective_beta
    return result


@torch.no_grad()
def validate(
    model: binary_model.DynamicDomainPromptBinarySAM,
    rows: list[dict],
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict[str, Any]:
    model.eval()
    cases = []
    for row in tqdm(rows, desc="validation D1", leave=False):
        inference_case = T1ceInferenceCase(
            str(row["case_id"]), Path(str(row["image_path"]))
        )
        inference_view = load_t1ce_inference_view(inference_case)
        image = inference_view.image.unsqueeze(0).to(device).float()
        with torch.amp.autocast(
            "cuda",
            enabled=amp_dtype is not None and device.type == "cuda",
            dtype=amp_dtype or torch.float16,
        ):
            probability = model(image).probabilities
        prediction = probability[0, 0].float().cpu() >= 0.5
        supervised_case = T1ceSupervisedCase.from_manifest_row(row)
        supervised_view = load_t1ce_supervised_view(supervised_case)
        if inference_view.geometry != supervised_view.geometry:
            raise RuntimeError(f"validation geometry changed for {inference_case.case_id}")
        dual = score_t1ce_dual_grid(prediction, supervised_view).to_dict()
        dual["headline_metric"] = "model_grid.dsc"
        cases.append({"case_id": inference_case.case_id, **dual})
    return b0_runner.summarize_dual_grid_cases(cases)


def validate_resume_payload(
    payload: Mapping[str, Any], implementation_manifest: Mapping[str, Any]
) -> None:
    if payload.get("training_schema") != TRAINING_SCHEMA:
        raise ValueError("resume checkpoint has an incompatible D1 training schema")
    if payload.get("training_schema_version") != TRAINING_SCHEMA_VERSION:
        raise ValueError("resume checkpoint has an incompatible D1 schema version")
    expected_hash = b0_runner.canonical_json_sha256(implementation_manifest)
    if payload.get("implementation_manifest_sha256") != expected_hash:
        raise ValueError("resume checkpoint implementation manifest hash mismatch")
    if payload.get("implementation_manifest") != implementation_manifest:
        raise ValueError("resume checkpoint run configuration mismatch")
    model_manifest = implementation_manifest.get("model")
    if payload.get("dpdsam_binary_config") != model_manifest:
        raise ValueError("resume checkpoint D1 model metadata mismatch")
    if payload.get("adapter_config") != model_manifest.get("adapter_config"):
        raise ValueError("resume checkpoint adapter architecture mismatch")
    required = {
        "model_state_dict",
        "dpdsam_dynamic_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "epoch",
        "history",
        "best",
        "rng_state",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"resume checkpoint is missing D1 fields: {sorted(missing)}")


def restore_training_checkpoint(
    payload: Mapping[str, Any],
    *,
    model: binary_model.DynamicDomainPromptBinarySAM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    generator: torch.Generator,
    implementation_manifest: Mapping[str, Any],
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    validate_resume_payload(payload, implementation_manifest)
    optimizer_manifest = implementation_manifest["optimizer"]
    groups = payload["optimizer_state_dict"].get("param_groups", [])
    if len(groups) != len(optimizer.param_groups):
        raise ValueError("resume optimizer parameter-group count mismatch")
    for group in groups:
        checks = (
            (float(group["lr"]), float(optimizer_manifest["learning_rate"]), "learning-rate"),
            (float(group["weight_decay"]), float(optimizer_manifest["weight_decay"]), "weight-decay"),
            (float(group["eps"]), float(optimizer_manifest["epsilon"]), "epsilon"),
        )
        for saved, expected, name in checks:
            if saved != expected:
                raise ValueError(f"resume optimizer {name} mismatch")
        if list(group["betas"]) != list(optimizer_manifest["betas"]):
            raise ValueError("resume optimizer beta mismatch")
        if bool(group["amsgrad"]) != bool(optimizer_manifest["amsgrad"]):
            raise ValueError("resume optimizer amsgrad mismatch")
    model.sam_model.load_state_dict(payload["model_state_dict"], strict=True)
    dynamic = payload["dpdsam_dynamic_state_dict"]
    for name, module in (
        ("domain_registry", model.domain_registry),
        ("domain_adapter", model.domain_adapter),
        ("prompt_generator", model.prompt_generator),
    ):
        module.load_state_dict(dynamic[name], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    saved_scaler = payload["scaler_state_dict"]
    if (saved_scaler is None) != (scaler is None):
        raise ValueError("resume checkpoint AMP scaler contract mismatch")
    if scaler is not None:
        scaler.load_state_dict(saved_scaler)
    epoch = int(payload["epoch"])
    history = payload["history"]
    best = payload["best"]
    if epoch < 0 or not isinstance(history, list) or len(history) != epoch:
        raise ValueError("resume checkpoint epoch/history contract is invalid")
    if not isinstance(best, dict):
        raise ValueError("resume checkpoint best record must be a dict")
    b0_runner.restore_rng_state(payload["rng_state"], generator)
    return epoch, list(history), dict(best)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--validation-jsonl", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path(engine.DEFAULT_CKPT))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--descriptor-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--adapter-beta", type=float, default=0.1)
    parser.add_argument("--domain-count", type=int, default=4)
    parser.add_argument("--prompt-tokens", type=int, default=1)
    parser.add_argument("--adapter-hidden-dim", type=int, default=384)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--subset-seed", type=int, default=20260721)
    parser.add_argument("--kmeans-seed", type=int, default=20260721)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--max-train-cases", type=int, default=0)
    parser.add_argument("--max-validation-cases", type=int, default=0)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--no-preload", action="store_true")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(
        args.epochs,
        args.batch_size,
        args.descriptor_batch_size,
        args.validation_interval,
        args.domain_count,
        args.prompt_tokens,
        args.adapter_hidden_dim,
    ) <= 0:
        raise ValueError("epoch, batch, interval, and D1 dimensions must be positive")
    if not 0 <= args.warmup_epochs <= args.epochs or args.adapter_beta < 0:
        raise ValueError("invalid D1 warmup or adapter beta")
    if args.domain_count != 4 or args.prompt_tokens != 1:
        raise ValueError("frozen D1 requires K=4 and N=1")

    latest = args.output_dir / "latest.pth"
    resume_payload: Mapping[str, Any] | None = None
    if args.resume is None:
        if args.output_dir.exists():
            raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=False)
    else:
        if not args.output_dir.is_dir() or args.resume.resolve() != latest.resolve():
            raise ValueError("--resume requires existing OUTPUT_DIR/latest.pth")
        loaded = torch.load(args.resume, map_location="cpu", weights_only=False)
        if not isinstance(loaded, Mapping):
            raise TypeError("resume checkpoint must contain a mapping")
        resume_payload = loaded

    b0_runner.seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[
        args.amp_dtype
    ]
    if device.type != "cuda":
        amp_dtype = None
    train_rows = b0_runner.read_rows(
        args.train_jsonl,
        args.max_train_cases,
        expected_split="train",
        subset_seed=args.subset_seed,
    )
    validation_rows = b0_runner.read_rows(
        args.validation_jsonl,
        args.max_validation_cases,
        expected_split="validation",
        subset_seed=args.subset_seed,
    )
    train_subjects = {str(row["subject_id"]) for row in train_rows}
    validation_subjects = {str(row["subject_id"]) for row in validation_rows}
    if train_subjects & validation_subjects:
        raise RuntimeError("train/validation subject leakage")
    source_hashes = {
        "train_manifest_path": str(args.train_jsonl.resolve()),
        "train_manifest_sha256": b0_runner.sha256_file(args.train_jsonl),
        "split_manifest_path": str(args.split_manifest.resolve()),
        "split_manifest_sha256": b0_runner.sha256_file(args.split_manifest),
        "source_checkpoint_path": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": b0_runner.sha256_file(args.checkpoint),
        "model_type": engine.MODEL_TYPE,
        "preprocessing": "T1ce_nonzero_zscore_whole_volume_128cubed",
        "kmeans_seed": args.kmeans_seed,
        "domain_count": args.domain_count,
    }
    model, initial_metadata = binary_model.load_untrained_d1(
        args.checkpoint,
        lambda model_type: engine.sam_model_registry3D[model_type](checkpoint=None),
        device=device,
        model_type=engine.MODEL_TYPE,
        domain_count=args.domain_count,
        prompt_tokens=args.prompt_tokens,
        adapter_hidden_dim=args.adapter_hidden_dim,
    )
    if resume_payload is None:
        case_ids, descriptors = compute_fit_descriptors(
            model,
            train_rows,
            device=device,
            amp_dtype=amp_dtype,
            batch_size=args.descriptor_batch_size,
            num_workers=args.num_workers,
        )
        centroids, labels, kmeans = fit_latent_domains(
            descriptors, domain_count=args.domain_count, seed=args.kmeans_seed
        )
        domain_artifact = write_domain_artifact(
            args.output_dir,
            case_ids=case_ids,
            descriptors=descriptors,
            centroids=centroids,
            pseudo_labels=labels,
            kmeans=kmeans,
            sources=source_hashes,
        )
        pseudo_domain_by_case_id = {
            case_id: int(label) for case_id, label in zip(case_ids, labels.tolist())
        }
    else:
        centroids, pseudo_domain_by_case_id, domain_artifact = load_domain_artifact(
            args.output_dir, expected_sources=source_hashes
        )
    expected_train_ids = {str(row["case_id"]) for row in train_rows}
    if set(pseudo_domain_by_case_id) != expected_train_ids:
        raise RuntimeError("latent-domain pseudo-label IDs do not equal the fit manifest")
    model.set_domain_centroids(torch.from_numpy(centroids).to(device))
    model_metadata = model.metadata()
    model_metadata["adapter_config"] = initial_metadata["adapter_config"]
    trainable = binary_model.configure_d1_trainable(model)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda" and amp_dtype == torch.float16
        else None
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        b0_runner.T1ceTrainDataset(
            train_rows,
            augment=not args.no_augmentation,
            preload=not args.no_preload,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=b0_runner.seed_worker,
        generator=generator,
        persistent_workers=False,
    )
    configuration_args = {
        key: (str(value.resolve()) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
        if key not in {"output_dir", "resume"}
    }
    config = {
        "schema": TRAINING_SCHEMA,
        "schema_version": TRAINING_SCHEMA_VERSION,
        "args": configuration_args,
        "model": model_metadata,
        "data": {
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "validation_manifest_sha256": b0_runner.sha256_file(args.validation_jsonl),
            "train_subjects": len(train_subjects),
            "validation_subjects": len(validation_subjects),
            **source_hashes,
        },
        "latent_domains": domain_artifact,
        "loss": {
            "segmentation": "Dice_plus_BCE_equal_weight",
            "adapter_ce_beta": args.adapter_beta,
            "adapter_ce_start_epoch": 1,
            "adapter_ce_target": "cached_unaugmented_fit_image_pseudo_label_by_case_id",
            "phase1_epochs_no_cdr": args.warmup_epochs,
            "phase1_interpretation": "diffusion_CDR_frozen_adapter_CE_active",
            "runner_contains_cdr": False,
        },
        "optimizer": {
            "class": "torch.optim.AdamW",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "betas": list(optimizer.defaults["betas"]),
            "epsilon": optimizer.defaults["eps"],
            "amsgrad": optimizer.defaults["amsgrad"],
        },
        "resume": {
            "checkpoint": "OUTPUT_DIR/latest.pth_only",
            "rng": "python_numpy_torch_cpu_cuda_all_and_dataloader_generator",
            "persistent_workers": False,
        },
        "metrics": {
            "headline": "model_grid_case_macro_DSC_IoU_HD95",
            "model_grid": "128cubed_target_with_physical_model_spacing",
            "secondary": "source_grid_case_macro_DSC_IoU_HD95",
            "source_grid_restore": "nearest-exact",
            "empty_counts": "both_empty_and_one_empty_per_grid",
        },
        "selection": "maximum_validation_model_grid_macro_dsc",
        "metric_threshold": 0.5,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    manifest_path = args.output_dir / "implementation_manifest.json"
    if resume_payload is None:
        b0_runner.atomic_json(config, manifest_path)
        start_epoch = 0
        history: list[dict[str, Any]] = []
        best: dict[str, Any] = {
            "dsc": -1.0,
            "epoch": -1,
            "checkpoint": BEST_CHECKPOINT_NAME,
            "grid": "model_grid",
            "metric": "dsc",
        }
    else:
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved_manifest != config:
            raise ValueError("on-disk implementation manifest configuration mismatch")
        start_epoch, history, best = restore_training_checkpoint(
            resume_payload,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            generator=generator,
            implementation_manifest=config,
        )
        if start_epoch > args.epochs:
            raise ValueError("resume epoch exceeds configured epochs")
    elapsed_offset = float(history[-1]["elapsed_seconds"]) if history else 0.0
    start = time.time()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            amp_dtype,
            epoch=epoch,
            warmup_epochs=args.warmup_epochs,
            adapter_beta=args.adapter_beta,
            pseudo_domain_by_case_id=pseudo_domain_by_case_id,
        )
        validation_metrics = None
        if epoch % args.validation_interval == 0 or epoch == args.epochs:
            validation_metrics = validate(model, validation_rows, device, amp_dtype)
        record = {
            "epoch": epoch,
            "phase": (
                "phase1_coarse_no_cdr_adapter_ce_active"
                if epoch <= args.warmup_epochs
                else "post_phase1_D1_no_cdr_adapter_ce_active"
            ),
            "train": train_metrics,
            "validation": (
                {key: value for key, value in validation_metrics.items() if key != "case_metrics"}
                if validation_metrics is not None
                else None
            ),
            "elapsed_seconds": elapsed_offset + time.time() - start,
        }
        history.append(record)
        improved = bool(
            validation_metrics is not None
            and b0_runner.validation_selection_score(validation_metrics) > best["dsc"]
        )
        if improved:
            best = {
                "dsc": b0_runner.validation_selection_score(validation_metrics),
                "epoch": epoch,
                "checkpoint": BEST_CHECKPOINT_NAME,
                "grid": "model_grid",
                "metric": "dsc",
            }
        payload = binary_model.checkpoint_payload(
            model,
            adapter_config=model_metadata["adapter_config"],
            extra={
                "training_schema": TRAINING_SCHEMA,
                "training_schema_version": TRAINING_SCHEMA_VERSION,
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "history": history,
                "best": best,
                "implementation_manifest": config,
                "implementation_manifest_sha256": b0_runner.canonical_json_sha256(config),
                "rng_state": b0_runner.capture_rng_state(generator),
            },
        )
        if validation_metrics is not None:
            b0_runner.atomic_json(
                {"epoch": epoch, **validation_metrics},
                args.output_dir / f"validation_epoch_{epoch:03d}.json",
            )
            if improved:
                b0_runner.atomic_torch_save(payload, args.output_dir / best["checkpoint"])
        b0_runner.atomic_torch_save(payload, latest)
        b0_runner.atomic_json(
            {
                "status": "running",
                "epoch": epoch,
                "history": history,
                "best": best,
                "implementation_manifest_sha256": b0_runner.canonical_json_sha256(config),
            },
            args.output_dir / "progress.json",
        )
        print(json.dumps(record, sort_keys=True), flush=True)
    b0_runner.atomic_json(
        {
            "status": "complete",
            "epoch": args.epochs,
            "history": history,
            "best": best,
            "implementation_manifest_sha256": b0_runner.canonical_json_sha256(config),
        },
        args.output_dir / "progress.json",
    )


if __name__ == "__main__":
    main()
