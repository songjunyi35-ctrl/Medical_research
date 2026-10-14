#!/usr/bin/env python3
"""Build a fit-only latent-domain registry for Domain-SAGE 3D."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from sammed3d.t1ce_wt_domain_sage import (
    fit_kmeans_plus_plus,
    frozen_image_descriptor,
    nearest_centroid_ids,
)
from scripts.train_sammed3d_t1ce_wt_residual_refiner import (
    RefinerDataset,
    ResidualRefiner,
    sha256_file,
    write_json_atomic,
)
from scripts.train_t1ce_wt_safe_adapter import SOURCE_CHECKPOINT_SCHEMA


SCHEMA = "aflow.sammed3d.t1ce_wt_domain_sage_registry"
SCHEMA_VERSION = 1


@torch.inference_mode()
def extract_descriptors(
    base: ResidualRefiner,
    source: RefinerDataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[list[str], np.ndarray]:
    loader = DataLoader(
        source,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    case_ids: list[str] = []
    values: list[np.ndarray] = []
    base.eval()
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        descriptor = frozen_image_descriptor(base, image)
        values.append(descriptor.cpu().numpy())
        case_ids.extend(list(batch["case_id"]))
    if case_ids != source.case_ids:
        raise RuntimeError("descriptor extraction changed cache order")
    array = np.concatenate(values, axis=0).astype(np.float32, copy=False)
    if array.shape[0] != len(source) or not np.isfinite(array).all():
        raise RuntimeError("descriptor extraction produced an invalid array")
    return case_ids, array


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("registry extraction requires CUDA")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_source = RefinerDataset(args.train_cache)
    validation_source = RefinerDataset(args.validation_cache)
    if train_source.index.get("split") != "train":
        raise ValueError("registry fit source must be the train cache")
    if validation_source.index.get("split") != "validation":
        raise ValueError("registry validation source must be the validation cache")
    if train_source.index.get("test_access") is not False:
        raise ValueError("test-accessible training cache is forbidden")
    if validation_source.index.get("test_access") is not False:
        raise ValueError("test-accessible validation cache is forbidden")

    checkpoint = args.init_checkpoint.resolve()
    checkpoint_sha256 = sha256_file(checkpoint)
    if args.expected_init_sha256 and checkpoint_sha256 != args.expected_init_sha256:
        raise RuntimeError("robust base checkpoint hash mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != SOURCE_CHECKPOINT_SCHEMA:
        raise ValueError("unexpected robust base checkpoint schema")
    architecture = payload.get("architecture", {})
    if int(architecture.get("init_filters", -1)) != args.base_init_filters:
        raise ValueError("base init_filters mismatch")
    if float(architecture.get("dropout", -1)) != args.base_dropout:
        raise ValueError("base dropout mismatch")
    base = ResidualRefiner(args.base_init_filters, args.base_dropout).to(device)
    base.load_state_dict(payload["model_state_dict"], strict=True)
    base.requires_grad_(False)
    base.eval()

    train_ids, train_raw = extract_descriptors(
        base,
        train_source,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    validation_ids, validation_raw = extract_descriptors(
        base,
        validation_source,
        device=device,
        batch_size=args.batch_size,
        num_workers=max(0, args.num_workers // 2),
    )
    feature_mean = train_raw.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = train_raw.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = np.maximum(feature_std, np.float32(1e-6))
    train_values = ((train_raw - feature_mean) / feature_std).astype(np.float32)
    validation_values = ((validation_raw - feature_mean) / feature_std).astype(np.float32)
    centroids, train_domains, iterations = fit_kmeans_plus_plus(
        train_values,
        cluster_count=args.domain_count,
        seed=args.kmeans_seed,
        max_iterations=args.kmeans_max_iterations,
    )
    validation_domains = nearest_centroid_ids(validation_values, centroids)
    counts = np.bincount(train_domains, minlength=args.domain_count)
    if np.any(counts == 0):
        raise RuntimeError("fit-only K-means produced an empty domain")

    registry_path = output_dir / "registry.npz"
    with registry_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            train_case_ids=np.asarray(train_ids, dtype="U64"),
            validation_case_ids=np.asarray(validation_ids, dtype="U64"),
            train_descriptors=train_values,
            validation_descriptors=validation_values,
            train_domain_ids=train_domains,
            validation_domain_ids=validation_domains,
            centroids=centroids,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )
    manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "fit_split": "train only",
        "validation_used_for_centroids": False,
        "labels_used_for_descriptors_or_clustering": False,
        "descriptor": {
            "name": "frozen robust-base image-only bottleneck GAP surrogate",
            "input": "T1ce with an identically zero coarse channel",
            "dimension": int(train_values.shape[1]),
            "normalization": "fit-only per-channel mean/std",
        },
        "clustering": {
            "algorithm": "deterministic local K-means++",
            "domain_count": args.domain_count,
            "seed": args.kmeans_seed,
            "iterations": iterations,
            "train_counts": counts.tolist(),
            "validation_nearest_counts": np.bincount(
                validation_domains, minlength=args.domain_count
            ).tolist(),
        },
        "cases": {"train": len(train_ids), "validation": len(validation_ids)},
        "source_checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha256,
        },
        "train_cache": {
            "path": str(args.train_cache.resolve()),
            "index_sha256": sha256_file(args.train_cache / "index.json"),
        },
        "validation_cache": {
            "path": str(args.validation_cache.resolve()),
            "index_sha256": sha256_file(args.validation_cache / "index.json"),
        },
        "registry_npz": {
            "path": str(registry_path),
            "sha256": sha256_file(registry_path),
        },
        "hyperparameters": serializable_args(args),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_atomic(output_dir / "registry.json", manifest)
    print(
        f"DOMAIN_REGISTRY_DONE train={len(train_ids)} validation={len(validation_ids)} "
        f"dim={train_values.shape[1]} counts={counts.tolist()}",
        flush=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-init-sha256", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--kmeans-seed", type=int, default=20260818)
    parser.add_argument("--kmeans-max-iterations", type=int, default=100)
    parser.add_argument("--domain-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-init-filters", type=int, default=16)
    parser.add_argument("--base-dropout", type=float, default=0.1)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
