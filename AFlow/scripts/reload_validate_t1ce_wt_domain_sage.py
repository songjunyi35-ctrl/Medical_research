#!/usr/bin/env python3
"""Independently reload and validate a Domain-SAGE paired checkpoint."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from sammed3d.t1ce_wt_domain_sage import DomainSAGESafeAdapter
from sammed3d.t1ce_wt_safe_adapter import FrozenBaseSafeAdapter
from scripts.train_sammed3d_t1ce_wt_residual_refiner import (
    RefinerDataset,
    ResidualRefiner,
    sha256_file,
    write_json_atomic,
)
from scripts.train_t1ce_wt_domain_sage import (
    CHECKPOINT_SCHEMA,
    DomainRegistry,
    load_registry,
    validate,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_domain_sage_reload_validation"


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("reload validation requires CUDA")
    torch.cuda.set_device(device)
    checkpoint_path = args.checkpoint.resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unexpected Domain-SAGE checkpoint schema")
    arm = payload.get("arm")
    if arm not in {"plain", "domain"}:
        raise ValueError("checkpoint arm is invalid")
    source_path = Path(payload["initialization"]["path"]).resolve()
    if sha256_file(source_path) != payload["initialization"]["sha256"]:
        raise RuntimeError("frozen source checkpoint hash mismatch")
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    architecture = source["architecture"]
    base = ResidualRefiner(
        int(architecture["init_filters"]), float(architecture["dropout"])
    )
    base.load_state_dict(source["model_state_dict"], strict=True)

    model_architecture = payload["architecture"]
    registry: DomainRegistry | None = None
    if arm == "plain":
        model: FrozenBaseSafeAdapter | DomainSAGESafeAdapter = FrozenBaseSafeAdapter(
            base,
            mode="plain",
            init_filters=int(model_architecture["adapter_init_filters"]),
            dropout=float(model_architecture["adapter_dropout"]),
        )
    else:
        registry_path = Path(payload["registry"]["manifest_path"]).resolve().parent
        registry = load_registry(registry_path)
        if sha256_file(registry.manifest_path) != payload["registry"]["manifest_sha256"]:
            raise RuntimeError("registry manifest hash differs from checkpoint binding")
        if sha256_file(registry.npz_path) != payload["registry"]["npz_sha256"]:
            raise RuntimeError("registry NPZ hash differs from checkpoint binding")
        model = DomainSAGESafeAdapter(
            base,
            descriptor_dim=int(model_architecture["descriptor_dim"]),
            domain_count=int(model_architecture["domain_count"]),
            domain_hidden_dim=int(model_architecture["domain_hidden_dim"]),
            film_rank=int(model_architecture["film_rank"]),
            init_filters=int(model_architecture["adapter_init_filters"]),
            dropout=float(model_architecture["adapter_dropout"]),
        )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device).eval()

    validation_source = RefinerDataset(args.validation_cache)
    if validation_source.index.get("split") != "validation":
        raise ValueError("reload evaluator accepts validation cache only")
    if validation_source.index.get("test_access") is not False:
        raise ValueError("test-accessible cache is forbidden")
    if registry is not None and list(registry.validation.row_by_case_id) != validation_source.case_ids:
        raise ValueError("registry validation membership/order differs from cache")
    loader = DataLoader(
        validation_source,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    result = validate(model, loader, device, arm=arm, registry=registry)
    embedded = payload["validation"]
    comparisons = {
        key: float(result[key]) - float(embedded[key])
        for key in (
            "base_dsc_mean",
            "base_subject_dsc_mean",
            "final_dsc_mean",
            "final_subject_dsc_mean",
            "delta_from_base",
        )
    }
    if any(abs(value) > args.tolerance for value in comparisons.values()):
        raise RuntimeError(f"reload metrics differ from checkpoint: {comparisons}")
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "arm": arm,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": payload["epoch"],
        },
        "validation_cache": {
            "path": str(args.validation_cache.resolve()),
            "index_sha256": sha256_file(args.validation_cache / "index.json"),
        },
        "validation": result,
        "differences_from_checkpoint": comparisons,
        "tolerance": args.tolerance,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_atomic(args.output, report)
    print(
        f"RELOAD_VALIDATION_DONE arm={arm} scan={result['final_dsc_mean']:.9f} "
        f"subject={result['final_subject_dsc_mean']:.9f}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=0.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
