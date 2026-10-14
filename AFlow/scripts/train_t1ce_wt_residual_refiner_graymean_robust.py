#!/usr/bin/env python3
"""Train a residual refiner with independent coarse-logit translation.

The translation makes in-sample train logits resemble harder held-out logits.
This wrapper keeps the frozen gray-mean runner intact for provenance and adds
the corruption immediately after its original joint spatial/intensity augment.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_t1ce_wt_residual_refiner_graymean as base  # noqa: E402
from scripts.train_sammed3d_t1ce_wt_residual_refiner import (  # noqa: E402
    shift_without_wrap,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_residual_refiner_graymean_robust_training"
SCHEMA_VERSION = 1


def corrupt_coarse_batch(
    coarse: torch.Tensor,
    *,
    max_shift: int,
    probability: float,
) -> torch.Tensor:
    """Translate each selected coarse-logit volume with zero-filled borders."""
    if max_shift < 0:
        raise ValueError("coarse max_shift must be non-negative")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("coarse shift probability must be in [0, 1]")
    if max_shift == 0 or probability == 0:
        return coarse
    for batch_index in range(coarse.shape[0]):
        if bool(torch.rand((), device=coarse.device) < probability):
            shifts = tuple(
                int(torch.randint(-max_shift, max_shift + 1, (), device=coarse.device))
                for _ in range(3)
            )
            coarse[batch_index] = shift_without_wrap(coarse[batch_index], shifts)
    return coarse


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.coarse_shift_max < 0:
        raise ValueError("coarse_shift_max must be non-negative")
    if not 0.0 <= args.coarse_shift_probability <= 1.0:
        raise ValueError("coarse_shift_probability must be in [0, 1]")
    base_runner = Path(base.__file__).resolve()
    base_runner_sha256 = base.sha256_file(base_runner)
    if (
        args.expected_base_runner_sha256
        and base_runner_sha256 != args.expected_base_runner_sha256
    ):
        raise RuntimeError("base gray-mean runner hash mismatch")

    original_augment_batch = base.augment_batch

    def robust_augment_batch(
        image: torch.Tensor,
        coarse: torch.Tensor,
        label: torch.Tensor,
        *,
        max_shift: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, coarse, label = original_augment_batch(
            image,
            coarse,
            label,
            max_shift=max_shift,
        )
        coarse = corrupt_coarse_batch(
            coarse,
            max_shift=args.coarse_shift_max,
            probability=args.coarse_shift_probability,
        )
        return image, coarse, label

    base.augment_batch = robust_augment_batch
    try:
        report = base.run(args)
    finally:
        base.augment_batch = original_augment_batch

    corruption = {
        "kind": "independent zero-filled integer translation",
        "max_shift_voxels_per_axis": args.coarse_shift_max,
        "probability": args.coarse_shift_probability,
        "training_only": True,
        "validation_corruption": False,
        "label_used_to_construct_corruption": False,
        "purpose": "reduce the in-sample train versus held-out validation coarse-logit difficulty gap",
    }
    best_path = Path(report["best_checkpoint"]["path"])
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    checkpoint["coarse_input_corruption"] = corruption
    checkpoint["training_runner"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": base.sha256_file(Path(__file__).resolve()),
        "base_runner_path": str(base_runner),
        "base_runner_sha256": base_runner_sha256,
    }
    torch.save(checkpoint, best_path)

    report["schema"] = SCHEMA
    report["schema_version"] = SCHEMA_VERSION
    report["hypothesis"] = (
        "training-only coarse-logit translation improves held-out robustness; "
        "gray-mean consistency may add benefit after difficulty alignment"
    )
    report["coarse_input_corruption"] = corruption
    report["base_runner_sha256"] = report["runner_sha256"]
    report["runner_sha256"] = base.sha256_file(Path(__file__).resolve())
    report["best_checkpoint"]["sha256"] = base.sha256_file(best_path)
    base.write_json_atomic(Path(args.output_dir).resolve() / "TRAIN_DONE.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-init-sha256", default="")
    parser.add_argument("--expected-base-runner-sha256", default="")
    parser.add_argument("--expected-baseline-dsc", type=float)
    parser.add_argument("--baseline-tolerance", type=float, default=1e-5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm-name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--init-filters", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-shift", type=int, default=12)
    parser.add_argument("--gray-weight", type=float, required=True)
    parser.add_argument("--gray-temperature", type=float, default=1.0)
    parser.add_argument("--gray-epsilon", type=float, default=1e-6)
    parser.add_argument("--coarse-shift-max", type=int, required=True)
    parser.add_argument("--coarse-shift-probability", type=float, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
