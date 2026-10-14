#!/usr/bin/env python3
"""Verify label-free parity between online execution and exact cached atoms."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_mcts import (  # noqa: E402
    ROBUST_CHECKPOINT_SHA256,
    RobustAtomCache,
    run_workflow,
    sha256_file,
    workflow_mask,
)
from scripts.benchmark_t1ce_wt_mcts_deployment import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_CHECKPOINTS,
    LabelFreeInferenceCache,
    execute_fixed_workflow,
    load_workflow,
)
from scripts.evaluate_t1ce_wt_residual_refiner_ensemble_validation import load_model  # noqa: E402
from scripts.train_sammed3d_t1ce_wt_residual_refiner import write_json_atomic  # noqa: E402


DEFAULT_ATOMS = Path("workspace/BraTS/cache/t1ce_wt_robust_mcts_atoms_f32")


def mask_dice(left: np.ndarray, right: np.ndarray) -> float:
    denominator = int(np.count_nonzero(left)) + int(np.count_nonzero(right))
    return (
        2.0 * int(np.count_nonzero(left & right)) / denominator
        if denominator
        else 1.0
    )


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict:
    if args.case_count < 1:
        raise ValueError("case-count must be positive")
    spec, workflow_receipt = load_workflow(
        args.workflow_json,
        locked_reference=args.locked_reference,
    )
    online_data = LabelFreeInferenceCache(args.validation_cache)
    atom_cache = RobustAtomCache(args.atom_cache)
    if online_data.case_ids != atom_cache.case_ids:
        raise RuntimeError("online input and exact atom case order mismatch")
    actual_hashes = tuple(sha256_file(path) for path in args.checkpoints)
    if actual_hashes != ROBUST_CHECKPOINT_SHA256:
        raise RuntimeError("robust checkpoint allowlist mismatch")
    selected_slots = sorted(
        slot for slot in range(3) if spec.model_mask & (1 << slot)
    )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("online parity requires CUDA")
    torch.cuda.set_device(device)
    models = {slot: load_model(args.checkpoints[slot], device) for slot in selected_slots}

    per_case = []
    for slot in range(min(args.case_count, len(online_data.case_ids))):
        image, coarse = online_data.tensors(slot)
        online_mask, forwards = execute_fixed_workflow(
            models=models,
            image_cpu=image,
            coarse_cpu=coarse,
            spec=spec,
            device=device,
        )
        cached_output = run_workflow(np.asarray(atom_cache.atoms[slot]), spec)
        cached_mask = workflow_mask(cached_output, spec)
        union = int(np.count_nonzero(online_mask | cached_mask))
        different = int(np.count_nonzero(online_mask ^ cached_mask))
        per_case.append(
            {
                "case_id": online_data.case_ids[slot],
                "member_view_forwards": forwards,
                "mask_identical": different == 0,
                "mask_dice": mask_dice(online_mask, cached_mask),
                "different_voxels": different,
                "symmetric_difference_over_union": different / union if union else 0.0,
                "online_foreground_voxels": int(np.count_nonzero(online_mask)),
                "cached_foreground_voxels": int(np.count_nonzero(cached_mask)),
            }
        )

    report = {
        "schema": "aflow.sammed3d.t1ce_wt_mcts_online_atom_parity",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "uses_labels": False,
        "workflow": workflow_receipt,
        "cases": len(per_case),
        "identical_cases": sum(value["mask_identical"] for value in per_case),
        "minimum_mask_dice": min(value["mask_dice"] for value in per_case),
        "mean_mask_dice": float(np.mean([value["mask_dice"] for value in per_case])),
        "maximum_different_voxels": max(value["different_voxels"] for value in per_case),
        "maximum_symmetric_difference_over_union": max(
            value["symmetric_difference_over_union"] for value in per_case
        ),
        "per_case": per_case,
        "atom_index_sha256": sha256_file(atom_cache.index_path),
        "online_cache_index_sha256": sha256_file(online_data.index_path),
        "checkpoint_sha256": list(actual_hashes),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {key: report[key] for key in (
                "cases",
                "identical_cases",
                "minimum_mask_dice",
                "mean_mask_dice",
                "maximum_different_voxels",
                "maximum_symmetric_difference_over_union",
            )},
            indent=2,
        ),
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--workflow-json", type=Path)
    choice.add_argument("--locked-reference", action="store_true")
    parser.add_argument("--validation-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--atom-cache", type=Path, default=DEFAULT_ATOMS)
    parser.add_argument("--checkpoints", type=Path, nargs=3, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--case-count", type=int, default=8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
