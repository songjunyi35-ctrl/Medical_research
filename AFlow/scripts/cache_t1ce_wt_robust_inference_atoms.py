#!/usr/bin/env python3
"""Materialize 3-model x 8-reflection validation logits for MCTS search.

The runner has no test mode.  It verifies the current persistent validation
cache and the three frozen no-gray robust checkpoints, writes every inverse-
transformed logit atom once, then requires the historical locked workflow to
pass a Dice parity gate before publishing ``index.json``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_mcts import (  # noqa: E402
    ATOM_CACHE_SCHEMA,
    ENSEMBLE_REPORT_SHA256,
    ENSEMBLE_TTA_REPORT_SHA256,
    FORMAL_AGGREGATE_PARITY_TOLERANCE,
    FORMAL_PER_CASE_PARITY_TOLERANCE,
    LOCKED_SCAN_MACRO_DSC,
    LOCKED_SUBJECT_MACRO_DSC,
    REFLECTION_AXES,
    ROBUST_CHECKPOINT_SHA256,
    ROBUST_SEEDS,
    VALIDATION_CACHE_INDEX_SHA256,
    WorkflowEvaluator,
    WorkflowSpec,
    sha256_file,
    stable_id,
)
from scripts.evaluate_t1ce_wt_residual_refiner_ensemble_validation import (  # noqa: E402
    load_model,
)
from scripts.train_sammed3d_t1ce_wt_residual_refiner import (  # noqa: E402
    RefinerDataset,
    write_json_atomic,
)


DEFAULT_VALIDATION_CACHE = Path(
    "workspace/BraTS/cache/t1ce_wt_refiner_graymean_20260813/validation"
)
DEFAULT_CHECKPOINTS = (
    Path(
        "workspace/BraTS/sammed3d_t1ce_wt/"
        "residual_refiner_robustshift4_graymean_paired_a100_20260813/"
        "robustshift4_control/refiner_val_dice_best.pth"
    ),
    Path(
        "workspace/BraTS/sammed3d_t1ce_wt/"
        "residual_refiner_robustshift4_three_seed_ensemble_20260813/"
        "robustshift4_seed20260814/refiner_val_dice_best.pth"
    ),
    Path(
        "workspace/BraTS/sammed3d_t1ce_wt/"
        "residual_refiner_robustshift4_three_seed_ensemble_20260813/"
        "robustshift4_seed20260815/refiner_val_dice_best.pth"
    ),
)
ENSEMBLE_TTA_REPORT = Path(
    "workspace/BraTS/sammed3d_t1ce_wt/"
    "residual_refiner_robustshift4_three_seed_ensemble_20260813/"
    "ENSEMBLE_TTA8_VALIDATION.json"
)
ENSEMBLE_REPORT = Path(
    "workspace/BraTS/sammed3d_t1ce_wt/"
    "residual_refiner_robustshift4_three_seed_ensemble_20260813/"
    "ENSEMBLE_VALIDATION.json"
)


def _fail_if_test_path(path: Path) -> None:
    lowered = str(path.resolve()).lower()
    if "sealed_test" in lowered or "test270" in lowered:
        raise ValueError("test paths are forbidden for validation atom materialization")


def _load_state(path: Path, expected: dict[str, Any]) -> int:
    if not path.exists():
        return 0
    value = json.loads(path.read_text())
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise RuntimeError(f"resume state mismatch for {key}")
    return int(value["completed_cases"])


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    validation_cache = args.validation_cache.resolve()
    output_dir = args.output_dir.resolve()
    _fail_if_test_path(validation_cache)
    _fail_if_test_path(output_dir)
    if (output_dir / "index.json").exists():
        raise FileExistsError("completed atom cache already exists; refusing to overwrite it")
    if (output_dir / "PARITY_FAILED.json").exists():
        raise RuntimeError("prior parity failure found; use a fresh output directory")
    if not np.isfinite(args.parity_tolerance) or args.parity_tolerance < 0:
        raise ValueError("parity tolerance must be a finite nonnegative value")
    if args.dtype == "float32" and args.parity_tolerance > FORMAL_AGGREGATE_PARITY_TOLERANCE:
        raise ValueError("exact float32 aggregate parity tolerance exceeds the frozen maximum")
    if args.dtype == "float16" and args.parity_tolerance > 1e-3:
        raise ValueError("compact float16 aggregate parity tolerance cannot exceed 1e-3")
    if not np.isfinite(args.proxy_per_case_tolerance) or args.proxy_per_case_tolerance < 0:
        raise ValueError("proxy per-case tolerance must be finite and nonnegative")
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_index_path = validation_cache / "index.json"
    validation_index_sha = sha256_file(validation_index_path)
    if validation_index_sha != VALIDATION_CACHE_INDEX_SHA256:
        raise RuntimeError("active validation cache index hash mismatch")
    data = RefinerDataset(validation_cache)
    if data.index.get("split") != "validation" or data.index.get("test_access") is not False:
        raise ValueError("atom materialization requires development-only validation data")
    if len(data) != 108:
        raise ValueError("active atom protocol requires exactly 108 validation scans")
    source_root = data.source
    actual_coarse_sha = sha256_file(validation_cache / "sam_logits.f16")
    actual_images_sha = sha256_file(source_root / "images.f32")
    actual_labels_sha = sha256_file(source_root / "labels.u8")
    if actual_coarse_sha != data.index["logits"]["sha256"]:
        raise RuntimeError("coarse logit blob hash mismatch")
    if actual_images_sha != data.index["source_cache"]["images_sha256"]:
        raise RuntimeError("source image blob hash mismatch")
    if actual_labels_sha != data.index["source_cache"]["labels_sha256"]:
        raise RuntimeError("source label blob hash mismatch")

    checkpoints = [path.resolve() for path in args.checkpoints]
    if len(checkpoints) != 3:
        raise ValueError("the frozen protocol requires exactly three robust checkpoints")
    checkpoint_hashes = tuple(sha256_file(path) for path in checkpoints)
    if checkpoint_hashes != ROBUST_CHECKPOINT_SHA256:
        raise RuntimeError("robust checkpoint allowlist mismatch")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("atom materialization requires a CUDA worker")
    torch.cuda.set_device(device)
    models = [load_model(path, device) for path in checkpoints]
    loader = DataLoader(
        data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    dtype = np.dtype(args.dtype)
    suffix = "f32" if dtype == np.dtype("float32") else "f16"
    atoms_file = f"member_view_logits.{suffix}"
    atoms_path = output_dir / atoms_file
    shape = (len(data), 24, 128, 128, 128)
    atom_order = [
        {
            "atom_index": model_slot * 8 + view_slot,
            "model_slot": model_slot,
            "seed": ROBUST_SEEDS[model_slot],
            "checkpoint_sha256": checkpoint_hashes[model_slot],
            "reflection_axes": list(axes),
            "inverse_reflection_axes": list(axes),
        }
        for model_slot in range(3)
        for view_slot, axes in enumerate(REFLECTION_AXES)
    ]
    runner_sha = sha256_file(Path(__file__).resolve())
    core_sha = sha256_file(ROOT / "sammed3d/t1ce_wt_mcts.py")
    expected_state = {
        "schema": "aflow.sammed3d.t1ce_wt_robust_inference_atoms.build_state",
        "shape": list(shape),
        "dtype": dtype.name,
        "checkpoint_sha256": list(checkpoint_hashes),
        "validation_cache_index_sha256": validation_index_sha,
        "runner_sha256": runner_sha,
        "core_sha256": core_sha,
        "atom_order_sha256": stable_id(atom_order),
        "parity_tolerance": args.parity_tolerance,
        "per_case_tolerance": (
            FORMAL_PER_CASE_PARITY_TOLERANCE
            if dtype == np.dtype("float32")
            else args.proxy_per_case_tolerance
        ),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "autocast": "bfloat16",
    }
    state_path = output_dir / "BUILD_STATE.json"
    completed = _load_state(state_path, expected_state)
    if completed:
        if not atoms_path.exists():
            raise RuntimeError("resume state exists but atom data file is missing")
        atoms = np.memmap(atoms_path, dtype=dtype, mode="r+", shape=shape)
    else:
        if atoms_path.exists():
            raise FileExistsError("atom data exists without a matching resume state")
        atoms = np.memmap(atoms_path, dtype=dtype, mode="w+", shape=shape)
        write_json_atomic(state_path, {**expected_state, "completed_cases": 0, "complete": False})
    if not 0 <= completed <= len(data):
        raise RuntimeError("resume completed_cases is outside the cache bounds")
    expected_bytes = int(np.prod(shape)) * dtype.itemsize
    if atoms_path.stat().st_size != expected_bytes:
        raise RuntimeError("atom cache byte size does not match its declared shape/dtype")

    for case_slot, batch in enumerate(loader):
        if case_slot < completed:
            continue
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        coarse = batch["coarse"].to(device=device, dtype=torch.float32, non_blocking=True)
        for model_slot, model in enumerate(models):
            for view_slot, axes in enumerate(REFLECTION_AXES):
                transformed_image = torch.flip(image, dims=axes) if axes else image
                transformed_coarse = torch.flip(coarse, dims=axes) if axes else coarse
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    transformed_logits = model(transformed_image, transformed_coarse)
                logits = torch.flip(transformed_logits, dims=axes) if axes else transformed_logits
                atoms[case_slot, model_slot * 8 + view_slot] = (
                    logits.float().cpu().numpy()[0, 0].astype(dtype, copy=False)
                )
        atoms.flush()
        write_json_atomic(
            state_path,
            {**expected_state, "completed_cases": case_slot + 1, "complete": False},
        )
        print(f"materialized {case_slot + 1}/{len(data)} cases", flush=True)

    data._open()
    assert data._labels is not None
    evaluator = WorkflowEvaluator(atoms, data._labels, data.case_ids)
    locked = evaluator.evaluate(WorkflowSpec.locked_reference(), range(len(data)))
    tta_report_path = (ROOT / ENSEMBLE_TTA_REPORT).resolve()
    ensemble_report_path = (ROOT / ENSEMBLE_REPORT).resolve()
    if sha256_file(tta_report_path) != ENSEMBLE_TTA_REPORT_SHA256:
        raise RuntimeError("historical ensemble-TTA report hash mismatch")
    if sha256_file(ensemble_report_path) != ENSEMBLE_REPORT_SHA256:
        raise RuntimeError("historical identity-ensemble report hash mismatch")
    historical_tta = json.loads(tta_report_path.read_text())
    historical_identity = json.loads(ensemble_report_path.read_text())
    historical_by_case = {
        item["case_id"]: item for item in historical_tta["per_case"]
    }
    locked_per_case_errors = [
        abs(
            float(actual)
            - float(historical_by_case[case_id]["ensemble_tta_locked_postprocess_dsc"])
        )
        for case_id, actual in zip(data.case_ids, locked["case_dice"], strict=True)
    ]
    raw_tta = evaluator.evaluate(
        WorkflowSpec(threshold_q=0, min_component_voxels=0), range(len(data))
    )
    raw_tta_per_case_errors = [
        abs(
            float(actual)
            - float(historical_by_case[case_id]["ensemble_tta_threshold0_raw_dsc"])
        )
        for case_id, actual in zip(data.case_ids, raw_tta["case_dice"], strict=True)
    ]
    identity_by_case = {
        item["case_id"]: item for item in historical_identity["per_case"]
    }
    identity_ensemble = evaluator.evaluate(
        WorkflowSpec(view_policy="identity", threshold_q=0, min_component_voxels=0),
        range(len(data)),
    )
    identity_ensemble_errors = [
        abs(
            float(actual)
            - float(identity_by_case[case_id]["ensemble_threshold0_raw_dsc"])
        )
        for case_id, actual in zip(
            data.case_ids, identity_ensemble["case_dice"], strict=True
        )
    ]
    identity_seed_errors: list[float] = []
    identity_seed_metrics: list[float] = []
    for model_slot in range(3):
        actual_seed = evaluator.evaluate(
            WorkflowSpec(
                model_mask=1 << model_slot,
                view_policy="identity",
                threshold_q=0,
                min_component_voxels=0,
            ),
            range(len(data)),
        )
        identity_seed_metrics.append(float(actual_seed["scan_macro_dsc"]))
        identity_seed_errors.extend(
            abs(
                float(actual)
                - float(identity_by_case[case_id]["individual_threshold0_raw_dsc"][model_slot])
            )
            for case_id, actual in zip(
                data.case_ids, actual_seed["case_dice"], strict=True
            )
        )
    maximum_per_case_error = max(
        locked_per_case_errors
        + raw_tta_per_case_errors
        + identity_ensemble_errors
        + identity_seed_errors
    )
    tolerance = args.parity_tolerance
    parity = {
        "reference": WorkflowSpec.locked_reference().canonical_dict(),
        "expected_scan_macro_dsc": LOCKED_SCAN_MACRO_DSC,
        "actual_scan_macro_dsc": locked["scan_macro_dsc"],
        "expected_subject_macro_dsc": LOCKED_SUBJECT_MACRO_DSC,
        "actual_subject_macro_dsc": locked["subject_macro_dsc"],
        "aggregate_absolute_tolerance": tolerance,
        "per_case_absolute_tolerance": (
            FORMAL_PER_CASE_PARITY_TOLERANCE
            if dtype == np.dtype("float32")
            else args.proxy_per_case_tolerance
        ),
        "maximum_per_case_absolute_dice_error": maximum_per_case_error,
        "raw_tta_scan_macro_dsc": raw_tta["scan_macro_dsc"],
        "identity_ensemble_scan_macro_dsc": identity_ensemble["scan_macro_dsc"],
        "identity_seed_scan_macro_dsc": identity_seed_metrics,
    }
    parity["passed"] = bool(
        abs(locked["scan_macro_dsc"] - LOCKED_SCAN_MACRO_DSC) <= tolerance
        and abs(locked["subject_macro_dsc"] - LOCKED_SUBJECT_MACRO_DSC) <= tolerance
        and maximum_per_case_error <= parity["per_case_absolute_tolerance"]
    )
    if not parity["passed"]:
        write_json_atomic(output_dir / "PARITY_FAILED.json", parity)
        raise RuntimeError("locked 0.700419 workflow parity gate failed; MCTS cache not published")

    atoms.flush()
    report = {
        "schema": ATOM_CACHE_SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "split": "validation",
        "test_access": False,
        "geometry": "GT-centered N4 128^3 oracle geometry",
        "target": "BraTS 2024 T1ce WT={1,2,3}; label 4 excluded",
        "uses_gray_loss": False,
        "count": len(data),
        "case_ids": data.case_ids,
        "shape": list(shape),
        "dtype": dtype.name,
        "evidence_level": "exact" if dtype == np.dtype("float32") else "proxy_only",
        "atoms_file": atoms_file,
        "atoms_sha256": sha256_file(atoms_path),
        "atom_order": atom_order,
        "checkpoints": [
            {"seed": seed, "path": str(path), "sha256": digest}
            for seed, path, digest in zip(ROBUST_SEEDS, checkpoints, checkpoint_hashes, strict=True)
        ],
        "validation_cache": str(validation_cache),
        "validation_cache_index_sha256": validation_index_sha,
        "coarse_logits_sha256": actual_coarse_sha,
        "source_images_sha256": actual_images_sha,
        "source_labels_sha256": actual_labels_sha,
        "aggregation_contract": "fixed-order float32 fusion after inverse reflection",
        "parity": parity,
        "historical_parity_evidence": {
            "ensemble_tta_report_sha256": ENSEMBLE_TTA_REPORT_SHA256,
            "ensemble_report_sha256": ENSEMBLE_REPORT_SHA256,
            "per_case_passed": True,
        },
        "runner_sha256": runner_sha,
        "core_sha256": core_sha,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "autocast": "bfloat16 model forward; stored dtype selected by CLI",
        "elapsed_seconds": time.time() - started,
    }
    write_json_atomic(output_dir / "index.json", report)
    write_json_atomic(state_path, {**expected_state, "completed_cases": len(data), "complete": True})
    print(json.dumps({"output": str(output_dir), "parity": parity}, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-cache", type=Path, default=DEFAULT_VALIDATION_CACHE)
    parser.add_argument("--checkpoints", type=Path, nargs=3, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--parity-tolerance",
        type=float,
        default=FORMAL_AGGREGATE_PARITY_TOLERANCE,
        help="Use a separately preregistered tolerance if compact float16 atoms are requested.",
    )
    parser.add_argument("--proxy-per-case-tolerance", type=float, default=5e-3)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
