#!/usr/bin/env python3
"""Label-free online benchmark for a frozen fixed MCTS inference workflow.

Unlike the atom-cache evaluator, this runner only loads the selected refiner
checkpoints and only executes the selected spatial views. It therefore tests
whether fewer member-view forwards become real refiner-stage wall-time and
model-memory savings. No label blob is opened and no Dice is computed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_mcts import (  # noqa: E402
    REFLECTION_AXES,
    ROBUST_CHECKPOINT_SHA256,
    WorkflowOutput,
    WorkflowSpec,
    selected_atom_indices,
    sha256_file,
    workflow_mask,
)
from sammed3d.t1ce_wt_cached import MODEL_SHAPE  # noqa: E402
from scripts.evaluate_t1ce_wt_residual_refiner_ensemble_validation import (  # noqa: E402
    load_model,
)
from scripts.train_sammed3d_t1ce_wt_residual_refiner import write_json_atomic  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_mcts_online_benchmark"
DEFAULT_CACHE = Path("workspace/BraTS/cache/t1ce_wt_refiner_graymean_20260813/validation")
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


def _fail_if_test_path(path: Path) -> None:
    lowered = str(path.resolve()).lower()
    if "sealed_test" in lowered or "test270" in lowered:
        raise ValueError("test paths are forbidden for deployment development benchmark")


def _spec_from_dict(value: dict[str, Any]) -> WorkflowSpec:
    fields = dict(value)
    fields.pop("schema_version", None)
    return WorkflowSpec(**fields)


def load_workflow(path: Path | None, *, locked_reference: bool) -> tuple[WorkflowSpec, dict[str, Any]]:
    if locked_reference:
        if path is not None:
            raise ValueError("--workflow-json and --locked-reference are mutually exclusive")
        spec = WorkflowSpec.locked_reference()
        return spec, {
            "source": "built-in locked reference",
            "workflow_id": spec.workflow_id,
            "spec": spec.canonical_dict(),
        }
    if path is None:
        raise ValueError("candidate benchmark requires --workflow-json")
    payload = json.loads(path.read_text())
    spec = _spec_from_dict(payload["spec"])
    if payload.get("workflow_id") != spec.workflow_id:
        raise RuntimeError("frozen workflow id does not match its canonical spec")
    if payload.get("test_access") is not False:
        raise ValueError("frozen workflow receipt has an invalid test-access boundary")
    return spec, {
        "source": str(path.resolve()),
        "source_sha256": sha256_file(path),
        "workflow_id": spec.workflow_id,
        "spec": spec.canonical_dict(),
        "selection_rule": payload.get("selection_rule"),
        "status": payload.get("status"),
    }


class LabelFreeInferenceCache:
    """Read image and coarse-logit blobs without opening validation labels."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.index_path = self.root / "index.json"
        self.index = json.loads(self.index_path.read_text())
        if self.index.get("split") != "validation" or self.index.get("test_access") is not False:
            raise ValueError("benchmark requires the development validation inference cache")
        self.count = int(self.index["count"])
        self.case_ids = list(self.index["case_ids"])
        source = Path(self.index["source_cache"]["path"])
        if not source.is_absolute():
            source = ROOT / source
        self.source = source.resolve()
        source_index = json.loads((self.source / "index.json").read_text())
        if source_index["images_sha256"] != self.index["source_cache"]["images_sha256"]:
            raise RuntimeError("source image cache binding mismatch")
        if list(source_index["case_ids"][: self.count]) != self.case_ids:
            raise RuntimeError("source/refiner case ordering mismatch")
        self.images = np.memmap(
            self.source / "images.f32",
            dtype=np.float32,
            mode="r",
            shape=(self.count, *MODEL_SHAPE),
        )
        self.coarse = np.memmap(
            self.root / "sam_logits.f16",
            dtype=np.float16,
            mode="r",
            shape=(self.count, *MODEL_SHAPE),
        )

    def tensors(self, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(np.array(self.images[slot], dtype=np.float32))[None, None]
        coarse = torch.from_numpy(np.array(self.coarse[slot], dtype=np.float32))[None, None]
        return image, coarse


def _fuse_torch(
    predictions: list[torch.Tensor],
    spec: WorkflowSpec,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not predictions:
        raise ValueError("workflow generated no predictions")
    stack = torch.stack([value.float() for value in predictions], dim=0)
    if spec.fusion == "logit_mean":
        fused = stack.mean(dim=0)
    elif spec.fusion == "probability_mean":
        probability = torch.sigmoid(stack).mean(dim=0).clamp_(1e-6, 1.0 - 1e-6)
        fused = torch.logit(probability)
    else:
        ordered = torch.sort(stack, dim=0).values
        middle = len(predictions) // 2
        fused = (
            ordered[middle]
            if len(predictions) % 2
            else (ordered[middle - 1] + ordered[middle]) * 0.5
        )
    uncertainty = None
    if spec.uncertainty_beta_q:
        probabilities = torch.sigmoid(stack)
        uncertainty = torch.sqrt(
            torch.clamp(
                (probabilities * probabilities).mean(dim=0)
                - probabilities.mean(dim=0).square(),
                min=0.0,
            )
        )
    return fused, uncertainty


@torch.inference_mode()
def execute_fixed_workflow(
    *,
    models: dict[int, torch.nn.Module],
    image_cpu: torch.Tensor,
    coarse_cpu: torch.Tensor,
    spec: WorkflowSpec,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    if spec.acquisition != "fixed":
        raise ValueError("the first deployment benchmark supports fixed workflows only")
    image = image_cpu.to(device=device, dtype=torch.float32, non_blocking=False)
    coarse = coarse_cpu.to(device=device, dtype=torch.float32, non_blocking=False)
    predictions: list[torch.Tensor] = []
    atom_indices = selected_atom_indices(spec.model_mask, spec.view_policy)
    for atom_index in atom_indices:
        model_slot, view_slot = divmod(atom_index, 8)
        axes = REFLECTION_AXES[view_slot]
        transformed_image = torch.flip(image, dims=axes) if axes else image
        transformed_coarse = torch.flip(coarse, dims=axes) if axes else coarse
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            transformed_logits = models[model_slot](transformed_image, transformed_coarse)
        logits = torch.flip(transformed_logits, dims=axes) if axes else transformed_logits
        predictions.append(logits.float())
    fused, uncertainty = _fuse_torch(predictions, spec)
    output = WorkflowOutput(
        fused_logits=fused.cpu().numpy()[0, 0],
        voxel_uncertainty=(
            uncertainty.cpu().numpy()[0, 0] if uncertainty is not None else None
        ),
        forward_count=len(atom_indices),
        stop_stage=len(atom_indices),
        case_disagreement=0.0,
    )
    return workflow_mask(output, spec), len(atom_indices)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    for path in (args.validation_cache, args.output):
        _fail_if_test_path(path)
    if args.workflow_json is not None:
        _fail_if_test_path(args.workflow_json)
    if args.case_count < 1 or args.repeats < 1 or args.warmup_cases < 0:
        raise ValueError("case-count/repeats must be positive and warmup nonnegative")
    spec, workflow_receipt = load_workflow(
        args.workflow_json,
        locked_reference=args.locked_reference,
    )
    if spec.acquisition != "fixed":
        raise ValueError("online benchmark currently requires a fixed workflow")
    selected_slots = sorted(
        model_slot for model_slot in range(3) if spec.model_mask & (1 << model_slot)
    )
    if len(args.checkpoints) != 3:
        raise ValueError("the frozen checkpoint list must contain three ordered paths")
    actual_hashes = tuple(sha256_file(path) for path in args.checkpoints)
    if actual_hashes != ROBUST_CHECKPOINT_SHA256:
        raise RuntimeError("robust checkpoint allowlist mismatch")

    data = LabelFreeInferenceCache(args.validation_cache)
    case_count = min(args.case_count, len(data.case_ids))
    warmup_count = min(args.warmup_cases, case_count)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("online deployment benchmark requires CUDA")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    models = {slot: load_model(args.checkpoints[slot], device) for slot in selected_slots}
    torch.cuda.synchronize(device)
    model_memory = int(torch.cuda.memory_allocated(device))

    for slot in range(warmup_count):
        image, coarse = data.tensors(slot)
        execute_fixed_workflow(
            models=models,
            image_cpu=image,
            coarse_cpu=coarse,
            spec=spec,
            device=device,
        )
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    latencies: list[float] = []
    output_voxels: list[int] = []
    measured_ids: list[str] = []
    measured_forwards = 0
    for _ in range(args.repeats):
        for slot in range(case_count):
            image, coarse = data.tensors(slot)
            torch.cuda.synchronize(device)
            case_started = time.perf_counter()
            prediction, forwards = execute_fixed_workflow(
                models=models,
                image_cpu=image,
                coarse_cpu=coarse,
                spec=spec,
                device=device,
            )
            torch.cuda.synchronize(device)
            latencies.append(time.perf_counter() - case_started)
            output_voxels.append(int(np.count_nonzero(prediction)))
            measured_ids.append(data.case_ids[slot])
            measured_forwards += forwards

    elapsed = float(sum(latencies))
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "uses_labels": False,
        "split": "validation inputs only",
        "geometry": "GT-centered N4 128^3 oracle geometry; not automatic localization",
        "scope": (
            "refiner-stage latency including host-to-device copy, selected model-view forwards, "
            "fusion, device-to-host copy and connected-component postprocess; excludes upstream "
            "SAM-Med3D coarse prediction and preprocessing"
        ),
        "workflow": workflow_receipt,
        "selected_model_slots": selected_slots,
        "checkpoint_sha256": list(actual_hashes),
        "validation_cache": str(args.validation_cache.resolve()),
        "validation_cache_index_sha256": sha256_file(args.validation_cache / "index.json"),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "autocast": "bfloat16 model forward; float32 fusion",
        "warmup_cases": warmup_count,
        "case_count": case_count,
        "repeats": args.repeats,
        "measurements": len(latencies),
        "member_view_forwards_per_case": len(selected_atom_indices(spec.model_mask, spec.view_policy)),
        "measured_member_view_forwards": measured_forwards,
        "latency_seconds": {
            "mean": float(statistics.fmean(latencies)),
            "median": float(statistics.median(latencies)),
            "p95": float(np.quantile(np.asarray(latencies, dtype=np.float64), 0.95)),
            "minimum": min(latencies),
            "maximum": max(latencies),
            "sum": elapsed,
        },
        "throughput_cases_per_second": len(latencies) / elapsed,
        "cuda_memory_bytes": {
            "allocated_after_model_load": model_memory,
            "peak_allocated_during_measured_inference": int(torch.cuda.max_memory_allocated(device)),
        },
        "output_sanity": {
            "all_finite_latency": all(math.isfinite(value) and value > 0 for value in latencies),
            "foreground_voxel_min": min(output_voxels),
            "foreground_voxel_max": max(output_voxels),
            "case_ids_sha256": hashlib.sha256(
                "\n".join(measured_ids).encode("utf-8")
            ).hexdigest(),
        },
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "wall_seconds_including_setup": time.time() - started,
    }
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "workflow_id": spec.workflow_id,
                "forwards_per_case": report["member_view_forwards_per_case"],
                "latency_seconds": report["latency_seconds"],
                "throughput_cases_per_second": report["throughput_cases_per_second"],
                "cuda_memory_bytes": report["cuda_memory_bytes"],
            },
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
    parser.add_argument("--checkpoints", type=Path, nargs=3, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--case-count", type=int, default=24)
    parser.add_argument("--warmup-cases", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
