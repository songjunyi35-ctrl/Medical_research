#!/usr/bin/env python3
"""Strict-reload, validation-only automatic CAPG-3D -> SAM evaluation.

The label tensor is never passed to prompt construction or model decoding.  It
is read only after prediction to compute metrics and the diagnostic point-hit
rate.  This makes the prompt path genuinely image-only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_cached import MODEL_SHAPE  # noqa: E402
from sammed3d.t1ce_wt_capg3d import (  # noqa: E402
    AutomaticPromptLocator,
    LocatorOutput,
    automatic_prompt_points,
)
from scripts.evaluate_sammed3d_t1ce_wt_checkpoint import load_model as load_sam  # noqa: E402
from scripts.train_t1ce_wt_capg3d import (  # noqa: E402
    BrainCropDataset,
    CHECKPOINT_SCHEMA,
    dice_per_case,
    sha256_file,
    validate as validate_locator,
    write_json_atomic,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_capg3d_auto_sam_validation"
METHODS = ("locator", "sam_point_only", "sam_dense_only", "sam_point_dense")


def load_locator(path: Path, device: torch.device) -> tuple[AutomaticPromptLocator, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unexpected CAPG-3D checkpoint schema")
    if payload.get("test_access") is not False:
        raise ValueError("test-accessible locator checkpoint is forbidden")
    architecture = payload["architecture"]
    model = AutomaticPromptLocator(
        center_aware=bool(architecture["center_aware"]),
        init_filters=int(architecture["init_filters"]),
        dropout=float(architecture["dropout"]),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval(), payload


def overlap(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    prediction = prediction.bool()
    target = target.bool()
    intersection = int((prediction & target).sum())
    prediction_voxels = int(prediction.sum())
    target_voxels = int(target.sum())
    denominator = prediction_voxels + target_voxels
    return {
        "dsc": 2.0 * intersection / denominator if denominator else 1.0,
        "prediction_voxels": prediction_voxels,
        "target_voxels": target_voxels,
    }


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for method in METHODS:
        values = [float(case["metrics"][method]["dsc"]) for case in cases]
        grouped: dict[str, list[float]] = defaultdict(list)
        for case in cases:
            grouped[str(case["subject_id"])].append(
                float(case["metrics"][method]["dsc"])
            )
        subject_values = [statistics.fmean(items) for items in grouped.values()]
        output[method] = {
            "cases": len(values),
            "subjects": len(subject_values),
            "scan_dsc_mean": statistics.fmean(values),
            "scan_dsc_median": statistics.median(values),
            "scan_dsc_std_population": statistics.pstdev(values),
            "subject_dsc_mean": statistics.fmean(subject_values),
            "subject_dsc_std_population": statistics.pstdev(subject_values),
        }
    output["automatic_point_hit_rate"] = statistics.fmean(
        float(case["automatic_point_hits_gt"]) for case in cases
    )
    return output


def _decode_sam(
    sam: torch.nn.Module,
    embedding: torch.Tensor,
    *,
    points: torch.Tensor | None,
    dense_mask: torch.Tensor | None,
) -> torch.Tensor:
    labels = None
    if points is not None:
        labels = torch.ones(
            (points.shape[0], points.shape[1]),
            device=points.device,
            dtype=torch.int64,
        )
    sparse, dense = sam.prompt_encoder(
        points=(points, labels) if points is not None and labels is not None else None,
        boxes=None,
        masks=dense_mask,
    )
    low_res, _ = sam.mask_decoder(
        image_embeddings=embedding,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    return F.interpolate(
        low_res, size=MODEL_SHAPE, mode="trilinear", align_corners=False
    )


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("automatic SAM evaluation requires CUDA")
    torch.cuda.set_device(device)
    locator_path = args.locator_checkpoint.resolve()
    if args.expected_locator_sha256 and sha256_file(locator_path) != args.expected_locator_sha256:
        raise RuntimeError("locator checkpoint SHA-256 mismatch")
    locator, locator_payload = load_locator(locator_path, device)
    data = BrainCropDataset(
        args.validation_cache, expected_split="validation", limit=args.limit_validation_cases
    )
    reload_batch_size = min(
        int(locator_payload["hyperparameters"]["validation_batch_size"]), len(data)
    )
    reload_loader = DataLoader(
        data,
        batch_size=reload_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    strict_reload = validate_locator(locator, reload_loader, device)
    embedded_validation = locator_payload["validation"]
    reload_difference = (
        float(strict_reload["locator_dsc_mean"])
        - float(embedded_validation["locator_dsc_mean"])
    )
    if abs(reload_difference) > args.reload_tolerance:
        raise RuntimeError(
            f"same-batch locator strict-reload DSC differs by {reload_difference}"
        )
    loader = DataLoader(
        data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    sam, sam_sha256, adapter_config = load_sam(
        args.sam_checkpoint.resolve(),
        device=device,
        expected_sha256=args.expected_sam_sha256,
    )
    cases: list[dict[str, Any]] = []
    started = time.time()
    for index, batch in enumerate(loader, start=1):
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        label = batch["label"][0, 0].bool()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            locator_output: LocatorOutput = locator(image)
            point = automatic_prompt_points(locator_output)
            # Fixed, preregistered bounded transfer of locator logits into the
            # scale expected by SAM's recurrent dense-mask prompt encoder.
            dense_mask = F.interpolate(
                5.0 * torch.tanh(locator_output.segmentation_logits / 5.0),
                # PromptEncoder3D downsamples the external mask twice (4x).
                # Its ``mask_input_size`` attribute misleadingly stores the
                # post-downsampling 8^3 embedding size, so derive the required
                # external 32^3 grid from the 128^3 model input explicitly.
                size=tuple(value // 4 for value in MODEL_SHAPE),
                mode="trilinear",
                align_corners=False,
            )
            embedding = sam.image_encoder(image)
            point_only = _decode_sam(
                sam, embedding, points=point, dense_mask=None
            )
            dense_only = _decode_sam(
                sam, embedding, points=None, dense_mask=dense_mask
            )
            point_dense = _decode_sam(
                sam, embedding, points=point, dense_mask=dense_mask
            )
        locator_prediction = locator_output.segmentation_logits[0, 0].float().cpu() > 0
        predictions = {
            "locator": locator_prediction,
            "sam_point_only": point_only[0, 0].float().cpu() > 0,
            "sam_dense_only": dense_only[0, 0].float().cpu() > 0,
            "sam_point_dense": point_dense[0, 0].float().cpu() > 0,
        }
        z, y, x = (int(value) for value in point[0, 0].long().cpu().tolist())
        case = {
            "case_id": str(batch["case_id"][0]),
            "subject_id": str(batch["subject_id"][0]),
            "automatic_point": [z, y, x],
            "automatic_point_source": (
                "learned_center_heatmap"
                if locator_output.center_logits is not None
                else "WT_segmentation_logit"
            ),
            # Metric-only diagnostic; this value is computed after all model
            # predictions and never feeds either network.
            "automatic_point_hits_gt": bool(label[z, y, x]),
            "metrics": {
                method: overlap(prediction, label)
                for method, prediction in predictions.items()
            },
        }
        cases.append(case)
        print(
            f"[{index}/{len(data)}] {case['case_id']} hit={int(case['automatic_point_hits_gt'])} "
            + " ".join(
                f"{method}={case['metrics'][method]['dsc']:.4f}" for method in METHODS
            ),
            flush=True,
        )

    summary = summarize(cases)
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "arm": locator_payload["arm"],
        "protocol": {
            "input": "T1ce only",
            "target": "WT={1,2,3}; raw label 4 excluded",
            "geometry": "image-nonzero brain crop resized to 128^3; no GT localization",
            "prompt_generation": (
                "one positive point from image-only locator argmax plus optional "
                "bounded locator logit as the initial 32^3 dense mask"
            ),
            "dense_logit_transfer": "5*tanh(locator_segmentation_logit/5)",
            "thresholds": "all segmentation logits > 0; no threshold search",
            "gt_use": "metrics and point-hit diagnostic only; never prompt construction",
        },
        "locator": {
            "checkpoint": str(locator_path),
            "checkpoint_sha256": sha256_file(locator_path),
            "selected_epoch": locator_payload["epoch"],
            "embedded_validation_scan_dsc": embedded_validation["locator_dsc_mean"],
            "strict_reload_validation": strict_reload,
            "strict_reload_batch_size": reload_batch_size,
            "strict_reload_scan_dsc_difference": reload_difference,
            "batch1_scan_dsc_difference_from_embedded": (
                float(summary["locator"]["scan_dsc_mean"])
                - float(embedded_validation["locator_dsc_mean"])
            ),
        },
        "sam": {
            "checkpoint": str(args.sam_checkpoint.resolve()),
            "checkpoint_sha256": sam_sha256,
            "adapter_config": adapter_config,
            "frozen": True,
        },
        "validation_cache": {
            "path": str(args.validation_cache.resolve()),
            "index_sha256": sha256_file(args.validation_cache / "index.json"),
            "cases": len(data),
        },
        "summary": summary,
        "cases": cases,
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_atomic(args.output.resolve(), report)
    print(json.dumps(summary, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--locator-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-locator-sha256", default=None)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sam-sha256", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit-validation-cases", type=int, default=None)
    parser.add_argument("--reload-tolerance", type=float, default=1e-7)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
