#!/usr/bin/env python3
"""Evaluate frozen SAM-Med3D with three cumulative GT corrective clicks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_sammed3d_t1ce_zero_shot import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_MANIFEST,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_MANIFEST_SHA256,
    MODEL_SHAPE,
    MODEL_TYPE,
    load_model,
    overlap_metrics,
    read_rows,
    resize_binary,
    write_json_atomic,
)
from sammed3d.t1ce_binary import (  # noqa: E402
    T1ceSupervisedCase,
    load_t1ce_supervised_view,
)


SCHEMA = "aflow.sammed3d.t1ce_corrective3_test"
SCHEMA_VERSION = 1
NUM_CLICKS = 3
DEFAULT_SEED = 20260722


def case_seed(seed: int, case_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{case_id}:residual".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def sample_click(
    target: torch.Tensor,
    prediction: torch.Tensor | None,
    generator: np.random.Generator,
) -> tuple[tuple[int, int, int], int, str]:
    if target.ndim != 3 or target.dtype is not torch.bool:
        raise ValueError("target must be a 3D boolean tensor")
    if prediction is None:
        candidates = torch.nonzero(target, as_tuple=False)
        region = "gt_foreground"
    else:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shapes differ")
        candidates = torch.nonzero(torch.logical_xor(prediction.bool(), target), as_tuple=False)
        region = "error_union"
    if len(candidates) == 0:
        raise RuntimeError("no eligible click voxel remains")
    point_tensor = candidates[int(generator.integers(len(candidates)))]
    point = tuple(int(value) for value in point_tensor.tolist())
    label = int(bool(target[point]))
    if prediction is None and label != 1:
        raise RuntimeError("initial click was not sampled from GT foreground")
    if prediction is not None:
        region = "fn" if label == 1 else "fp"
    return point, label, region


@torch.inference_mode()
def predict_click_curve(
    model: torch.nn.Module,
    image: torch.Tensor,
    target: torch.Tensor,
    *,
    device: torch.device,
    generator: np.random.Generator,
) -> tuple[dict[int, torch.Tensor], list[dict[str, Any]]]:
    if tuple(target.shape) != MODEL_SHAPE or not bool(target.any()):
        raise ValueError("residual target must be nonempty on the 128^3 model grid")
    encoder_input = image.unsqueeze(0).to(device=device, dtype=torch.float32)
    target_cpu = target.bool().cpu()
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        embedding = model.image_encoder(encoder_input)

    points: list[tuple[int, int, int]] = []
    labels: list[int] = []
    records: list[dict[str, Any]] = []
    predictions: dict[int, torch.Tensor] = {}
    prediction: torch.Tensor | None = None
    for click_index in range(1, NUM_CLICKS + 1):
        try:
            point, label, region = sample_click(target_cpu, prediction, generator)
        except RuntimeError:
            if prediction is None:
                raise
            predictions[click_index] = prediction.clone()
            records.append(
                {
                    "click_index": click_index,
                    "status": "skipped_prediction_already_exact",
                    "accumulated_points": len(points),
                }
            )
            continue
        points.append(point)
        labels.append(label)
        point_tensor = torch.tensor(
            points, dtype=torch.float32, device=device
        ).reshape(1, len(points), 3)
        label_tensor = torch.tensor(
            labels, dtype=torch.int64, device=device
        ).reshape(1, len(labels))
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            sparse, dense = model.prompt_encoder(
                points=(point_tensor, label_tensor), boxes=None, masks=None
            )
            low_res, _ = model.mask_decoder(
                image_embeddings=embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            logits = F.interpolate(
                low_res, size=MODEL_SHAPE, mode="trilinear", align_corners=False
            )
        prediction = (logits[0, 0] >= 0).cpu()
        predictions[click_index] = prediction
        records.append(
            {
                "click_index": click_index,
                "status": "sampled",
                "point_model_grid": list(point),
                "label": label,
                "sampled_region": region,
                "accumulated_points": len(points),
                "prediction_voxels": int(prediction.sum()),
            }
        )
    return predictions, records


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for click_index in range(1, NUM_CLICKS + 1):
        key = str(click_index)
        summary[key] = {}
        for grid in ("model_grid", "source_grid"):
            rows = [case["metrics"][key][grid] for case in cases]
            dsc = [float(row["dsc"]) for row in rows]
            iou = [float(row["iou"]) for row in rows]
            summary[key][grid] = {
                "cases": len(rows),
                "dsc_macro_mean": statistics.fmean(dsc),
                "dsc_macro_std_population": statistics.pstdev(dsc),
                "dsc_median": statistics.median(dsc),
                "iou_macro_mean": statistics.fmean(iou),
                "prediction_present_cases": sum(
                    bool(row["prediction_present"]) for row in rows
                ),
            }
        sampled = [
            record
            for case in cases
            for record in case["clicks"]
            if int(record["click_index"]) == click_index
            and record["status"] == "sampled"
        ]
        summary[key]["clicks"] = {
            "sampled": len(sampled),
            "skipped_exact": len(cases) - len(sampled),
            "positive": sum(int(record["label"]) == 1 for record in sampled),
            "negative": sum(int(record["label"]) == 0 for record in sampled),
            "fn": sum(record["sampled_region"] == "fn" for record in sampled),
            "fp": sum(record["sampled_region"] == "fp" for record in sampled),
            "gt_foreground": sum(
                record["sampled_region"] == "gt_foreground" for record in sampled
            ),
        }
    return summary


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows, total_cases = read_rows(
        args.manifest.resolve(), args.num_shards, args.shard_index, args.max_cases
    )
    device = torch.device(args.device)
    model = load_model(args.checkpoint.resolve(), device)
    started = time.monotonic()
    cases: list[dict[str, Any]] = []
    for sequence, row in enumerate(rows, start=1):
        case_started = time.monotonic()
        view = load_t1ce_supervised_view(T1ceSupervisedCase.from_manifest_row(row))
        target = view.target[0].bool()
        predictions, clicks = predict_click_curve(
            model,
            view.image,
            target,
            device=device,
            generator=np.random.default_rng(case_seed(args.seed, str(row["case_id"]))),
        )
        metrics: dict[str, Any] = {}
        for click_index, prediction in predictions.items():
            restored = resize_binary(prediction, view.geometry.original_shape)
            metrics[str(click_index)] = {
                "model_grid": overlap_metrics(prediction, target),
                "source_grid": overlap_metrics(restored, view.source_target[0]),
            }
        cases.append(
            {
                "case_id": row["case_id"],
                "clicks": clicks,
                "metrics": metrics,
                "original_shape": list(view.geometry.original_shape),
                "elapsed_seconds": time.monotonic() - case_started,
            }
        )
        print(
            f"[{sequence}/{len(rows)}] {row['case_id']} "
            f"dsc_c1={metrics['1']['model_grid']['dsc']:.6f} "
            f"dsc_c3={metrics['3']['model_grid']['dsc']:.6f}",
            flush=True,
        )
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "protocol": {
            "modality": "T1ce only (manifest key t1c)",
            "target": "residual tumor = raw BraTS labels {1,2,3}; label 4 excluded",
            "training": "none; untouched pretrained checkpoint",
            "num_clicks": NUM_CLICKS,
            "initial_click": "uniform random GT-foreground voxel; positive",
            "corrective_clicks": (
                "uniform random voxel from prediction XOR GT; FN positive, FP negative"
            ),
            "point_accumulation": True,
            "recurrent_mask_prompt": False,
            "sampling_seed": args.seed,
            "case_seed": "uint64_le(sha256(f'{seed}:{case_id}:residual')[:8])",
            "threshold": "raw_logit >= 0 (sigmoid >= 0.5), fixed",
            "postprocessing": "none",
            "tta": "none",
            "input_geometry": "whole-volume resize to 128^3; no crop",
            "normalization": "source T1ce nonzero population z-score",
            "decoder": "multimask_output=False, token0",
            "metric": "case-macro binary DSC and IoU after each click",
        },
        "bindings": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "manifest_total_cases": total_cases,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "model_type": MODEL_TYPE,
        },
        "shard": {
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "selected_cases": len(rows),
            "max_cases": args.max_cases,
        },
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "elapsed_seconds": time.monotonic() - started,
        },
        "summary": summarize(cases),
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate(args)
    write_json_atomic(args.output, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
