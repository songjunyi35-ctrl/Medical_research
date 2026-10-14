#!/usr/bin/env python3
"""Cache frozen stage-2 SAM-Med3D click-3 logits for a residual refiner.

This development-only cache keeps the frozen T1ce/WT main-line contract:
T1ce only, WT={1,2,3}, three accumulated corrective clicks, recurrent low-
resolution mask prompt, and raw-logit threshold zero.  It cannot open test
data.  Images and labels remain in the immutable source cache; this script
writes only the final click-3 raw logits and a provenance index.
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
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_cached import MODEL_SHAPE  # noqa: E402
from scripts.evaluate_sammed3d_t1ce_corrective3 import sample_click  # noqa: E402
from scripts.evaluate_sammed3d_t1ce_wt_cached import open_cache  # noqa: E402
from scripts.evaluate_sammed3d_t1ce_wt_checkpoint import (  # noqa: E402
    case_seed,
    load_model,
    overlap_metrics,
    sha256_file,
    write_json_atomic,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_refiner_input_cache"
SCHEMA_VERSION = 1
NUM_CLICKS = 3


def portable_repo_path(path: Path) -> str:
    """Record repo-contained assets without machine-specific absolute paths."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


@torch.inference_mode()
def predict_final_logits(
    model: torch.nn.Module,
    image: torch.Tensor,
    target: torch.Tensor,
    *,
    device: torch.device,
    generator: np.random.Generator,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Run the frozen accumulated-click protocol and return click-3 logits."""
    encoder_input = image.unsqueeze(0).to(device=device, dtype=torch.float32)
    target_cpu = target.bool().cpu()
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        embedding = model.image_encoder(encoder_input)
    low_res = torch.zeros((1, 1, 32, 32, 32), device=device)
    points: list[tuple[int, int, int]] = []
    labels: list[int] = []
    prediction: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    records: list[dict[str, Any]] = []
    for click_index in range(1, NUM_CLICKS + 1):
        try:
            point, label, region = sample_click(target_cpu, prediction, generator)
        except RuntimeError:
            if prediction is None or logits is None:
                raise
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
                points=(point_tensor, label_tensor), boxes=None, masks=low_res
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
        prediction = (logits[0, 0] > 0).cpu()
        records.append(
            {
                "click_index": click_index,
                "status": "sampled",
                "point_crop_grid": list(point),
                "label": int(label),
                "sampled_region": region,
                "accumulated_points": len(points),
                "points_passed_to_encoder": len(points),
                "recurrent_mask_prompt": True,
            }
        )
    if logits is None:
        raise RuntimeError("click protocol produced no logits")
    result = logits[0, 0].float().cpu()
    if tuple(result.shape) != MODEL_SHAPE or not bool(torch.isfinite(result).all()):
        raise RuntimeError("non-finite or malformed final logits")
    return result, records


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.split not in {"train", "validation"}:
        raise ValueError("refiner input cache is development-only")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "sam_logits.f16"
    temporary_path = output_dir / "sam_logits.f16.tmp"
    index, images, labels = open_cache(args.cache_dir.resolve(), args.split)
    count = int(index["count"])
    if args.max_cases is not None:
        if args.max_cases <= 0:
            raise ValueError("max-cases must be positive")
        count = min(count, int(args.max_cases))
    case_ids = list(index["case_ids"][:count])

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SAM logit caching requires CUDA")
    model, checkpoint_sha256, adapter_config = load_model(
        args.checkpoint.resolve(),
        device=device,
        expected_sha256=args.expected_checkpoint_sha256,
    )
    logits_mm = np.memmap(
        temporary_path,
        dtype=np.float16,
        mode="w+",
        shape=(count, *MODEL_SHAPE),
    )
    cases: list[dict[str, Any]] = []
    started = time.time()
    for slot, case_id in enumerate(case_ids):
        image = torch.from_numpy(np.array(images[slot], dtype=np.float32))[None]
        target = torch.from_numpy(np.array(labels[slot], dtype=np.uint8)).bool()
        logits, clicks = predict_final_logits(
            model,
            image,
            target,
            device=device,
            generator=np.random.default_rng(case_seed(args.seed, case_id)),
        )
        logits_mm[slot] = logits.numpy().astype(np.float16)
        metrics = overlap_metrics(logits > 0, target)
        cases.append({"case_id": case_id, "metrics": metrics, "clicks": clicks})
        print(
            f"[{slot + 1}/{count}] {case_id} click3_dsc={metrics['dsc']:.6f}",
            flush=True,
        )
    logits_mm.flush()
    del logits_mm
    os.replace(temporary_path, output_path)
    values = [float(case["metrics"]["dsc"]) for case in cases]
    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "test_access": False,
        "protocol": {
            "input": "T1ce single channel",
            "target": "WT = raw labels {1,2,3}; raw label 4 excluded",
            "geometry": index["preprocessing"]["spatial"],
            "num_clicks": NUM_CLICKS,
            "clicks": "accumulated GT-derived corrective clicks",
            "mask_prompt": "previous low-resolution mask recurrently supplied",
            "threshold": "raw logit > 0",
        },
        "source_cache": {
            "path": portable_repo_path(args.cache_dir),
            "manifest_sha256": index["manifest_sha256"],
            "images_sha256": index["images_sha256"],
            "labels_sha256": index["labels_sha256"],
            "preprocessing": index["preprocessing"],
        },
        "model": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "adapter_config": adapter_config,
        },
        "seed": int(args.seed),
        "count": count,
        "case_ids": case_ids,
        "logits": {
            "path": portable_repo_path(output_path),
            "dtype": "float16",
            "shape": [count, *MODEL_SHAPE],
            "sha256": sha256_file(output_path),
        },
        "baseline_click3": {
            "dsc_mean": float(np.mean(values)),
            "dsc_median": float(np.median(values)),
        },
        "elapsed_seconds": time.time() - started,
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "cases": cases,
    }
    write_json_atomic(output_dir / "index.json", report)
    print(json.dumps(report["baseline_click3"], indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--max-cases", type=int, default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
