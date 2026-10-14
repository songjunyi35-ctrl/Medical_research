#!/usr/bin/env python3
"""Cache frozen coarse/refined logits and clicks for prompt-local refinement.

This development-only cache is derived from the immutable GT-crop train and
validation caches.  It records the three frozen SAM corrective clicks, the
click-3 raw SAM logit, and the frozen epoch-58 full-resolution refiner logit.
No test split is accepted.
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_cached import MODEL_SHAPE  # noqa: E402
from scripts.cache_sammed3d_t1ce_wt_refiner_inputs import (  # noqa: E402
    predict_final_logits,
)
from scripts.evaluate_sammed3d_t1ce_wt_cached import open_cache  # noqa: E402
from scripts.evaluate_sammed3d_t1ce_wt_checkpoint import (  # noqa: E402
    case_seed,
    load_model,
    overlap_metrics,
    sha256_file,
    write_json_atomic,
)
from scripts.train_sammed3d_t1ce_wt_residual_refiner import (  # noqa: E402
    ResidualRefiner,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_prompt_local_refiner_input_cache"
SCHEMA_VERSION = 1


def load_global_refiner(
    checkpoint: Path,
    expected_sha256: str,
    device: torch.device,
) -> tuple[ResidualRefiner, dict[str, Any], str]:
    got = sha256_file(checkpoint)
    if got != expected_sha256:
        raise RuntimeError(f"global refiner SHA {got} != expected {expected_sha256}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    architecture = dict(payload["architecture"])
    model = ResidualRefiner(
        init_filters=int(architecture["init_filters"]),
        dropout=float(architecture["dropout"]),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload, got


def click_arrays(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    points = np.full((3, 3), -1, dtype=np.int16)
    labels = np.full(3, -1, dtype=np.int8)
    for record in records:
        if record.get("status") != "sampled":
            continue
        position = int(record["click_index"]) - 1
        points[position] = np.asarray(record["point_crop_grid"], dtype=np.int16)
        labels[position] = int(record["label"])
    if np.any(points[0] < 0) or int(labels[0]) != 1:
        raise RuntimeError("the frozen first click must be a positive foreground click")
    return points, labels


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.split not in {"train", "validation"}:
        raise ValueError("prompt-local cache is development-only")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    index, images, labels = open_cache(args.cache_dir.resolve(), args.split)
    all_count = int(index["count"])
    slots = list(range(args.shard_index, all_count, args.shard_count))
    if args.max_cases is not None:
        slots = slots[: int(args.max_cases)]
    if not slots:
        raise ValueError("selected shard is empty")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("prompt-local cache generation requires CUDA")
    sam, sam_sha, adapter_config = load_model(
        args.sam_checkpoint.resolve(),
        device=device,
        expected_sha256=args.expected_sam_sha256,
    )
    global_refiner, global_payload, global_sha = load_global_refiner(
        args.global_refiner_checkpoint.resolve(),
        args.expected_global_refiner_sha256,
        device,
    )

    count = len(slots)
    sam_tmp = output_dir / "sam_logits.f16.tmp"
    refined_tmp = output_dir / "global_refined_logits.f16.tmp"
    sam_path = output_dir / "sam_logits.f16"
    refined_path = output_dir / "global_refined_logits.f16"
    sam_mm = np.memmap(sam_tmp, dtype=np.float16, mode="w+", shape=(count, *MODEL_SHAPE))
    refined_mm = np.memmap(
        refined_tmp, dtype=np.float16, mode="w+", shape=(count, *MODEL_SHAPE)
    )
    points = np.full((count, 3, 3), -1, dtype=np.int16)
    point_labels = np.full((count, 3), -1, dtype=np.int8)
    cases: list[dict[str, Any]] = []
    started = time.time()

    for local_slot, source_slot in enumerate(slots):
        case_id = str(index["case_ids"][source_slot])
        image = torch.from_numpy(np.array(images[source_slot], dtype=np.float32))[None]
        target = torch.from_numpy(np.array(labels[source_slot], dtype=np.uint8)).bool()
        sam_logits, clicks = predict_final_logits(
            sam,
            image,
            target,
            device=device,
            generator=np.random.default_rng(case_seed(args.seed, case_id)),
        )
        point_array, label_array = click_arrays(clicks)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            refined = global_refiner(
                image.unsqueeze(0).to(device=device, dtype=torch.float32),
                sam_logits[None, None].to(device=device, dtype=torch.float32),
            )[0, 0].float().cpu()
        if not bool(torch.isfinite(refined).all()):
            raise RuntimeError(f"non-finite global-refiner output for {case_id}")
        sam_mm[local_slot] = sam_logits.numpy().astype(np.float16)
        refined_mm[local_slot] = refined.numpy().astype(np.float16)
        points[local_slot] = point_array
        point_labels[local_slot] = label_array
        sam_metrics = overlap_metrics(sam_logits > 0, target)
        refined_metrics = overlap_metrics(refined > 0, target)
        cases.append(
            {
                "case_id": case_id,
                "source_slot": source_slot,
                "sam_dsc": float(sam_metrics["dsc"]),
                "global_refined_dsc": float(refined_metrics["dsc"]),
                "target_voxels": int(target.sum()),
                "clicks": clicks,
            }
        )
        print(
            f"[{local_slot + 1}/{count}] {case_id} "
            f"sam={sam_metrics['dsc']:.6f} global={refined_metrics['dsc']:.6f}",
            flush=True,
        )

    sam_mm.flush()
    refined_mm.flush()
    del sam_mm, refined_mm
    os.replace(sam_tmp, sam_path)
    os.replace(refined_tmp, refined_path)
    np.save(output_dir / "click_points.i16.npy", points, allow_pickle=False)
    np.save(output_dir / "click_labels.i8.npy", point_labels, allow_pickle=False)
    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "test_access": False,
        "source_cache": {
            "path": str(args.cache_dir.resolve()),
            "index_sha256": sha256_file(args.cache_dir / "index.json"),
            "manifest_sha256": index["manifest_sha256"],
            "images_sha256": index["images_sha256"],
            "labels_sha256": index["labels_sha256"],
            "preprocessing": index["preprocessing"],
            "count": all_count,
        },
        "shard": {"index": args.shard_index, "count": args.shard_count},
        "source_slots": slots,
        "count": count,
        "case_ids": [case["case_id"] for case in cases],
        "sam": {
            "checkpoint": str(args.sam_checkpoint.resolve()),
            "checkpoint_sha256": sam_sha,
            "adapter_config": adapter_config,
            "seed": args.seed,
            "logits_sha256": sha256_file(sam_path),
        },
        "global_refiner": {
            "checkpoint": str(args.global_refiner_checkpoint.resolve()),
            "checkpoint_sha256": global_sha,
            "epoch": int(global_payload["epoch"]),
            "logits_sha256": sha256_file(refined_path),
        },
        "click_points_sha256": sha256_file(output_dir / "click_points.i16.npy"),
        "click_labels_sha256": sha256_file(output_dir / "click_labels.i8.npy"),
        "sam_dsc_mean": float(np.mean([case["sam_dsc"] for case in cases])),
        "global_refined_dsc_mean": float(
            np.mean([case["global_refined_dsc"] for case in cases])
        ),
        "elapsed_seconds": time.time() - started,
        "cases": cases,
        "generator_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_atomic(output_dir / "index.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sam-sha256", required=True)
    parser.add_argument("--global-refiner-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-global-refiner-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
