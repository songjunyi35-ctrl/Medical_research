#!/usr/bin/env python
"""Evaluate the frozen upstream-literal checkpoint on the authorized test once.

The inference curve is not reimplemented here: each case calls
``upstream_validation_curve`` from the exact trainer that selected the epoch-21
checkpoint.  Two torchrun ranks process disjoint manifest-order shards and
rank zero creates one audited 270-case report.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d import engine, peft  # noqa: E402
from scripts import build_t1ce_wt_official15_cache as official_cache  # noqa: E402
from scripts import build_t1ce_wt_official15_test_cache as test_cache  # noqa: E402
from scripts import train_sammed3d_t1ce_wt_upstream_literal_va as upstream  # noqa: E402

SCHEMA = "aflow.sammed3d.t1ce_wt_upstream_literal_official15_test"
SCHEMA_VERSION = 1
EXPECTED_CHECKPOINT_SHA256 = "c0c85a0fb90f6301344053f97cabe8a4972bd9febd4c6ae248f3a0908598fe31"
EXPECTED_CHECKPOINT_EPOCH = 21
EXPECTED_VAL_BEST5 = 0.5713061351995669
EXPECTED_PARAMETERS = 100_509_232
EXPECTED_WORLD_SIZE = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{field} mismatch: {actual!r} != {expected!r}")


def validate_training_binding(training_output: Path, checkpoint: Path) -> dict:
    done_path = training_output / "TRAIN_DONE.json"
    best_path = training_output / "best.json"
    reload_path = training_output / "selected_checkpoint_reload.json"
    config_path = training_output / "train_config.json"
    for path in (done_path, best_path, reload_path, config_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    done, best = load_json(done_path), load_json(best_path)
    reloads, config = load_json(reload_path), load_json(config_path)
    require_equal(done.get("completed_epochs"), 200, "completed epochs")
    require_equal(done.get("scientific_result_eligible"), True, "result eligibility")
    require_equal(done.get("test_access"), False, "training test access")
    require_equal(best["epochs"]["val_best5"], EXPECTED_CHECKPOINT_EPOCH, "best epoch")
    require_equal(best["values"]["val_best5"], EXPECTED_VAL_BEST5, "best value")
    selected = reloads["val_best5"]
    require_equal(selected["epoch"], EXPECTED_CHECKPOINT_EPOCH, "reload epoch")
    require_equal(selected["checkpoint_sha256"], EXPECTED_CHECKPOINT_SHA256, "reload SHA")
    require_equal(selected["bitwise_identical"], True, "strict reload identity")
    require_equal(selected["strict_state_load"], True, "strict state load")
    require_equal(selected["total_parameters"], EXPECTED_PARAMETERS, "parameter count")
    require_equal(config["test_access"], False, "config test access")
    require_equal(config["world_size"], 2, "training world size")
    require_equal(
        config["dataset_and_interaction_protocol"]["validation_interaction"]["clicks"],
        upstream.VALIDATION_CLICKS,
        "validation clicks",
    )
    actual_checkpoint_sha = sha256_file(checkpoint)
    require_equal(actual_checkpoint_sha, EXPECTED_CHECKPOINT_SHA256, "checkpoint SHA")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require_equal(payload.get("epoch"), EXPECTED_CHECKPOINT_EPOCH, "checkpoint epoch")
    metadata = payload.get("checkpoint_metadata", {})
    require_equal(metadata.get("epoch"), EXPECTED_CHECKPOINT_EPOCH, "metadata epoch")
    require_equal(metadata.get("test_access"), False, "checkpoint test access")
    require_equal(metadata.get("val_official_best5_dice"), EXPECTED_VAL_BEST5, "metadata value")
    del payload
    return {
        "training_output": str(training_output),
        "train_done_sha256": sha256_file(done_path),
        "best_sha256": sha256_file(best_path),
        "train_config_sha256": sha256_file(config_path),
        "selected_checkpoint_reload_sha256": sha256_file(reload_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_checkpoint_sha,
        "checkpoint_epoch": EXPECTED_CHECKPOINT_EPOCH,
        "selection_metric": "validation mean per-case best Dice over clicks 1..5",
        "selection_value": EXPECTED_VAL_BEST5,
        "selection_used_test": False,
    }


def validate_cache(cache_dir: Path, *, verify_payload_hashes: bool) -> dict:
    index_path = cache_dir / "index.json"
    index = load_json(index_path)
    require_equal(index.get("schema"), official_cache.CACHE_SCHEMA, "cache schema")
    require_equal(index.get("protocol_schema"), official_cache.PROTOCOL_SCHEMA, "protocol")
    require_equal(index.get("manifest_sha256"), test_cache.TEST_MANIFEST_SHA256, "manifest SHA")
    require_equal(index.get("count"), test_cache.TEST_COUNT, "test count")
    require_equal(index.get("split"), "test", "cache split")
    require_equal(index.get("test_access"), True, "cache test access")
    require_equal(index.get("authorization"), test_cache.AUTHORIZATION, "authorization")
    require_equal(index.get("development_overlap"), {
        "case_ids": 0,
        "subject_ids": 0,
        "train_manifest_sha256": test_cache.TRAIN_MANIFEST_SHA256,
        "validation_manifest_sha256": test_cache.VAL_MANIFEST_SHA256,
    }, "development overlap audit")
    case_ids = list(index.get("case_ids", []))
    subject_ids = list(index.get("subject_ids", []))
    if len(case_ids) != test_cache.TEST_COUNT or len(set(case_ids)) != len(case_ids):
        raise RuntimeError("cache case IDs are incomplete or duplicated")
    if len(subject_ids) != test_cache.TEST_COUNT:
        raise RuntimeError("cache subject IDs are incomplete")
    shape = tuple(index.get("model_shape", ()))
    require_equal(shape, official_cache.MODEL_SHAPE, "model shape")
    images_path, labels_path = cache_dir / "images.f32", cache_dir / "labels.u8"
    voxels = int(np.prod(shape))
    require_equal(images_path.stat().st_size, test_cache.TEST_COUNT * voxels * 4, "image bytes")
    require_equal(labels_path.stat().st_size, test_cache.TEST_COUNT * voxels, "label bytes")
    if verify_payload_hashes:
        require_equal(sha256_file(images_path), index["images_sha256"], "images SHA")
        require_equal(sha256_file(labels_path), index["labels_sha256"], "labels SHA")
    return index


def load_model(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    model, adapter_config = peft.load_model_with_adapters(
        checkpoint,
        lambda model_type: engine.sam_model_registry3D[model_type](checkpoint=None),
        device=device,
        default_model_type=engine.MODEL_TYPE,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    model.eval()
    require_equal(sum(parameter.numel() for parameter in model.parameters()), EXPECTED_PARAMETERS, "loaded parameters")
    if peft.adapters_enabled(adapter_config):
        raise RuntimeError("vanilla baseline checkpoint unexpectedly enables adapters")
    return model, adapter_config


def evaluate_shard(
    *,
    rank: int,
    world: int,
    device: torch.device,
    model: torch.nn.Module,
    cache_dir: Path,
    index: dict,
) -> list[dict]:
    shape = tuple(index["model_shape"])
    count = int(index["count"])
    images = np.memmap(cache_dir / "images.f32", dtype=np.float32, mode="r", shape=(count, *shape))
    labels = np.memmap(cache_dir / "labels.u8", dtype=np.uint8, mode="r", shape=(count, *shape))
    records = []
    shard_indices = list(range(rank, count, world))
    saved_numpy_state = np.random.get_state()
    try:
        for completed, case_index in enumerate(shard_indices, start=1):
            case_id = str(index["case_ids"][case_index])
            subject_id = str(index["subject_ids"][case_index])
            np.random.seed(upstream.stable_case_seed(case_id))
            image = torch.from_numpy(np.array(images[case_index], dtype=np.float32))[None, None].to(
                device, non_blocking=False
            )
            target = torch.from_numpy(np.array(labels[case_index], dtype=np.uint8))[None, None].to(
                device=device, dtype=torch.float32, non_blocking=False
            )
            started = time.perf_counter()
            with upstream._autocast(device):
                curve = upstream.upstream_validation_curve(model=model, image=image, target=target)
            elapsed = time.perf_counter() - started
            if len(curve) != upstream.VALIDATION_CLICKS or not all(np.isfinite(curve)):
                raise RuntimeError(f"invalid click curve for {case_id}: {curve}")
            record = {
                "manifest_index": case_index,
                "case_id": case_id,
                "subject_id": subject_id,
                "target_voxels": int(index["per_case_positive_voxels"][case_id]),
                "click_dice": [float(value) for value in curve],
                "best_of_5_dice": float(max(curve)),
                "best_click": int(np.argmax(curve) + 1),
                "final_click_5_dice": float(curve[-1]),
                "inference_seconds": elapsed,
            }
            records.append(record)
            print(
                f"rank={rank} [{completed}/{len(shard_indices)}] {case_id} "
                f"best5={record['best_of_5_dice']:.4f} final5={record['final_click_5_dice']:.4f}",
                flush=True,
            )
    finally:
        np.random.set_state(saved_numpy_state)
    return records


def distribution(values: list[float]) -> dict[str, float | int]:
    ordered = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(values)),
        "mean": float(ordered.mean()),
        "std_population": float(ordered.std()),
        "median": float(np.median(ordered)),
        "q25": float(np.quantile(ordered, 0.25)),
        "q75": float(np.quantile(ordered, 0.75)),
        "min": float(ordered.min()),
        "max": float(ordered.max()),
    }


def subject_balanced(cases: list[dict], field: str, click: int | None = None) -> dict:
    grouped: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        value = case[field] if click is None else case[field][click]
        grouped[str(case["subject_id"])].append(float(value))
    means = [statistics.fmean(values) for values in grouped.values()]
    result = distribution(means)
    result["subjects"] = len(grouped)
    return result


def summarize(cases: list[dict]) -> dict:
    summary = {
        "primary_best_of_5": {
            "scan_macro": distribution([case["best_of_5_dice"] for case in cases]),
            "subject_balanced": subject_balanced(cases, "best_of_5_dice"),
        },
        "final_click_5": {
            "scan_macro": distribution([case["final_click_5_dice"] for case in cases]),
            "subject_balanced": subject_balanced(cases, "final_click_5_dice"),
        },
        "per_click": {},
        "best_click_counts": {
            str(click): sum(case["best_click"] == click for case in cases)
            for click in range(1, upstream.VALIDATION_CLICKS + 1)
        },
    }
    for click in range(upstream.VALIDATION_CLICKS):
        summary["per_click"][str(click + 1)] = {
            "scan_macro": distribution([case["click_dice"][click] for case in cases]),
            "subject_balanced": subject_balanced(cases, "click_dice", click),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--training-output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()
    if args.authorization != test_cache.AUTHORIZATION:
        raise RuntimeError("explicit sealed-test authorization token is required")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    require_equal(world, EXPECTED_WORLD_SIZE, "evaluation world size")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    training_output = args.training_output.resolve()
    started = time.time()
    if rank == 0:
        freeze = load_json(args.freeze.resolve())
        require_equal(freeze.get("status"), "frozen_before_test_nifti_open", "freeze status")
        require_equal(freeze["test"]["manifest_sha256"], test_cache.TEST_MANIFEST_SHA256, "freeze manifest")
        require_equal(freeze["model"]["checkpoint_sha256"], EXPECTED_CHECKPOINT_SHA256, "freeze checkpoint")
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"refusing to reuse non-empty test output: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        training_binding = validate_training_binding(training_output, checkpoint)
        index = validate_cache(cache_dir, verify_payload_hashes=True)
        upstream.validate_upstream_sources()
    else:
        training_binding = None
        index = None
    dist.barrier(device_ids=[local_rank])
    if rank != 0:
        index = validate_cache(cache_dir, verify_payload_hashes=False)
    assert index is not None
    model, adapter_config = load_model(checkpoint, device)
    cases = evaluate_shard(
        rank=rank,
        world=world,
        device=device,
        model=model,
        cache_dir=cache_dir,
        index=index,
    )
    shard_path = output_dir / f"shard_rank{rank}.json"
    write_json_atomic(shard_path, {
        "rank": rank,
        "world_size": world,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "manifest_sha256": test_cache.TEST_MANIFEST_SHA256,
        "cases": cases,
    })
    dist.barrier(device_ids=[local_rank])

    if rank == 0:
        merged = []
        shard_hashes = {}
        for shard_rank in range(world):
            path = output_dir / f"shard_rank{shard_rank}.json"
            shard = load_json(path)
            require_equal(shard["rank"], shard_rank, "shard rank")
            require_equal(shard["checkpoint_sha256"], EXPECTED_CHECKPOINT_SHA256, "shard checkpoint")
            require_equal(shard["manifest_sha256"], test_cache.TEST_MANIFEST_SHA256, "shard manifest")
            merged.extend(shard["cases"])
            shard_hashes[path.name] = sha256_file(path)
        merged.sort(key=lambda case: int(case["manifest_index"]))
        require_equal(len(merged), test_cache.TEST_COUNT, "merged test count")
        require_equal([case["case_id"] for case in merged], index["case_ids"], "manifest order")
        if len({case["case_id"] for case in merged}) != test_cache.TEST_COUNT:
            raise RuntimeError("merged test cases contain duplicates")
        report = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": "complete_270_case_test",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "authorization": test_cache.AUTHORIZATION,
            "freeze": {
                "path": str(args.freeze.resolve()),
                "sha256": sha256_file(args.freeze.resolve()),
            },
            "data": {
                "manifest": str(test_cache.TEST_MANIFEST.resolve()),
                "manifest_sha256": test_cache.TEST_MANIFEST_SHA256,
                "cases": test_cache.TEST_COUNT,
                "subjects": len(set(index["subject_ids"])),
                "development_case_overlap": 0,
                "development_subject_overlap": 0,
                "cache": str(cache_dir),
                "cache_index_sha256": sha256_file(cache_dir / "index.json"),
                "cache_images_sha256": index["images_sha256"],
                "cache_labels_sha256": index["labels_sha256"],
                "preprocessing": index["preprocessing"],
            },
            "model": training_binding,
            "protocol": {
                "input": "T1ce single channel",
                "target": "WT={1,2,3}; label 4 excluded",
                "clicks": 5,
                "clicks_1_2": "upstream RITM current point",
                "clicks_3_5": "upstream uniform XOR current point",
                "prompt_history": "current point only",
                "recurrent_low_resolution_mask": True,
                "prediction_threshold": "sigmoid > 0.5 (raw logit > 0)",
                "case_seed": "SHA256(seed=2023,case_id), identical to validation",
                "precision": "FP16 autocast, identical to validation",
                "primary_metric": "mean per-case max Dice over clicks 1..5",
                "test_used_for_selection_or_adaptation": False,
                "inference_function": "scripts.train_sammed3d_t1ce_wt_upstream_literal_va.upstream_validation_curve",
                "inference_source_sha256": sha256_file(Path(upstream.__file__).resolve()),
            },
            "runtime": {
                "world_size": world,
                "gpus": [torch.cuda.get_device_name(i) for i in range(world)],
                "elapsed_seconds": time.time() - started,
                "shard_sha256": shard_hashes,
            },
            "adapter_config": adapter_config,
            "summary": summarize(merged),
            "cases": merged,
        }
        write_json_atomic(output_dir / "test_report.json", report)
        final = {
            "schema": SCHEMA + ".binding",
            "status": "complete",
            "test_report": str((output_dir / "test_report.json").resolve()),
            "test_report_sha256": sha256_file(output_dir / "test_report.json"),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "manifest_sha256": test_cache.TEST_MANIFEST_SHA256,
            "summary": report["summary"],
        }
        write_json_atomic(output_dir / "TEST_DONE.json", final)
        print(json.dumps(final, indent=2, sort_keys=True), flush=True)
    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
