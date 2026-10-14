#!/usr/bin/env python
"""One-time two-A100 test of the frozen epoch-58 GT-crop residual refiner."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
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

from scripts import build_t1ce_wt_gtcrop_n4_test_cache as test_cache  # noqa: E402
from sammed3d import peft  # noqa: E402
from scripts.cache_sammed3d_t1ce_wt_refiner_inputs import predict_final_logits  # noqa: E402
from scripts.evaluate_sammed3d_t1ce_corrective3 import case_seed  # noqa: E402
from scripts.evaluate_sammed3d_t1ce_wt_checkpoint import (  # noqa: E402
    load_model,
    overlap_metrics,
)
from scripts.train_sammed3d_t1ce_wt_residual_refiner import ResidualRefiner  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_residual_refiner_test"
SCHEMA_VERSION = 1
EXPECTED_WORLD_SIZE = 2
EXPECTED_BASE_SHA256 = "42f96a0a2ade9d98d9db897a11a5a285f7a6351a7693c6884d6ac97a29dfde52"
EXPECTED_REFINER_SHA256 = "4513ea4544ae153064575ec2ed5c56e1b78adf5e89e9d094754227b3bd429f21"
EXPECTED_REFINER_EPOCH = 58
EXPECTED_VALIDATION_DSC = 0.6683989912099032
EXPECTED_VALIDATION_COARSE_DSC = 0.6359485293721298
EXPECTED_SEED = 20260723
MODEL_SHAPE = (128, 128, 128)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{field} mismatch: {actual!r} != {expected!r}")


def validate_freeze(path: Path) -> dict:
    freeze = load_json(path)
    require_equal(freeze.get("status"), "frozen_before_test_nifti_open", "freeze status")
    require_equal(freeze["authorization"]["token"], test_cache.AUTHORIZATION, "authorization")
    require_equal(freeze["test"]["manifest_sha256"], test_cache.TEST_MANIFEST_SHA256, "manifest")
    require_equal(freeze["test"]["cases"], test_cache.TEST_COUNT, "test cases")
    require_equal(freeze["test"]["subjects"], test_cache.TEST_SUBJECTS, "test subjects")
    require_equal(freeze["models"]["base_checkpoint_sha256"], EXPECTED_BASE_SHA256, "base SHA")
    require_equal(freeze["models"]["refiner_checkpoint_sha256"], EXPECTED_REFINER_SHA256, "refiner SHA")
    require_equal(freeze["models"]["refiner_epoch"], EXPECTED_REFINER_EPOCH, "refiner epoch")
    require_equal(freeze["models"]["selection_used_test"], False, "selection test access")
    require_equal(freeze["inference"]["world_size"], EXPECTED_WORLD_SIZE, "world size")
    require_equal(freeze["inference"]["seed"], EXPECTED_SEED, "seed")
    require_equal(freeze["inference"]["test_used_for_adaptation"], False, "test adaptation")
    require_equal(
        freeze["implementation"]["test_evaluator_sha256"],
        sha256_file(Path(__file__).resolve()),
        "evaluator SHA",
    )
    require_equal(
        freeze["implementation"]["test_cache_builder_sha256"],
        sha256_file(Path(test_cache.__file__).resolve()),
        "cache builder SHA",
    )
    return freeze


def validate_cache(cache_dir: Path, *, payload_hashes: bool) -> tuple[dict, list[dict]]:
    index = load_json(cache_dir / "index.json")
    raw_rows, _ = test_cache.read_and_audit()
    require_equal(index.get("schema"), "aflow.sammed3d.t1ce_wt_preprocess_cache", "cache schema")
    require_equal(index.get("manifest_sha256"), test_cache.TEST_MANIFEST_SHA256, "cache manifest")
    require_equal(index.get("count"), test_cache.TEST_COUNT, "cache count")
    require_equal(index.get("split"), "test", "cache split")
    require_equal(index.get("test_access"), True, "cache test access")
    require_equal(index.get("authorization"), test_cache.AUTHORIZATION, "cache authorization")
    require_equal(index.get("development_overlap"), {"case_ids": 0, "subject_ids": 0}, "cache overlap")
    preprocessing = index["preprocessing"]
    expected_preprocessing = {
        "spatial": "gt_crop",
        "normalization": "post-crop positive-voxel sample z-score",
        "uses_gt_for_localization": True,
        "n4_bias_field_correction": True,
        "n4_shrink_factor": 2,
    }
    for key, expected in expected_preprocessing.items():
        require_equal(preprocessing.get(key), expected, f"preprocessing {key}")
    case_ids = [str(row["case_id"]) for row in raw_rows]
    subject_ids = [str(row["subject_id"]) for row in raw_rows]
    require_equal(index.get("case_ids"), case_ids, "cache case order")
    require_equal(index.get("subject_ids"), subject_ids, "cache subject order")
    images_path, labels_path = cache_dir / "images.f32", cache_dir / "labels.u8"
    voxels = int(np.prod(MODEL_SHAPE))
    require_equal(images_path.stat().st_size, test_cache.TEST_COUNT * voxels * 4, "image bytes")
    require_equal(labels_path.stat().st_size, test_cache.TEST_COUNT * voxels, "label bytes")
    if payload_hashes:
        require_equal(sha256_file(images_path), index["images_sha256"], "image payload SHA")
        require_equal(sha256_file(labels_path), index["labels_sha256"], "label payload SHA")
    return index, raw_rows


def validate_model_binding(base_checkpoint: Path, refiner_checkpoint: Path) -> dict:
    base_sha = sha256_file(base_checkpoint)
    refiner_sha = sha256_file(refiner_checkpoint)
    require_equal(base_sha, EXPECTED_BASE_SHA256, "base checkpoint SHA")
    require_equal(refiner_sha, EXPECTED_REFINER_SHA256, "refiner checkpoint SHA")
    payload = torch.load(refiner_checkpoint, map_location="cpu", weights_only=False)
    require_equal(payload.get("schema"), "aflow.sammed3d.t1ce_wt_residual_refiner_checkpoint", "refiner schema")
    require_equal(payload.get("epoch"), EXPECTED_REFINER_EPOCH, "refiner epoch")
    require_equal(payload["validation"]["refined_dsc_mean"], EXPECTED_VALIDATION_DSC, "selected validation DSC")
    require_equal(payload["validation"]["coarse_dsc_mean"], EXPECTED_VALIDATION_COARSE_DSC, "validation coarse DSC")
    architecture = payload["architecture"]
    require_equal(architecture["init_filters"], 16, "init filters")
    require_equal(architecture["dropout"], 0.1, "dropout")
    strict_model = ResidualRefiner(init_filters=16, dropout=0.1)
    strict_model.load_state_dict(payload["model_state_dict"], strict=True)
    del strict_model, payload
    result_path = ROOT / "workspace/BraTS/sammed3d_t1ce_wt/RESIDUAL_REFINER_RESULT_20260810.json"
    result = load_json(result_path)
    require_equal(result.get("test_access"), False, "development result test access")
    require_equal(result["selected_result"]["epoch"], EXPECTED_REFINER_EPOCH, "result epoch")
    require_equal(result["selected_result"]["validation_scan_macro_dsc"], EXPECTED_VALIDATION_DSC, "result DSC")
    require_equal(result["selected_result"]["checkpoint_sha256"], EXPECTED_REFINER_SHA256, "result checkpoint")
    return {
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": base_sha,
        "refiner_checkpoint": str(refiner_checkpoint),
        "refiner_checkpoint_sha256": refiner_sha,
        "refiner_epoch": EXPECTED_REFINER_EPOCH,
        "selection_metric": "108-case validation scan-macro refined DSC",
        "selection_value": EXPECTED_VALIDATION_DSC,
        "validation_coarse_dsc": EXPECTED_VALIDATION_COARSE_DSC,
        "selection_used_test": False,
        "development_result": str(result_path),
        "development_result_sha256": sha256_file(result_path),
    }


def load_refiner(checkpoint: Path, device: torch.device) -> ResidualRefiner:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ResidualRefiner(init_filters=16, dropout=0.1)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _shard_worker(
    rank: int,
    world: int,
    cache_dir_string: str,
    output_dir_string: str,
    base_checkpoint_string: str,
    refiner_checkpoint_string: str,
) -> None:
    cache_dir = Path(cache_dir_string)
    output_dir = Path(output_dir_string)
    base_checkpoint = Path(base_checkpoint_string)
    refiner_checkpoint = Path(refiner_checkpoint_string)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)
    index, rows = validate_cache(cache_dir, payload_hashes=False)
    base, base_sha, adapter_config = load_model(
        base_checkpoint,
        device=device,
        expected_sha256=EXPECTED_BASE_SHA256,
    )
    if peft.adapters_enabled(adapter_config):
        raise RuntimeError("base checkpoint unexpectedly enables adapters")
    refiner = load_refiner(refiner_checkpoint, device)
    count = int(index["count"])
    images = np.memmap(cache_dir / "images.f32", dtype=np.float32, mode="r", shape=(count, *MODEL_SHAPE))
    labels = np.memmap(cache_dir / "labels.u8", dtype=np.uint8, mode="r", shape=(count, *MODEL_SHAPE))
    records: list[dict] = []
    shard_indices = list(range(rank, count, world))
    for completed, case_index in enumerate(shard_indices, start=1):
        case_id = str(index["case_ids"][case_index])
        subject_id = str(rows[case_index]["subject_id"])
        image_cpu = torch.from_numpy(np.array(images[case_index], dtype=np.float32))[None]
        target_cpu = torch.from_numpy(np.array(labels[case_index], dtype=np.uint8)).bool()
        started = time.perf_counter()
        coarse_cpu, clicks = predict_final_logits(
            base,
            image_cpu,
            target_cpu,
            device=device,
            generator=np.random.default_rng(case_seed(EXPECTED_SEED, case_id)),
        )
        image = image_cpu[None].to(device=device, dtype=torch.float32)
        coarse = coarse_cpu[None, None].to(device=device, dtype=torch.float32)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            refined = refiner(image, coarse).float()[0, 0].cpu()
        coarse_metrics = overlap_metrics(coarse_cpu > 0, target_cpu)
        refined_metrics = overlap_metrics(refined > 0, target_cpu)
        record = {
            "manifest_index": case_index,
            "case_id": case_id,
            "subject_id": subject_id,
            "target_voxels": int(target_cpu.sum()),
            "coarse": coarse_metrics,
            "refined": refined_metrics,
            "delta_dsc": float(refined_metrics["dsc"] - coarse_metrics["dsc"]),
            "clicks": clicks,
            "inference_seconds": time.perf_counter() - started,
        }
        records.append(record)
        print(
            f"rank={rank} [{completed}/{len(shard_indices)}] {case_id} "
            f"coarse={coarse_metrics['dsc']:.4f} refined={refined_metrics['dsc']:.4f} "
            f"delta={record['delta_dsc']:+.4f}",
            flush=True,
        )
    write_json_atomic(
        output_dir / f"shard_rank{rank}.json",
        {
            "rank": rank,
            "world_size": world,
            "base_checkpoint_sha256": base_sha,
            "refiner_checkpoint_sha256": EXPECTED_REFINER_SHA256,
            "manifest_sha256": test_cache.TEST_MANIFEST_SHA256,
            "gpu": torch.cuda.get_device_name(rank),
            "cases": records,
        },
    )


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "mean": float(array.mean()),
        "std_population": float(array.std()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def subject_distribution(cases: list[dict], field: str) -> dict[str, float | int]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        value = case[field]["dsc"] if field in {"coarse", "refined"} else case[field]
        grouped[str(case["subject_id"])].append(float(value))
    result = distribution([statistics.fmean(values) for values in grouped.values()])
    result["subjects"] = len(grouped)
    return result


def summarize(cases: list[dict]) -> dict:
    coarse = [float(case["coarse"]["dsc"]) for case in cases]
    refined = [float(case["refined"]["dsc"]) for case in cases]
    deltas = [float(case["delta_dsc"]) for case in cases]
    bins = [
        ("<8000", 0, 8000),
        ("8000-19999", 8000, 20000),
        ("20000-39999", 20000, 40000),
        (">=40000", 40000, None),
    ]
    stratified = []
    for name, low, high in bins:
        selected = [
            case for case in cases
            if case["target_voxels"] >= low and (high is None or case["target_voxels"] < high)
        ]
        stratified.append(
            {
                "bin": name,
                "cases": len(selected),
                "coarse_dsc": float(np.mean([case["coarse"]["dsc"] for case in selected])) if selected else None,
                "refined_dsc": float(np.mean([case["refined"]["dsc"] for case in selected])) if selected else None,
                "gain": float(np.mean([case["delta_dsc"] for case in selected])) if selected else None,
                "improved_cases": sum(case["delta_dsc"] > 0 for case in selected),
            }
        )
    return {
        "scan_macro": {
            "coarse": distribution(coarse),
            "refined": distribution(refined),
            "paired_delta": distribution(deltas),
            "improved_cases": sum(delta > 0 for delta in deltas),
            "worsened_cases": sum(delta < 0 for delta in deltas),
            "unchanged_cases": sum(delta == 0 for delta in deltas),
        },
        "subject_balanced": {
            "coarse": subject_distribution(cases, "coarse"),
            "refined": subject_distribution(cases, "refined"),
            "paired_delta": subject_distribution(cases, "delta_dsc"),
        },
        "by_gt_crop_target_voxels": stratified,
    }


def run(args: argparse.Namespace) -> dict:
    if args.authorization != test_cache.AUTHORIZATION:
        raise RuntimeError("explicit residual-refiner test authorization is required")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    freeze = validate_freeze(args.freeze.resolve())
    cache_index, _ = validate_cache(args.cache_dir.resolve(), payload_hashes=True)
    model_binding = validate_model_binding(
        args.base_checkpoint.resolve(), args.refiner_checkpoint.resolve()
    )
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_shard_worker,
            args=(
                rank,
                EXPECTED_WORLD_SIZE,
                str(args.cache_dir.resolve()),
                str(output_dir),
                str(args.base_checkpoint.resolve()),
                str(args.refiner_checkpoint.resolve()),
            ),
        )
        for rank in range(EXPECTED_WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failures = [(rank, process.exitcode) for rank, process in enumerate(processes) if process.exitcode != 0]
    if failures:
        raise RuntimeError(f"evaluation shard failures: {failures}")
    merged: list[dict] = []
    shard_hashes: dict[str, str] = {}
    gpus: list[str] = []
    for rank in range(EXPECTED_WORLD_SIZE):
        path = output_dir / f"shard_rank{rank}.json"
        shard = load_json(path)
        require_equal(shard["rank"], rank, "shard rank")
        require_equal(shard["world_size"], EXPECTED_WORLD_SIZE, "shard world")
        require_equal(shard["base_checkpoint_sha256"], EXPECTED_BASE_SHA256, "shard base")
        require_equal(shard["refiner_checkpoint_sha256"], EXPECTED_REFINER_SHA256, "shard refiner")
        require_equal(shard["manifest_sha256"], test_cache.TEST_MANIFEST_SHA256, "shard manifest")
        merged.extend(shard["cases"])
        gpus.append(str(shard["gpu"]))
        shard_hashes[path.name] = sha256_file(path)
    merged.sort(key=lambda case: int(case["manifest_index"]))
    require_equal(len(merged), test_cache.TEST_COUNT, "merged case count")
    require_equal([case["case_id"] for case in merged], cache_index["case_ids"], "merged order")
    if len({case["case_id"] for case in merged}) != test_cache.TEST_COUNT:
        raise RuntimeError("merged cases contain duplicates")
    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete_270_case_test",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": test_cache.AUTHORIZATION,
        "freeze": {"path": str(args.freeze.resolve()), "sha256": sha256_file(args.freeze.resolve())},
        "data": {
            "manifest": str(test_cache.TEST_MANIFEST),
            "manifest_sha256": test_cache.TEST_MANIFEST_SHA256,
            "cases": test_cache.TEST_COUNT,
            "subjects": test_cache.TEST_SUBJECTS,
            "development_case_overlap": 0,
            "development_subject_overlap": 0,
            "cache": str(args.cache_dir.resolve()),
            "cache_index_sha256": sha256_file(args.cache_dir.resolve() / "index.json"),
            "cache_images_sha256": cache_index["images_sha256"],
            "cache_labels_sha256": cache_index["labels_sha256"],
            "preprocessing": cache_index["preprocessing"],
        },
        "models": model_binding,
        "protocol": freeze["inference"],
        "runtime": {
            "world_size": EXPECTED_WORLD_SIZE,
            "gpus": gpus,
            "elapsed_seconds": time.time() - started,
            "shard_sha256": shard_hashes,
        },
        "summary": summarize(merged),
        "evidence_boundary": [
            "This is a GT-centered/oracle-localization test and is not an automatic end-to-end score.",
            "The refiner was selected only on the 108-case validation split; test was not used for selection or adaptation.",
            "The 270-case internal test has been revealed and consumed by earlier project arms; this is not pristine external validation.",
            "No threshold, TTA, morphology, connected-component, or postprocessing search was run.",
        ],
        "cases": merged,
    }
    report_path = output_dir / "test_report.json"
    write_json_atomic(report_path, report)
    done = {
        "schema": SCHEMA + ".binding",
        "status": "complete",
        "test_report": str(report_path),
        "test_report_sha256": sha256_file(report_path),
        "base_checkpoint_sha256": EXPECTED_BASE_SHA256,
        "refiner_checkpoint_sha256": EXPECTED_REFINER_SHA256,
        "manifest_sha256": test_cache.TEST_MANIFEST_SHA256,
        "summary": report["summary"],
    }
    write_json_atomic(output_dir / "TEST_DONE.json", done)
    print(json.dumps(done, indent=2, sort_keys=True), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--refiner-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--authorization", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
