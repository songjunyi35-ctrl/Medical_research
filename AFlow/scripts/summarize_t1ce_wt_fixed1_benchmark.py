#!/usr/bin/env python3
"""Summarize the preregistered two-A100 fixed1/locked cross-over benchmark."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_mcts import sha256_file  # noqa: E402
from scripts.train_sammed3d_t1ce_wt_residual_refiner import write_json_atomic  # noqa: E402


DEFAULT_ROOT = Path(
    "workspace/BraTS/sammed3d_t1ce_wt/adaptive_router_20260814/fixed1_benchmark"
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "FINAL_FIXED1_DEPLOYMENT_BENCHMARK.json"
DEFAULT_PREREG = Path("FIXED1_BENCHMARK_PREREG_20260814.md")
DEFAULT_CANDIDATE = Path("FIXED1_DEPLOYMENT_CANDIDATE_20260814.json")


def run(args: argparse.Namespace) -> dict:
    candidate_paths = [
        args.input_dir / "FIXED1_GPU0_RUN1.json",
        args.input_dir / "FIXED1_GPU1_RUN2.json",
    ]
    reference_paths = [
        args.input_dir / "LOCKED_GPU1_RUN1.json",
        args.input_dir / "LOCKED_GPU0_RUN2.json",
    ]
    candidate_parity_path = args.input_dir / "FIXED1_ONLINE_PARITY.json"
    locked_parity_path = args.input_dir / "LOCKED_ONLINE_PARITY.json"
    candidate = [json.loads(path.read_text()) for path in candidate_paths]
    reference = [json.loads(path.read_text()) for path in reference_paths]
    candidate_parity = json.loads(candidate_parity_path.read_text())
    locked_parity = json.loads(locked_parity_path.read_text())
    receipt = json.loads(args.candidate.read_text())
    all_benchmarks = candidate + reference
    if any(
        value.get("test_access") is not False or value.get("uses_labels") is not False
        for value in all_benchmarks
    ):
        raise RuntimeError("benchmark receipt violated its label/test boundary")
    if {value["device"] for value in all_benchmarks} != {"NVIDIA A100-SXM4-80GB"}:
        raise RuntimeError("cross-over benchmark device mismatch")
    if {value["member_view_forwards_per_case"] for value in candidate} != {1}:
        raise RuntimeError("fixed1 candidate did not execute exactly one member view")
    if {value["member_view_forwards_per_case"] for value in reference} != {24}:
        raise RuntimeError("reference did not execute exactly 24 member views")
    if any(value["measurements"] != 48 for value in all_benchmarks):
        raise RuntimeError("cross-over benchmark requires 48 measurements per receipt")
    if {value["workflow"]["workflow_id"] for value in candidate} != {
        receipt["workflow_id"]
    }:
        raise RuntimeError("fixed1 benchmark workflow binding mismatch")
    for parity, expected_forwards in ((candidate_parity, 1), (locked_parity, 24)):
        if (
            parity.get("test_access") is not False
            or parity.get("uses_labels") is not False
            or parity.get("cases") != 8
            or parity.get("identical_cases") != 8
            or parity.get("minimum_mask_dice") != 1.0
            or parity.get("maximum_different_voxels") != 0
            or {value["member_view_forwards"] for value in parity["per_case"]}
            != {expected_forwards}
        ):
            raise RuntimeError("online/exact atom parity failed")

    candidate_mean = statistics.fmean(
        value["latency_seconds"]["mean"] for value in candidate
    )
    reference_mean = statistics.fmean(
        value["latency_seconds"]["mean"] for value in reference
    )
    pair_speedups = [
        reference_value["latency_seconds"]["mean"]
        / candidate_value["latency_seconds"]["mean"]
        for candidate_value, reference_value in zip(candidate, reference, strict=True)
    ]
    candidate_model_memory = candidate[0]["cuda_memory_bytes"][
        "allocated_after_model_load"
    ]
    reference_model_memory = reference[0]["cuda_memory_bytes"][
        "allocated_after_model_load"
    ]
    candidate_peak = candidate[0]["cuda_memory_bytes"][
        "peak_allocated_during_measured_inference"
    ]
    reference_peak = reference[0]["cuda_memory_bytes"][
        "peak_allocated_during_measured_inference"
    ]
    report = {
        "schema": "aflow.sammed3d.t1ce_wt_fixed1_deployment_benchmark.summary",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "uses_labels": False,
        "device": "NVIDIA A100-SXM4-80GB",
        "design": "two-run cross-over: fixed1/locked swap cuda:0 and cuda:1",
        "scope": candidate[0]["scope"],
        "candidate": {
            "workflow_id": candidate[0]["workflow"]["workflow_id"],
            "member_view_forwards": 1,
            "measurements": sum(value["measurements"] for value in candidate),
            "mean_latency_seconds": candidate_mean,
            "throughput_cases_per_second": 1.0 / candidate_mean,
            "allocated_after_model_load_bytes": candidate_model_memory,
            "peak_allocated_bytes": candidate_peak,
        },
        "locked_reference": {
            "workflow_id": reference[0]["workflow"]["workflow_id"],
            "member_view_forwards": 24,
            "measurements": sum(value["measurements"] for value in reference),
            "mean_latency_seconds": reference_mean,
            "throughput_cases_per_second": 1.0 / reference_mean,
            "allocated_after_model_load_bytes": reference_model_memory,
            "peak_allocated_bytes": reference_peak,
        },
        "comparison": {
            "member_view_forward_reduction_fraction": 1.0 - 1.0 / 24.0,
            "latency_reduction_fraction": 1.0 - candidate_mean / reference_mean,
            "speedup": reference_mean / candidate_mean,
            "cross_over_pair_speedups": pair_speedups,
            "model_memory_reduction_fraction": 1.0
            - candidate_model_memory / reference_model_memory,
            "peak_allocated_memory_reduction_fraction": 1.0
            - candidate_peak / reference_peak,
        },
        "parity": {
            "candidate": {
                "path": str(candidate_parity_path),
                "sha256": sha256_file(candidate_parity_path),
                "cases": candidate_parity["cases"],
                "identical_cases": candidate_parity["identical_cases"],
            },
            "locked_reference": {
                "path": str(locked_parity_path),
                "sha256": sha256_file(locked_parity_path),
                "cases": locked_parity["cases"],
                "identical_cases": locked_parity["identical_cases"],
            },
        },
        "receipts": {
            str(path): sha256_file(path)
            for path in candidate_paths
            + reference_paths
            + [candidate_parity_path, locked_parity_path]
        },
        "bindings": {
            "candidate": str(args.candidate),
            "candidate_sha256": sha256_file(args.candidate),
            "prereg": str(args.prereg),
            "prereg_sha256": sha256_file(args.prereg),
            "benchmark_runner_sha256": candidate[0]["runner_sha256"],
            "parity_runner_sha256": candidate_parity["runner_sha256"],
            "summary_runner_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report["comparison"], indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
