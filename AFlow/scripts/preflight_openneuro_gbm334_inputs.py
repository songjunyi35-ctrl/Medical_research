#!/usr/bin/env python3
"""Stream and validate all frozen OpenNeuro-GBM334 inputs without a model."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

from sammed3d import t1ce_binary
from scripts import openneuro_gbm334_protocol as p
from scripts.probe_openneuro_gbm334_first_case import prepare


ORIGINAL_CASE_ID = t1ce_binary._case_id


def initialize(allowed: list[str]) -> None:
    torch.set_num_threads(1)
    import SimpleITK as sitk

    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    valid = set(allowed)
    t1ce_binary._case_id = lambda value: value if value in valid else ORIGINAL_CASE_ID(value)


def worker(row: dict) -> dict:
    try:
        with tempfile.TemporaryDirectory(prefix="aflow_gbm334_input_", dir="/tmp") as directory:
            _, _, details = prepare(row, Path(directory))
        return {"status": "passed", "subject_id": row["subject_id"], "details": details}
    except Exception as exc:
        return {
            "status": "failed",
            "subject_id": row["subject_id"],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=p.RECORD / "INPUT_PREFLIGHT.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = p.cases()
    first = json.loads((p.RECORD / "FIRST_CASE_PROBE.json").read_text())
    if first["status"] != "passed" or first["subject_id"] != rows[0]["subject_id"]:
        raise ValueError("first probe contract")
    begin = time.monotonic()
    results = []
    allowed = [row["subject_id"] for row in rows]
    with mp.get_context("spawn").Pool(
        args.workers, initializer=initialize, initargs=(allowed,)
    ) as pool:
        for item in pool.imap_unordered(worker, rows, chunksize=1):
            results.append(item)
            if len(results) % 20 == 0 or len(results) == len(rows):
                failures = sum(row["status"] != "passed" for row in results)
                print(f"input preflight {len(results)}/{len(rows)} failures={failures}", flush=True)
    results.sort(key=lambda row: row["subject_id"])
    passed = [row for row in results if row["status"] == "passed"]
    failures = [row for row in results if row["status"] != "passed"]
    raw_union = sorted({value for row in passed for value in row["details"]["raw_values"]})
    first_again = next(row for row in passed if row["subject_id"] == first["subject_id"])
    gates = {
        "all_334_completed": len(passed) == p.CASES and not failures,
        "all_source_sha_exact": all(
            row["details"]["image_source_sha256"]
            == next(x for x in rows if x["subject_id"] == row["subject_id"])["image"]["sha256"]
            and row["details"]["label_source_sha256"]
            == next(x for x in rows if x["subject_id"] == row["subject_id"])["label"]["sha256"]
            for row in passed
        ),
        "all_geometry_exact": all(row["details"]["affine_error"] <= 1e-5 for row in passed),
        "all_zero_wt_crop_loss": all(
            row["details"]["wt_outside_image_positive_crop"] == 0 for row in passed
        ),
        "all_wt_nonempty_before_after": all(
            row["details"]["raw_wt"] > 0 and row["details"]["resized_wt"] > 0
            for row in passed
        ),
        "global_raw_union_exact": raw_union == [0, 1, 2, 3],
        "first_probe_image_exact": first_again["details"]["image_array_sha256"]
        == first["details"]["image_array_sha256"],
        "first_probe_label_exact": first_again["details"]["label_array_sha256"]
        == first["details"]["label_array_sha256"],
    }
    receipt = {
        "schema": "aflow.openneuro_gbm334.input_preflight",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(gates.values()) else "failed",
        "cases": p.CASES,
        "workers": args.workers,
        "snapshot": "ds007045:1.0.1",
        "gates": gates,
        "raw_union": raw_union,
        "failures": failures,
        "details": {row["subject_id"]: row["details"] for row in passed},
        "elapsed_seconds": time.monotonic() - begin,
        "source_bytes_streamed": sum(
            row[k]["bytes"] for row in rows for k in ("image", "label")
        ),
        "persistent_source_or_cache_created": False,
        "checkpoint_loaded": False,
        "model_forwards": 0,
        "performance_accessed": False,
        "first_probe_sha256": p.sha256(p.RECORD / "FIRST_CASE_PROBE.json"),
        "runner_sha256": p.sha256(Path(__file__)),
    }
    p.write_new(args.output, receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "gates", "elapsed_seconds")}, indent=2))
    if receipt["status"] != "passed":
        raise RuntimeError(failures[:3] if failures else gates)


if __name__ == "__main__":
    main()
