#!/usr/bin/env python3
"""One-time streaming frozen-checkpoint evaluation on OpenNeuro-GBM334."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import scripts.evaluate_brats_africa95_frozen_baselines as metrics
import scripts.evaluate_ucsd_ptgbm_external_confirmation as model_io
from sammed3d import t1ce_binary
from scripts import openneuro_gbm334_protocol as p
from scripts.probe_openneuro_gbm334_first_case import prepare


ORIGINAL_CASE_ID = t1ce_binary._case_id
metrics.p = p


def initialize(allowed: list[str]) -> None:
    torch.set_num_threads(1)
    import SimpleITK as sitk

    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    valid = set(allowed)
    t1ce_binary._case_id = lambda value: value if value in valid else ORIGINAL_CASE_ID(value)


def stream_case(row: dict) -> dict:
    for attempt in range(3):
        try:
            with tempfile.TemporaryDirectory(prefix="aflow_gbm334_formal_", dir="/tmp") as directory:
                image, label, details = prepare(row, Path(directory))
            return {
                "subject_id": row["subject_id"],
                "image": image,
                "label": label,
                "details": details,
            }
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 + attempt)
    raise AssertionError("unreachable")


def interpretation(comparison: dict, delta_key: str) -> str:
    delta = comparison[delta_key]["scan_dice"]
    lower = comparison["paired_scan_dice_95ci"][0]
    if delta > 0 and lower > 0:
        return "supported_on_OpenNeuro_GBM334"
    if delta > 0:
        return "positive_but_uncertain"
    return "not_supported_or_reversed"


def save_progress(path: Path, rows: dict, started_sha: str) -> None:
    payload = {
        "schema": "aflow.openneuro_gbm334.formal_progress",
        "schema_version": 1,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_sha256": started_sha,
        "completed_cases": len(rows["full_identity"]),
        "model_forwards": len(rows["full_identity"]) * 9,
        "partial_performance_accessed": bool(rows["full_identity"]),
        "rows": rows,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    p.docs()
    preaccess_path = p.RECORD / "MODEL_PREACCESS.json"
    preaccess = json.loads(preaccess_path.read_text())
    preflight_path = p.RECORD / "INPUT_PREFLIGHT.json"
    smoke_path = p.RECORD / "SMOKE_RECEIPT.json"
    if (
        preaccess.get("status") != "frozen_after_all_input_gates_before_external_model"
        or preaccess["access"]["external_performance"] is not False
        or preaccess["evaluator_sha256"] != p.sha256(Path(__file__))
        or preaccess["input_preflight_sha256"] != p.sha256(preflight_path)
        or preaccess["smoke_receipt_sha256"] != p.sha256(smoke_path)
    ):
        raise ValueError("model preaccess contract")
    preflight = json.loads(preflight_path.read_text())
    if preflight["status"] != "passed" or not all(preflight["gates"].values()):
        raise ValueError("input preflight not passed")
    output = p.RECORD / "RAW_OPENNEURO_GBM334.json"
    started = p.RECORD / "RAW_OPENNEURO_GBM334_STARTED.json"
    progress = p.RECORD / "RAW_OPENNEURO_GBM334_PROGRESS.json"
    if output.exists() or started.exists() or progress.exists():
        raise FileExistsError("one-time evaluation already started")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    p.write_new(
        started,
        {
            "schema": "aflow.openneuro_gbm334.started",
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "preaccess_sha256": p.sha256(preaccess_path),
            "external_access": True,
            "checkpoint_loaded_at_write": False,
            "model_forwards_at_write": 0,
        },
    )
    started_sha = p.sha256(started)
    full, full_payload = model_io.load_model(p.FULL, p.FULL_SHA, "primary", device)
    gcsam, gcsam_payload = model_io.load_model(p.GCSAM, p.GCSAM_SHA, "secondary", device)
    rows = {"full_identity": [], "full_flip8": [], "gcsam_identity": []}
    source_details = {}
    torch.cuda.reset_peak_memory_stats(device)
    begin = time.monotonic()
    cases = p.cases()
    allowed = [row["subject_id"] for row in cases]
    with mp.get_context("spawn").Pool(
        args.workers, initializer=initialize, initargs=(allowed,)
    ) as pool:
        for item in pool.imap_unordered(stream_case, cases, chunksize=1):
            subject = item["subject_id"]
            image = torch.from_numpy(item["image"])[None, None].to(device, dtype=torch.float32)
            target = torch.from_numpy(item["label"])[None, None].to(device, dtype=torch.float32)
            total = None
            identity = None
            for axes in p.AXES:
                view = torch.flip(image, axes) if axes else image
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = full(view).segmentation_logits.float()
                logits = torch.flip(logits, axes) if axes else logits
                if identity is None:
                    identity = logits.clone()
                    total = torch.zeros_like(logits)
                total.add_(logits)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                gcsam_logits = gcsam(image).segmentation_logits.float()
            rows["full_identity"].append(metrics.row(identity, target, subject, subject))
            rows["full_flip8"].append(metrics.row(total / 8, target, subject, subject))
            rows["gcsam_identity"].append(metrics.row(gcsam_logits, target, subject, subject))
            source_details[subject] = item["details"]
            save_progress(progress, rows, started_sha)
            if len(rows["full_identity"]) % 10 == 0 or len(rows["full_identity"]) == p.CASES:
                print(f"formal streamed {len(rows['full_identity'])}/{p.CASES}", flush=True)
    for values in rows.values():
        values.sort(key=lambda row: row["case_id"])
    full_identity = metrics.summarize(rows["full_identity"])
    full_flip8 = metrics.summarize(rows["full_flip8"])
    gcsam_identity = metrics.summarize(rows["gcsam_identity"])
    flip_comparison = metrics.compare(
        full_identity, full_flip8, p.BOOTSTRAP_SEED + 100, "flip8_minus_identity"
    )
    gcsam_comparison = metrics.compare(
        full_identity, gcsam_identity, p.BOOTSTRAP_SEED + 200, "gcsam_minus_full"
    )
    report = {
        "schema": "aflow.openneuro_gbm334.frozen_checkpoint_panel",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "OpenNeuro ds007045 v1.0.1 public preoperative expert-reviewed/refined GBM334; eight institutions disjoint from BraTS2024 sites; no patient-ID bridge; not hidden blind",
        "preregistered": True,
        "not_test_tuned": True,
        "not_pristine_blind": True,
        "data": {
            "cases": p.CASES,
            "subjects": p.CASES,
            "source_inventory_sha256": p.INVENTORY_SHA,
            "manifest_sha256": p.MANIFEST_SHA,
            "input_preflight_sha256": p.sha256(preflight_path),
            "modality": "publisher MNI152 registered and skull-stripped T1ce before publisher N4/z-score",
            "target": "preoperative WT=all expert-reviewed/refined labels {1,2,3}",
            "published_count_correction": "participants337; exact frozen processing-T1ce/final-mask intersection334",
            "source_bytes_restreamed": sum(
                row[k]["bytes"] for row in cases for k in ("image", "label")
            ),
            "persistent_source_or_cache_created": False,
        },
        "full972_kd168": {
            "checkpoint_sha256": p.FULL_SHA,
            "epoch": full_payload["epoch"],
            "identity": full_identity,
            "identity_bootstrap": metrics.absolute(full_identity, p.BOOTSTRAP_SEED),
            "flip8": full_flip8,
            "flip8_bootstrap": metrics.absolute(full_flip8, p.BOOTSTRAP_SEED + 10),
        },
        "gcsam184": {
            "checkpoint_sha256": p.GCSAM_SHA,
            "epoch": gcsam_payload["epoch"],
            "identity": gcsam_identity,
            "identity_bootstrap": metrics.absolute(gcsam_identity, p.BOOTSTRAP_SEED + 20),
        },
        "comparisons": {
            "full_flip8_minus_identity": flip_comparison,
            "gcsam_identity_minus_full_identity": gcsam_comparison,
        },
        "interpretation_gates": {
            "full_flip8_minus_identity": interpretation(flip_comparison, "flip8_minus_identity"),
            "gcsam_identity_minus_full_identity": interpretation(gcsam_comparison, "gcsam_minus_full"),
        },
        "performance_promotion_gate": None,
        "result_must_not_trigger_tuning_or_repeat": True,
        "forwards": p.CASES * 9,
        "elapsed_seconds": time.monotonic() - begin,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "source_details": source_details,
        "access": {
            "openneuro_gbm334": True,
            "internal_test270": False,
            "fixed_validation108": False,
            "outer": False,
            "confirmation": False,
            "ucsd75": False,
            "africa95": False,
            "cptac33": False,
            "waw_pre152": False,
            "waw_post31": False,
        },
        "training_source_caveat": "local BraTS2024 Kaggle mirror; official Synapse parity unresolved",
        "evaluator_sha256": p.sha256(Path(__file__)),
    }
    if report["forwards"] != 3006:
        raise RuntimeError("forward count contract")
    p.write_new(output, report)
    print(
        json.dumps(
            {
                "full_identity": [full_identity["locator_dsc_mean"], full_identity["automatic_point_hits"]],
                "full_flip8": [full_flip8["locator_dsc_mean"], full_flip8["automatic_point_hits"]],
                "gcsam_identity": [gcsam_identity["locator_dsc_mean"], gcsam_identity["automatic_point_hits"]],
                "comparisons": report["comparisons"],
                "interpretation_gates": report["interpretation_gates"],
                "forwards": report["forwards"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
