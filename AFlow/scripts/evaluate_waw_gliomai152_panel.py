#!/usr/bin/env python3
"""One-time frozen checkpoint panel evaluation on public WAW-GliomAI152."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import scripts.evaluate_brats_africa95_frozen_baselines as metrics
import scripts.evaluate_ucsd_ptgbm_external_confirmation as model_io
from scripts import waw_gliomai152_protocol as p


metrics.p = p


class WAWData(Dataset):
    def __init__(self) -> None:
        receipt = json.loads(p.CACHE_RECEIPT.read_text())
        p.checked(p.CACHE / "index.json", receipt["index_sha256"])
        self.index = json.loads((p.CACHE / "index.json").read_text())
        if not receipt["all_gates_passed"] or self.index["cases"] != p.CASES:
            raise ValueError("cache contract")
        p.checked(p.CACHE / "images.f32", receipt["images_sha256"])
        p.checked(p.CACHE / "labels.u8", receipt["labels_sha256"])
        shape = (p.CASES, 128, 128, 128)
        self.images = np.memmap(p.CACHE / "images.f32", dtype=np.float32, mode="r", shape=shape)
        self.labels = np.memmap(p.CACHE / "labels.u8", dtype=np.uint8, mode="r", shape=shape)

    def __len__(self) -> int:
        return p.CASES

    def __getitem__(self, index: int) -> dict:
        return {
            "image": torch.from_numpy(np.array(self.images[index]))[None],
            "label": torch.from_numpy(np.array(self.labels[index]))[None],
            "case_id": self.index["case_ids"][index],
            "subject_id": self.index["subject_ids"][index],
        }


def interpretation(comparison: dict, delta_key: str) -> str:
    delta = comparison[delta_key]["scan_dice"]
    lower = comparison["paired_scan_dice_95ci"][0]
    if delta > 0 and lower > 0:
        return "supported_on_WAW"
    if delta > 0:
        return "positive_but_uncertain"
    return "not_supported_or_reversed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    p.docs()
    preaccess = json.loads(p.MODEL_PREACCESS.read_text())
    cache = json.loads(p.CACHE_RECEIPT.read_text())
    smoke_path = p.RECORD / "SMOKE_RECEIPT.json"
    if (
        preaccess.get("status") != "frozen_after_cache_before_model"
        or preaccess["access"]["model_inference"] is not False
        or preaccess["code"]["evaluator_sha256"] != p.sha256(Path(__file__))
        or preaccess["cache_receipt_sha256"] != p.sha256(p.CACHE_RECEIPT)
        or preaccess["smoke_receipt_sha256"] != p.sha256(smoke_path)
    ):
        raise ValueError("model preaccess contract")
    output = p.RECORD / "RAW_WAW152.json"
    started = p.RECORD / "RAW_WAW152_STARTED.json"
    if output.exists() or started.exists():
        raise FileExistsError("one-time result already started")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    loader = DataLoader(WAWData(), batch_size=8, shuffle=False, num_workers=0)
    p.write_new(
        started,
        {
            "schema": "aflow.waw_gliomai152.started",
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "preaccess_sha256": p.sha256(p.MODEL_PREACCESS),
            "external_access": True,
            "checkpoint_loaded_at_write": False,
            "model_forwards_at_write": 0,
        },
    )
    full, _ = model_io.load_model(p.FULL, p.FULL_SHA, "primary", device)
    gcsam, _ = model_io.load_model(p.GCSAM, p.GCSAM_SHA, "secondary", device)
    torch.cuda.reset_peak_memory_stats(device)
    begin = time.monotonic()
    full_summaries, full_forwards = metrics.evaluate_full(full, loader, device)
    del full
    torch.cuda.empty_cache()
    gcsam_summary, gcsam_forwards = metrics.evaluate_identity(gcsam, loader, device)
    forwards = full_forwards + gcsam_forwards
    if forwards != 1368:
        raise RuntimeError(f"forward count mismatch: {forwards}")
    gcsam_comparison = metrics.compare(
        full_summaries["identity"],
        gcsam_summary,
        p.BOOTSTRAP_SEED + 200,
        "gcsam_minus_full",
    )
    flip8_comparison = metrics.compare(
        full_summaries["identity"],
        full_summaries["flip8"],
        p.BOOTSTRAP_SEED + 100,
        "flip8_minus_identity",
    )
    report = {
        "schema": "aflow.waw_gliomai152.frozen_checkpoint_panel",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "WAW-GliomAI final public preop152; external to project; institution-disjoint/no-known-overlap; no patient-ID bridge; not hidden blind",
        "preregistered": True,
        "not_test_tuned": True,
        "not_pristine_blind": True,
        "data": {
            "cases": p.CASES,
            "subjects": p.CASES,
            "cohort_sha256": p.COHORT_SHA,
            "source_receipt_sha256": p.SOURCE_RECEIPT_SHA,
            "cache_index_sha256": cache["index_sha256"],
            "modality": "publisher delayed post-contrast T1 named t1c",
            "target": "preoperative WT=all positive labels {1,2,3}",
            "description_count_correction": "description156; final exact T1c-mask intersection152",
        },
        "full972_kd168": {
            "checkpoint_sha256": p.FULL_SHA,
            "identity": full_summaries["identity"],
            "identity_bootstrap": metrics.absolute(full_summaries["identity"], p.BOOTSTRAP_SEED),
            "flip8": full_summaries["flip8"],
            "flip8_bootstrap": metrics.absolute(full_summaries["flip8"], p.BOOTSTRAP_SEED + 10),
        },
        "gcsam184": {
            "checkpoint_sha256": p.GCSAM_SHA,
            "identity": gcsam_summary,
            "identity_bootstrap": metrics.absolute(gcsam_summary, p.BOOTSTRAP_SEED + 20),
        },
        "comparisons": {
            "gcsam_identity_minus_full_identity": gcsam_comparison,
            "full_flip8_minus_identity": flip8_comparison,
        },
        "interpretation_gates": {
            "gcsam_identity_minus_full_identity": interpretation(gcsam_comparison, "gcsam_minus_full"),
            "full_flip8_minus_identity": interpretation(flip8_comparison, "flip8_minus_identity"),
        },
        "performance_promotion_gate": None,
        "result_must_not_trigger_tuning_or_repeat": True,
        "forwards": forwards,
        "elapsed_seconds": time.monotonic() - begin,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "access": {
            "waw_gliomai152": True,
            "internal_test270": False,
            "fixed_validation108": False,
            "outer": False,
            "confirmation": False,
            "ucsd75": False,
            "africa95": False,
            "cptac33": False,
        },
        "training_source_caveat": "local BraTS2024 Kaggle mirror; official Synapse parity unresolved",
        "evaluator_sha256": p.sha256(Path(__file__)),
    }
    p.write_new(output, report)
    print(
        json.dumps(
            {
                "full_identity": {
                    key: full_summaries["identity"][key]
                    for key in ("locator_dsc_mean", "automatic_point_hits")
                },
                "full_flip8": {
                    key: full_summaries["flip8"][key]
                    for key in ("locator_dsc_mean", "automatic_point_hits")
                },
                "gcsam_identity": {
                    key: gcsam_summary[key] for key in ("locator_dsc_mean", "automatic_point_hits")
                },
                "comparisons": report["comparisons"],
                "interpretation_gates": report["interpretation_gates"],
                "forwards": forwards,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
