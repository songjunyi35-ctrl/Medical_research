#!/usr/bin/env python3
"""One-time Full972 identity/Flip8 and GCSAM identity evaluation on Africa95."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import scripts.evaluate_ucsd_ptgbm_external_confirmation as model_io
from scripts import brats_africa95_protocol as p


class Data(Dataset):
    def __init__(self):
        receipt = json.loads(p.CACHE_RECEIPT.read_text())
        if receipt.get("all_gates_passed") is not True: raise ValueError("cache receipt failed")
        p.checked(p.CACHE / "index.json", receipt["index_sha256"]); self.index = json.loads((p.CACHE / "index.json").read_text())
        if self.index.get("cases") != p.CASES or self.index.get("split") != "public_postchallenge_external95": raise ValueError("cache identity")
        p.checked(p.CACHE / "images.f32", self.index["images_sha256"]); p.checked(p.CACHE / "labels.u8", self.index["labels_sha256"])
        self.images = np.memmap(p.CACHE / "images.f32", dtype=np.float32, mode="r", shape=(p.CASES, 128, 128, 128)); self.labels = np.memmap(p.CACHE / "labels.u8", dtype=np.uint8, mode="r", shape=(p.CASES, 128, 128, 128))
    def __len__(self): return p.CASES
    def __getitem__(self, i): return {"image": torch.from_numpy(np.array(self.images[i]))[None], "label": torch.from_numpy(np.array(self.labels[i]))[None], "case_id": self.index["case_ids"][i], "subject_id": self.index["subject_ids"][i]}


def row(logits, target, case_id, subject_id):
    pred = logits[0, 0] > 0; truth = target[0, 0] > 0.5
    intersection = int((pred & truth).sum()); denominator = int(pred.sum() + truth.sum()); point = int(logits[0, 0].flatten().argmax())
    return {"case_id": str(case_id), "subject_id": str(subject_id), "locator_dsc": 2 * intersection / denominator, "automatic_point_hits_gt": bool(truth.flatten()[point]), "automatic_point_flat_index": point, "predicted_wt_voxels": int(pred.sum()), "target_wt_voxels": int(truth.sum()), "intersection_voxels": intersection}


def summarize(rows):
    if len(rows) != p.CASES or len({x["case_id"] for x in rows}) != p.CASES: raise ValueError("summary cohort")
    dice = statistics.fmean(x["locator_dsc"] for x in rows); hits = sum(x["automatic_point_hits_gt"] for x in rows)
    return {"cases": p.CASES, "subjects": p.CASES, "locator_dsc_mean": dice, "locator_subject_dsc_mean": dice, "automatic_point_hits": hits, "automatic_point_hit_rate_positive_only": hits / p.CASES, "per_case": rows}


def interval(values, seed):
    a = np.asarray(values, dtype=np.float64); draw = np.random.default_rng(seed).integers(0, len(a), size=(p.BOOTSTRAP_SAMPLES, len(a)))
    return [float(x) for x in np.quantile(a[draw].mean(1), (.025, .975))]


def absolute(summary, seed):
    ci = interval([x["locator_dsc"] for x in summary["per_case"]], seed)
    return {"scan_dsc_mean_95ci": ci, "subject_dsc_mean_95ci": ci}


def compare(left, right, seed, name):
    if [x["case_id"] for x in left["per_case"]] != [x["case_id"] for x in right["per_case"]]: raise ValueError("paired order")
    delta = [r["locator_dsc"] - l["locator_dsc"] for l, r in zip(left["per_case"], right["per_case"], strict=True)]
    point = right["automatic_point_hits"] - left["automatic_point_hits"]
    ci = interval(delta, seed)
    return {name: {"scan_dice": statistics.fmean(delta), "subject_dice": statistics.fmean(delta), "point_hits": point}, "paired_scan_dice_95ci": ci, "paired_subject_dice_95ci": ci, "cases_improved": sum(x > 0 for x in delta), "cases_worse": sum(x < 0 for x in delta), "cases_tied": sum(x == 0 for x in delta)}


@torch.inference_mode()
def evaluate_full(model, loader, device):
    identity_rows = []; flip_rows = []; forwards = 0
    for batch in loader:
        image = batch["image"].to(device, dtype=torch.float32); target = batch["label"].to(device, dtype=torch.float32); total = None; identity = None
        for axes in p.AXES:
            view = torch.flip(image, axes) if axes else image
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): logits = model(view).segmentation_logits.float()
            logits = torch.flip(logits, axes) if axes else logits
            if identity is None: identity = logits.clone(); total = torch.zeros_like(logits)
            total.add_(logits); forwards += len(image)
        for i, case in enumerate(batch["case_id"]):
            identity_rows.append(row(identity[i:i+1], target[i:i+1], case, batch["subject_id"][i])); flip_rows.append(row((total / 8)[i:i+1], target[i:i+1], case, batch["subject_id"][i]))
    return {"identity": summarize(identity_rows), "flip8": summarize(flip_rows)}, forwards


@torch.inference_mode()
def evaluate_identity(model, loader, device):
    rows = []; forwards = 0
    for batch in loader:
        image = batch["image"].to(device, dtype=torch.float32); target = batch["label"].to(device, dtype=torch.float32)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): logits = model(image).segmentation_logits.float()
        forwards += len(image)
        for i, case in enumerate(batch["case_id"]): rows.append(row(logits[i:i+1], target[i:i+1], case, batch["subject_id"][i]))
    return summarize(rows), forwards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--batch-size", type=int, default=8); args = parser.parse_args()
    p.documents(); pre = json.loads(p.PREACCESS.read_text()); cache = json.loads(p.CACHE_RECEIPT.read_text())
    if pre.get("status") != "frozen_after_cache_before_model_inference" or pre.get("access", {}).get("model_inference") is not False or cache.get("all_gates_passed") is not True: raise ValueError("preaccess/cache")
    if pre.get("evaluator_sha256") != p.sha(Path(__file__)) or pre.get("cache", {}).get("receipt_sha256") != p.sha(p.CACHE_RECEIPT): raise ValueError("preaccess code/cache binding")
    output = p.RECORD / "RAW_EXTERNAL95.json"; started = p.RECORD / "RAW_EXTERNAL95_STARTED.json"
    if output.exists() or started.exists(): raise FileExistsError("refusing repeated Africa95 evaluation")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    torch.cuda.set_device(device); torch.set_num_threads(1); loader = DataLoader(Data(), batch_size=args.batch_size, shuffle=False, num_workers=0)
    full, full_payload = model_io.load_model(p.FULL972, p.FULL972_SHA, "primary", device)
    gcsam, gcsam_payload = model_io.load_model(p.GCSAM184, p.GCSAM184_SHA, "secondary", device)
    p.write_new(started, {"schema": "aflow.brats_africa95.evaluation.started", "created_at_utc": datetime.now(timezone.utc).isoformat(), "preaccess_sha256": p.sha(p.PREACCESS), "cache_receipt_sha256": p.sha(p.CACHE_RECEIPT), "checkpoint_sha256": [p.FULL972_SHA, p.GCSAM184_SHA], "views": {"full972": ["identity", "flip8"], "gcsam184": ["identity"]}, "external_metrics_access": True})
    torch.cuda.reset_peak_memory_stats(device); begin = time.monotonic(); full_summary, n_full = evaluate_full(full, loader, device); del full; torch.cuda.empty_cache(); gcsam_summary, n_gc = evaluate_identity(gcsam, loader, device); peak = torch.cuda.max_memory_allocated(device)
    forwards = n_full + n_gc
    if forwards != 855: raise RuntimeError("forward count mismatch")
    report = {"schema": "aflow.brats_africa95.frozen_baseline_evaluation", "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "evidence_scope": "one-time public post-challenge BraTS-Africa 95_Glioma external cohort; six Nigerian centers; institution-disjoint but no patient-ID bridge; not hidden blind", "preregistered": True, "not_test_tuned": True, "not_pristine_blind": True, "external_confirmation": True, "protocol": {"target": "preoperative WT=all positive raw labels", "threshold": 0.0, "full972_views": ["identity", "fixed_flip8_equal_raw_logit_mean"], "gcsam_views": ["identity"], "sam": False, "postprocessing": False, "model_ensemble": False, "bootstrap_samples": p.BOOTSTRAP_SAMPLES, "bootstrap_seed": p.BOOTSTRAP_SEED}, "data": {"cases": p.CASES, "subjects": p.CASES, "cohort_sha256": p.COHORT_SHA, "cache_index_sha256": cache["index_sha256"], "images_sha256": cache["images_sha256"], "labels_sha256": cache["labels_sha256"], "original_roles_combined_without_selection": {"train": 60, "validation": 15, "test": 20}, "other_neoplasms_excluded_before_pixels": 51}, "full972_kd168": {"checkpoint_sha256": p.FULL972_SHA, "epoch": full_payload["epoch"], "identity": full_summary["identity"], "identity_bootstrap": absolute(full_summary["identity"], p.BOOTSTRAP_SEED), "flip8": full_summary["flip8"], "flip8_bootstrap": absolute(full_summary["flip8"], p.BOOTSTRAP_SEED + 10)}, "gcsam184": {"checkpoint_sha256": p.GCSAM184_SHA, "epoch": gcsam_payload["epoch"], "identity": gcsam_summary, "identity_bootstrap": absolute(gcsam_summary, p.BOOTSTRAP_SEED + 20)}, "comparisons": {"full_flip8_minus_identity": compare(full_summary["identity"], full_summary["flip8"], p.BOOTSTRAP_SEED + 100, "flip8_minus_identity"), "gcsam_identity_minus_full_identity": compare(full_summary["identity"], gcsam_summary, p.BOOTSTRAP_SEED + 200, "gcsam_minus_full")}, "performance_promotion_gate": None, "result_must_not_trigger_tuning_or_repeat": True, "forwards": forwards, "elapsed_seconds": time.monotonic() - begin, "peak_gpu_memory_bytes": peak, "grid_caveat": "source clinical scans include 3-5mm through-plane acquisition resampled by publisher to 1mm, then image-support crop rasterized to 128^3", "access": {"brats_africa95": True, "internal_test270": False, "fixed_validation108": False, "outer": False, "confirmation": False, "ucsd75": False, "utsw": False, "rhuh": False}, "evaluator_sha256": p.sha(Path(__file__))}
    p.write_new(output, report); print(json.dumps({"full_identity": {k: full_summary["identity"][k] for k in ("locator_dsc_mean", "automatic_point_hits")}, "full_flip8": {k: full_summary["flip8"][k] for k in ("locator_dsc_mean", "automatic_point_hits")}, "gcsam_identity": {k: gcsam_summary[k] for k in ("locator_dsc_mean", "automatic_point_hits")}, "comparisons": report["comparisons"], "forwards": forwards}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
