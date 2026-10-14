"""One-time paired identity/Flip8 UTSW362 external evaluation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from scripts import train_t1ce_wt_gcsam_inner as base
from scripts import utsw_no_n4_paired_external_protocol as p


class UTSWDataset(Dataset):
    def __init__(self, arm: str) -> None:
        if arm not in ("canonical", "no_n4"):
            raise ValueError("unknown UTSW cache arm")
        receipt = json.loads((p.RECORD / "CACHE_RECEIPT.json").read_text())
        if receipt.get("all_gates_passed") is not True:
            raise ValueError("UTSW paired caches did not pass")
        row = receipt["indices"][arm]
        index_path = p.CACHE_ROOT / arm / "index.json"
        p.checked(index_path, row["sha256"])
        self.index = json.loads(index_path.read_text())
        if self.index.get("arm") != arm or self.index.get("cases") != p.CASES or self.index.get("split") != "external_cross_stage":
            raise ValueError("wrong UTSW cache identity")
        p.checked(p.CACHE_ROOT / arm / "images.f32", self.index["images_sha256"])
        p.checked(p.CACHE_ROOT / arm / "labels.u8", self.index["labels_sha256"])
        self.images = np.memmap(p.CACHE_ROOT / arm / "images.f32", mode="r", dtype=np.float32, shape=(p.CASES, *p.MODEL_SHAPE))
        self.labels = np.memmap(p.CACHE_ROOT / arm / "labels.u8", mode="r", dtype=np.uint8, shape=(p.CASES, *p.MODEL_SHAPE))

    def __len__(self) -> int:
        return p.CASES

    def __getitem__(self, slot: int) -> dict:
        return {
            "image": torch.from_numpy(np.array(self.images[slot], dtype=np.float32))[None],
            "label": torch.from_numpy(np.array(self.labels[slot], dtype=np.uint8))[None],
            "case_id": self.index["case_ids"][slot],
            "subject_id": self.index["subject_ids"][slot],
        }


def load_model(role: str, device: torch.device) -> tuple[torch.nn.Module, dict, dict]:
    preaccess = p.verify_preaccess()
    model_row = preaccess["models"][role]
    checkpoint_path = Path(model_row["checkpoint"])
    p.checked(checkpoint_path, model_row["checkpoint_sha256"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "aflow.n4_family_full972.train.checkpoint":
        raise ValueError("wrong Full972 checkpoint schema")
    if payload.get("role") != role or payload.get("seed") != 20261343 or payload.get("epoch") != 168 or payload.get("smoke") is not False:
        raise ValueError(f"wrong Full972 checkpoint identity: {role}")
    if payload.get("fit_sha256") != p.full.SOURCE_SHA or payload.get("stage_a_inner_metrics_used_for_training") is not False:
        raise ValueError("Full972 checkpoint fit/access mismatch")
    _, expected_training_cache_sha = p.full.verify_cache(role)
    if payload.get("input_cache_index_sha256") != expected_training_cache_sha:
        raise ValueError(f"wrong Full972 training cache identity: {role}")
    if payload.get("architecture") != {"init_filters": 16, "dropout": 0.1, "center_aware": False}:
        raise ValueError("Full972 checkpoint architecture mismatch")
    model = base.AutomaticPromptLocator(center_aware=False, init_filters=16, dropout=0.1).to(device).eval()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != 4_700_897:
        raise ValueError("wrong UTSW model parameter count")
    return model, payload, preaccess


def summarize(rows: list[dict]) -> dict:
    if len(rows) != p.CASES or len({row["case_id"] for row in rows}) != p.CASES:
        raise ValueError("UTSW summary cohort mismatch")
    if any(row["case_id"] != row["subject_id"] for row in rows):
        raise ValueError("UTSW requires one scan per subject")
    scan = statistics.fmean(row["dice"] for row in rows)
    return {
        "cases": p.CASES, "subjects": p.CASES,
        "scan_dice": scan, "subject_dice": scan,
        "point_hits": sum(row["point_hit"] for row in rows),
        "point_hit_rate": statistics.fmean(float(row["point_hit"]) for row in rows),
        "predicted_voxels_mean": statistics.fmean(row["predicted_voxels"] for row in rows),
        "target_voxels_mean": statistics.fmean(row["target_voxels"] for row in rows),
    }


def bootstrap(canonical: list[dict], no_n4: list[dict]) -> dict:
    canonical = sorted(canonical, key=lambda row: row["case_id"])
    no_n4 = sorted(no_n4, key=lambda row: row["case_id"])
    if [row["case_id"] for row in canonical] != [row["case_id"] for row in no_n4] or len(canonical) != p.CASES:
        raise ValueError("UTSW paired bootstrap IDs mismatch")
    can_dice = np.asarray([row["dice"] for row in canonical], dtype=np.float64)
    no_dice = np.asarray([row["dice"] for row in no_n4], dtype=np.float64)
    can_point = np.asarray([row["point_hit"] for row in canonical], dtype=np.float64)
    no_point = np.asarray([row["point_hit"] for row in no_n4], dtype=np.float64)
    rng = np.random.default_rng(p.BOOTSTRAP_SEED)
    draws = rng.integers(0, p.CASES, size=(p.BOOTSTRAP_SAMPLES, p.CASES))
    series = {
        "canonical_scan": can_dice[draws].mean(axis=1),
        "no_n4_scan": no_dice[draws].mean(axis=1),
        "scan_delta_no_n4_minus_canonical": (no_dice - can_dice)[draws].mean(axis=1),
        "canonical_point_rate": can_point[draws].mean(axis=1),
        "no_n4_point_rate": no_point[draws].mean(axis=1),
        "point_rate_delta_no_n4_minus_canonical": (no_point - can_point)[draws].mean(axis=1),
    }
    return {
        "unit": "subject_one_scan_each", "samples": p.BOOTSTRAP_SAMPLES, "seed": p.BOOTSTRAP_SEED,
        **{name + "_95ci": [float(value) for value in np.quantile(values, (0.025, 0.975))] for name, values in series.items()},
    }


def support_gates(canonical_summary: dict, no_n4_summary: dict, interval: dict) -> dict:
    delta = no_n4_summary["scan_dice"] - canonical_summary["scan_dice"]
    return {
        "flip8_scan_delta_positive": delta > 0,
        "flip8_scan_delta_ci_lower_at_least_minus_0p003": interval["scan_delta_no_n4_minus_canonical_95ci"][0] >= -0.003,
        "flip8_point_deficit_at_most_3": no_n4_summary["point_hits"] >= canonical_summary["point_hits"] - 3,
    }


def rows_from_logits(logits: torch.Tensor, target: torch.Tensor, batch: dict, view: str) -> list[dict]:
    prediction = logits > 0
    rows = []
    for slot, case_id in enumerate(batch["case_id"]):
        pred = prediction[slot].flatten()
        truth = target[slot].flatten()
        pred_count, truth_count = int(pred.sum()), int(truth.sum())
        intersection = int((pred & truth).sum())
        point = int(logits[slot].flatten().argmax())
        rows.append({
            "case_id": str(case_id), "subject_id": str(batch["subject_id"][slot]), "view": view,
            "dice": 2 * intersection / (pred_count + truth_count),
            "point_hit": bool(truth[point]), "point_flat_index": point,
            "predicted_voxels": pred_count, "target_voxels": truth_count, "intersection_voxels": intersection,
            "prediction_mask_sha256": hashlib.sha256(np.packbits(pred.cpu().numpy()).tobytes()).hexdigest(),
        })
    return rows


@torch.inference_mode()
def evaluate_arm(role: str, device: torch.device) -> tuple[dict[str, list[dict]], int, int]:
    data = UTSWDataset(role)
    loader = DataLoader(data, batch_size=8, shuffle=False, num_workers=0)
    model, _, _ = load_model(role, device)
    rows = {"identity": [], "flip8": []}
    forwards = 0
    torch.cuda.reset_peak_memory_stats(device)
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32)
        target = batch["label"].to(device=device, dtype=torch.bool)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            identity = model(image).segmentation_logits.float()
        forwards += image.shape[0]
        rows["identity"].extend(rows_from_logits(identity, target, batch, "identity"))
        total = None
        for axes in p.AXES:
            view = torch.flip(image, axes) if axes else image
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(view).segmentation_logits.float()
            logits = torch.flip(logits, axes) if axes else logits
            total = logits.clone() if total is None else total.add_(logits)
            forwards += image.shape[0]
        rows["flip8"].extend(rows_from_logits(total / 8, target, batch, "flip8"))
    peak = torch.cuda.max_memory_allocated(device)
    del model
    torch.cuda.empty_cache()
    return rows, forwards, peak


@torch.inference_mode()
def main(device_name: str) -> None:
    from scripts.freeze_utsw_no_n4_paired_external import check_implementation

    check_implementation()
    preaccess = p.verify_preaccess()
    smoke = json.loads((p.RECORD / "GPU_SMOKE_RECEIPT.json").read_text())
    cache = json.loads((p.RECORD / "CACHE_RECEIPT.json").read_text())
    if smoke.get("status") != "passed" or smoke.get("preaccess_sha256") != p.sha(p.PREACCESS):
        raise ValueError("UTSW paired smoke mismatch")
    if cache.get("all_gates_passed") is not True:
        raise ValueError("UTSW paired cache mismatch")
    output = p.RESULT_WORK / "UTSW_PAIRED_EXTERNAL.json"
    started = p.RESULT_WORK / "UTSW_PAIRED_EXTERNAL_STARTED.json"
    if output.exists() or started.exists():
        raise FileExistsError("refusing repeated UTSW external evaluation")
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("UTSW external evaluation requires CUDA")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    p.RESULT_WORK.mkdir(parents=True, exist_ok=False)
    p.write_new(started, {
        "schema": "aflow.utsw_no_n4_paired_external.started",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preaccess_sha256": p.sha(p.PREACCESS), "cache_receipt_sha256": p.sha(p.RECORD / "CACHE_RECEIPT.json"),
        "views": ["identity", "Flip8"], "threshold": 0.0, "external_metrics_access": True,
    })
    begin = time.monotonic()
    rows, summaries, intervals, peaks = {}, {}, {}, {}
    forwards = 0
    for role in ("canonical", "no_n4"):
        rows[role], count, peaks[role] = evaluate_arm(role, device)
        forwards += count
        summaries[role] = {view: summarize(rows[role][view]) for view in ("identity", "flip8")}
    for view in ("identity", "flip8"):
        intervals[view] = bootstrap(rows["canonical"][view], rows["no_n4"][view])
    if forwards != 6516:
        raise RuntimeError(f"UTSW fixed forward count mismatch: {forwards}")
    gates = support_gates(summaries["canonical"]["flip8"], summaries["no_n4"]["flip8"], intervals["flip8"])
    report = {
        "schema": "aflow.utsw_no_n4_paired_external.result",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "one-time UTSW362 preoperative SRI24 cross-stage external confirmation",
        "models": preaccess["models"], "protocol": preaccess["protocol"],
        "summaries": summaries, "bootstrap": intervals, "per_case": rows,
        "gates": gates, "all_support_gates_passed": all(gates.values()),
        "forwards": forwards, "elapsed_seconds": time.monotonic() - begin,
        "peak_gpu_memory_bytes": peaks,
        "preaccess_sha256": p.sha(p.PREACCESS), "cache_receipt_sha256": p.sha(p.RECORD / "CACHE_RECEIPT.json"),
        "evaluator_sha256": p.sha(Path(__file__)),
        "access": {"utsw_external": True, "fixed_validation": False, "outer": False, "confirmation": False, "test270": False, "ucsd75": False, "rhuh": False},
        "result_must_not_trigger_tuning_or_repeat": True,
        "grid_caveat": "Dice/Point on image-support crop rasterized to128^3, not native millimeter space",
        "domain_caveat": "UTSW is preoperative/SRI24; training is post-treatment/MNI. Keep separate from post-treatment rankings.",
    }
    p.write_new(output, report)
    print(json.dumps({"summaries": summaries, "bootstrap": intervals, "gates": gates, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    main(parser.parse_args().device)
