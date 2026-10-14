"""One-time fixed-Flip8 RHUH80 evaluation of the eligible Full972 no-N4 model."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from scripts import rhuh_no_n4_external_protocol as p
from scripts import train_t1ce_wt_gcsam_inner as base


class RHUHDataset(Dataset):
    def __init__(self) -> None:
        receipt = json.loads((p.RECORD / "CACHE_RECEIPT.json").read_text())
        if receipt.get("all_gates_passed") is not True:
            raise ValueError("RHUH cache did not pass")
        p.checked(p.CACHE / "index.json", receipt["cache_index_sha256"])
        self.index = json.loads((p.CACHE / "index.json").read_text())
        if self.index.get("split") != "external_confirmation" or self.index.get("cases") != 80:
            raise ValueError("wrong RHUH cache role/count")
        p.checked(p.CACHE / "images.f32", self.index["images_sha256"])
        p.checked(p.CACHE / "labels.u8", self.index["labels_sha256"])
        self.images = np.memmap(p.CACHE / "images.f32", mode="r", dtype=np.float32, shape=(80, 128, 128, 128))
        self.labels = np.memmap(p.CACHE / "labels.u8", mode="r", dtype=np.uint8, shape=(80, 128, 128, 128))

    def __len__(self) -> int:
        return 80

    def __getitem__(self, slot: int) -> dict:
        return {
            "image": torch.from_numpy(np.array(self.images[slot], dtype=np.float32))[None],
            "label": torch.from_numpy(np.array(self.labels[slot], dtype=np.uint8))[None],
            "case_id": self.index["case_ids"][slot],
            "subject_id": self.index["subject_ids"][slot],
        }


def load_model(device: torch.device) -> tuple[torch.nn.Module, dict, dict]:
    preaccess = p.verify_preaccess()
    checkpoint_path = Path(preaccess["checkpoint"]["path"])
    p.checked(checkpoint_path, preaccess["checkpoint"]["sha256"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "aflow.n4_family_full972.train.checkpoint":
        raise ValueError("wrong Full972 checkpoint schema")
    if payload.get("role") != "no_n4" or payload.get("epoch") != 168 or payload.get("seed") != 20261343:
        raise ValueError("wrong Full972 no-N4 checkpoint identity")
    if payload.get("fit_sha256") != p.full.SOURCE_SHA:
        raise ValueError("wrong Full972 fit ownership")
    model = base.AutomaticPromptLocator(center_aware=False, init_filters=16, dropout=0.1).to(device).eval()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != 4_700_897:
        raise ValueError("wrong RHUH locator parameter count")
    return model, payload, preaccess


def summarize(rows: list[dict]) -> dict:
    by_subject: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_subject[row["subject_id"]].append(row["dice"])
    if len(rows) != 80 or len(by_subject) != 40 or not all(len(values) == 2 for values in by_subject.values()):
        raise ValueError("RHUH summary cohort mismatch")
    return {
        "cases": 80,
        "subjects": 40,
        "scan_dice": statistics.fmean(row["dice"] for row in rows),
        "subject_dice": statistics.fmean(statistics.fmean(values) for values in by_subject.values()),
        "point_hits": sum(row["point_hit"] for row in rows),
        "point_hit_rate": statistics.fmean(float(row["point_hit"]) for row in rows),
        "predicted_voxels_mean": statistics.fmean(row["predicted_voxels"] for row in rows),
        "target_voxels_mean": statistics.fmean(row["target_voxels"] for row in rows),
    }


def bootstrap(rows: list[dict]) -> dict:
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_subject[row["subject_id"]].append(row)
    subjects = sorted(by_subject)
    rng = np.random.default_rng(p.BOOTSTRAP_SEED)
    draws = rng.integers(0, len(subjects), size=(p.BOOTSTRAP_SAMPLES, len(subjects)))
    scan, subject, point = [], [], []
    for draw in draws:
        groups = [by_subject[subjects[index]] for index in draw]
        subject_values = [statistics.fmean(item["dice"] for item in group) for group in groups]
        flat = [item for group in groups for item in group]
        scan.append(statistics.fmean(item["dice"] for item in flat))
        subject.append(statistics.fmean(subject_values))
        point.append(statistics.fmean(float(item["point_hit"]) for item in flat))
    return {
        "unit": "patient_cluster_all_two_timepoints",
        "samples": p.BOOTSTRAP_SAMPLES,
        "seed": p.BOOTSTRAP_SEED,
        "scan_dice_95ci": [float(value) for value in np.quantile(scan, (0.025, 0.975))],
        "subject_dice_95ci": [float(value) for value in np.quantile(subject, (0.025, 0.975))],
        "point_hit_rate_95ci": [float(value) for value in np.quantile(point, (0.025, 0.975))],
    }


@torch.inference_mode()
def main(device_name: str) -> None:
    from scripts.freeze_rhuh_no_n4_external import check_implementation

    check_implementation()
    preaccess = p.verify_preaccess()
    smoke = json.loads((p.RECORD / "GPU_SMOKE_RECEIPT.json").read_text())
    if smoke.get("status") != "passed" or smoke.get("checkpoint_sha256") != preaccess["checkpoint"]["sha256"]:
        raise ValueError("matching RHUH synthetic GPU smoke did not pass")
    output = p.RESULT_WORK / "RHUH_NO_N4_EXTERNAL.json"
    started = p.RESULT_WORK / "RHUH_NO_N4_EXTERNAL_STARTED.json"
    if output.exists() or started.exists():
        raise FileExistsError("refusing repeated RHUH external evaluation")
    data = RHUHDataset()
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RHUH external evaluation requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(1)
    model, payload, _ = load_model(device)
    p.RESULT_WORK.mkdir(parents=True, exist_ok=False)
    p.write_new(started, {
        "schema": "aflow.rhuh_no_n4_external.started",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preaccess_sha256": p.sha(p.PREACCESS),
        "cache_index_sha256": p.sha(p.CACHE / "index.json"),
        "checkpoint_sha256": preaccess["checkpoint"]["sha256"],
        "views": "Flip8",
        "threshold": 0.0,
        "external_metrics_access": True,
    })
    loader = DataLoader(data, batch_size=8, shuffle=False, num_workers=0)
    rows = []
    forwards = 0
    begin = time.monotonic()
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32)
        target = batch["label"].to(device=device, dtype=torch.bool)
        total = None
        for axes in p.AXES:
            view = torch.flip(image, axes) if axes else image
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(view).segmentation_logits.float()
            logits = torch.flip(logits, axes) if axes else logits
            total = logits.clone() if total is None else total.add_(logits)
            forwards += image.shape[0]
        logits = total / 8
        prediction = logits > 0
        for slot, case_id in enumerate(batch["case_id"]):
            pred = prediction[slot].flatten()
            truth = target[slot].flatten()
            pred_count, truth_count = int(pred.sum()), int(truth.sum())
            intersection = int((pred & truth).sum())
            point = int(logits[slot].flatten().argmax())
            rows.append({
                "case_id": str(case_id),
                "subject_id": str(batch["subject_id"][slot]),
                "dice": 2 * intersection / (pred_count + truth_count),
                "point_hit": bool(truth[point]),
                "point_flat_index": point,
                "predicted_voxels": pred_count,
                "target_voxels": truth_count,
                "intersection_voxels": intersection,
                "prediction_mask_sha256": hashlib.sha256(np.packbits(pred.cpu().numpy()).tobytes()).hexdigest(),
            })
    if forwards != 640:
        raise RuntimeError("RHUH fixed Flip8 forward count mismatch")
    summary = summarize(rows)
    intervals = bootstrap(rows)
    gates = {
        "scan_at_least_0p70": summary["scan_dice"] >= 0.70,
        "subject_at_least_0p70": summary["subject_dice"] >= 0.70,
        "point_hits_at_least_72": summary["point_hits"] >= 72,
    }
    report = {
        "schema": "aflow.rhuh_no_n4_external.result",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "one-time RHUH-GBM external confirmation; official timepoints1/2; 80 scans/40 subjects",
        "model": {
            "identifier": "full972_no_n4_seed20261343_epoch168_flip8",
            "checkpoint": preaccess["checkpoint"]["path"],
            "checkpoint_sha256": preaccess["checkpoint"]["sha256"],
            "epoch": payload["epoch"],
            "parameters": 4_700_897,
        },
        "protocol": preaccess["protocol"],
        "summary": summary,
        "bootstrap": intervals,
        "per_case": rows,
        "gates": gates,
        "all_absolute_gates_passed": all(gates.values()),
        "forwards": forwards,
        "elapsed_seconds": time.monotonic() - begin,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "preaccess_sha256": p.sha(p.PREACCESS),
        "cache_index_sha256": p.sha(p.CACHE / "index.json"),
        "evaluator_sha256": p.sha(Path(__file__)),
        "access": {"rhuh_external": True, "fixed_validation": False, "outer": False,
                   "confirmation": False, "test270": False, "ucsd75": False},
        "result_must_not_trigger_tuning_or_repeat": True,
        "grid_caveat": "Dice and Point are measured after image-only crop and nearest-exact label rasterization to 128^3, not native millimeter space",
    }
    p.write_new(output, report)
    print(json.dumps({"summary": summary, "bootstrap": intervals, "gates": gates}, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    main(parser.parse_args().device)
