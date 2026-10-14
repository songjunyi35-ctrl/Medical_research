#!/usr/bin/env python3
"""One-time evaluation of two frozen locators on UCSD patient-disjoint external data."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator, automatic_prompt_points  # noqa: E402
from sammed3d.t1ce_wt_cached import CACHE_SCHEMA, MODEL_SHAPE  # noqa: E402
from scripts.prepare_ucsd_ptgbm_external_confirmation import (  # noqa: E402
    EXPECTED_ELIGIBLE_CASES,
    EXPECTED_ELIGIBLE_SUBJECTS,
    SCHEMA as PREPARE_SCHEMA,
    sha256_file,
    write_json_atomic,
)


SCHEMA = "aflow.ucsd_ptgbm.patient_disjoint_external_confirmation_evaluation"
PRIMARY_SHA256 = "f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b"
SECONDARY_SHA256 = "acf2fce112370e3cdb58fe295642680d5db1db3ceb619defdc47824ef4016d0d"
BOOTSTRAP_SEED = 2026090201
BOOTSTRAP_SAMPLES = 10_000


class ExternalEvaluationError(RuntimeError):
    pass


class ExternalCacheDataset(Dataset):
    def __init__(self, cache: Path, expected_cache_index_sha256: str) -> None:
        self.cache = cache.resolve()
        index_path = self.cache / "index.json"
        if sha256_file(index_path) != expected_cache_index_sha256:
            raise ExternalEvaluationError("external cache index SHA-256 mismatch")
        self.index = json.loads(index_path.read_text())
        if self.index.get("schema") != CACHE_SCHEMA or self.index.get("external_schema") != PREPARE_SCHEMA + ".cache":
            raise ExternalEvaluationError("unexpected external cache schema")
        if self.index.get("split") != "external_confirmation" or self.index.get("external_confirmation") is not True:
            raise ExternalEvaluationError("cache is not declared external confirmation")
        if int(self.index.get("count", -1)) != EXPECTED_ELIGIBLE_CASES or int(self.index.get("subjects", -1)) != EXPECTED_ELIGIBLE_SUBJECTS:
            raise ExternalEvaluationError("external cache cohort count mismatch")
        required = {
            "spatial": "braincrop_resize",
            "output_resolution": 128,
            "uses_gt_for_localization": False,
            "n4_bias_field_correction": True,
            "n4_shrink_factor": 2,
        }
        for key, value in required.items():
            if self.index.get("preprocessing", {}).get(key) != value:
                raise ExternalEvaluationError(f"external cache preprocessing mismatch: {key}")
        if self.index.get("target") != {"name": "WT", "foreground_raw_labels": [1, 2, 3], "excluded_raw_labels": [4]}:
            raise ExternalEvaluationError("external target contract mismatch")
        if sha256_file(self.cache / "images.f32") != self.index.get("images_sha256"):
            raise ExternalEvaluationError("external images tensor SHA-256 mismatch")
        if sha256_file(self.cache / "labels.u8") != self.index.get("labels_sha256"):
            raise ExternalEvaluationError("external labels tensor SHA-256 mismatch")
        self.case_ids = [str(value) for value in self.index["case_ids"]]
        self.subject_ids = [str(value) for value in self.index["subject_ids"]]
        if len(self.case_ids) != len(set(self.case_ids)) or len(set(self.subject_ids)) != EXPECTED_ELIGIBLE_SUBJECTS:
            raise ExternalEvaluationError("external cache IDs are not unique/expected")
        self._images: np.memmap | None = None
        self._labels: np.memmap | None = None

    def _open(self) -> None:
        if self._images is None:
            self._images = np.memmap(self.cache / "images.f32", dtype=np.float32, mode="r", shape=(len(self.case_ids), *MODEL_SHAPE))
            self._labels = np.memmap(self.cache / "labels.u8", dtype=np.uint8, mode="r", shape=(len(self.case_ids), *MODEL_SHAPE))

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, slot: int) -> dict[str, Any]:
        self._open()
        assert self._images is not None and self._labels is not None
        return {
            "image": torch.from_numpy(np.array(self._images[slot], dtype=np.float32))[None],
            "label": torch.from_numpy(np.array(self._labels[slot], dtype=np.float32))[None],
            "case_id": self.case_ids[slot],
            "subject_id": self.subject_ids[slot],
        }


def load_model(path: Path, expected_sha256: str, role: str, device: torch.device) -> tuple[AutomaticPromptLocator, dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise ExternalEvaluationError(f"{role} checkpoint SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "primary": {"arm": "full972_candidate", "epoch": 168},
        "secondary": {"arm": "gcsam_candidate", "epoch": 184},
    }[role]
    if payload.get("arm") != expected["arm"] or int(payload.get("epoch", -1)) != expected["epoch"]:
        raise ExternalEvaluationError(f"{role} checkpoint contract mismatch")
    if role == "secondary" and payload.get("window") != "post100":
        raise ExternalEvaluationError("secondary checkpoint is not the frozen post100 selection")
    architecture = payload.get("architecture", {})
    init_filters = int(architecture.get("init_filters", 16))
    dropout = float(architecture.get("dropout", 0.1))
    if bool(architecture.get("center_aware", False)):
        raise ExternalEvaluationError(f"{role} checkpoint unexpectedly has a center head")
    model = AutomaticPromptLocator(center_aware=False, init_filters=init_filters, dropout=dropout)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device).eval()
    if sum(parameter.numel() for parameter in model.parameters()) != 4_700_897:
        raise ExternalEvaluationError("model parameter count differs from frozen plain locator")
    return model, payload


@torch.inference_mode()
def evaluate_model(model: AutomaticPromptLocator, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    peak_before = torch.cuda.max_memory_allocated(device)
    started = time.time()
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        target = batch["label"].to(device=device, dtype=torch.bool, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(image)
        logits = output.segmentation_logits.float()
        prediction = logits > 0
        intersection = (prediction & target).flatten(1).sum(dim=1)
        prediction_voxels = prediction.flatten(1).sum(dim=1)
        target_voxels = target.flatten(1).sum(dim=1)
        denominator = prediction_voxels + target_voxels
        dice = torch.where(denominator > 0, 2.0 * intersection.float() / denominator.float(), torch.ones_like(intersection, dtype=torch.float32))
        points = automatic_prompt_points(output).long()
        for slot, case_id in enumerate(batch["case_id"]):
            z, y, x = (int(value) for value in points[slot, 0].tolist())
            positive = int(target_voxels[slot]) > 0
            point_hit = bool(target[slot, 0, z, y, x]) if positive else None
            cases.append({
                "case_id": str(case_id),
                "subject_id": str(batch["subject_id"][slot]),
                "locator_dsc": float(dice[slot]),
                "target_wt_voxels": int(target_voxels[slot]),
                "predicted_wt_voxels": int(prediction_voxels[slot]),
                "target_wt_positive": positive,
                "automatic_point": [z, y, x],
                "automatic_point_hits_gt": point_hit,
            })
    values = [float(case["locator_dsc"]) for case in cases]
    by_subject: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        by_subject[case["subject_id"]].append(float(case["locator_dsc"]))
    subject_values = [statistics.fmean(group) for group in by_subject.values()]
    positive_cases = [case for case in cases if case["target_wt_positive"]]
    negative_cases = [case for case in cases if not case["target_wt_positive"]]
    return {
        "cases": len(cases),
        "subjects": len(by_subject),
        "locator_dsc_mean": statistics.fmean(values),
        "locator_dsc_median": statistics.median(values),
        "locator_dsc_std_population": statistics.pstdev(values),
        "locator_subject_dsc_mean": statistics.fmean(subject_values),
        "wt_positive_cases": len(positive_cases),
        "wt_negative_cases": len(negative_cases),
        "automatic_point_hits": sum(bool(case["automatic_point_hits_gt"]) for case in positive_cases),
        "automatic_point_hit_rate_positive_only": statistics.fmean(float(case["automatic_point_hits_gt"]) for case in positive_cases) if positive_cases else None,
        "negative_predicted_empty_cases": sum(int(case["predicted_wt_voxels"]) == 0 for case in negative_cases),
        "negative_predicted_empty_rate": statistics.fmean(float(int(case["predicted_wt_voxels"]) == 0) for case in negative_cases) if negative_cases else None,
        "negative_false_positive_voxels_mean": statistics.fmean(float(case["predicted_wt_voxels"]) for case in negative_cases) if negative_cases else None,
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": max(0, torch.cuda.max_memory_allocated(device) - peak_before),
        "per_case": cases,
    }


def _bootstrap(values: list[float], rng: np.random.Generator) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    draws = rng.integers(0, len(array), size=(BOOTSTRAP_SAMPLES, len(array)))
    means = array[draws].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def bootstrap_absolute(summary: dict[str, Any], seed: int) -> dict[str, Any]:
    cases = summary["per_case"]
    by_subject: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        by_subject[case["subject_id"]].append(float(case["locator_dsc"]))
    rng = np.random.default_rng(seed)
    return {
        "scan_dsc_mean_95ci": _bootstrap([float(case["locator_dsc"]) for case in cases], rng),
        "subject_dsc_mean_95ci": _bootstrap([statistics.fmean(values) for values in by_subject.values()], rng),
    }


def compare(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    p_cases = {case["case_id"]: case for case in primary["per_case"]}
    s_cases = {case["case_id"]: case for case in secondary["per_case"]}
    if list(p_cases) != list(s_cases):
        raise ExternalEvaluationError("model reports have different case order")
    scan_delta = [float(p_cases[key]["locator_dsc"]) - float(s_cases[key]["locator_dsc"]) for key in p_cases]
    subject_scan: dict[str, list[float]] = defaultdict(list)
    for key in p_cases:
        subject_scan[p_cases[key]["subject_id"]].append(float(p_cases[key]["locator_dsc"]) - float(s_cases[key]["locator_dsc"]))
    subject_delta = [statistics.fmean(values) for values in subject_scan.values()]
    rng = np.random.default_rng(BOOTSTRAP_SEED + 2)
    primary_hits = int(primary["automatic_point_hits"])
    secondary_hits = int(secondary["automatic_point_hits"])
    return {
        "primary_minus_secondary": {
            "scan_dsc_mean": statistics.fmean(scan_delta),
            "subject_dsc_mean": statistics.fmean(subject_delta),
            "positive_point_hits": primary_hits - secondary_hits,
        },
        "paired_scan_dsc_mean_95ci": _bootstrap(scan_delta, rng),
        "paired_subject_dsc_mean_95ci": _bootstrap(subject_delta, rng),
        "scan_cases_improved": sum(value > 0 for value in scan_delta),
        "scan_cases_worse": sum(value < 0 for value in scan_delta),
        "scan_cases_tied": sum(value == 0 for value in scan_delta),
        "subjects_improved": sum(value > 0 for value in subject_delta),
        "subjects_worse": sum(value < 0 for value in subject_delta),
        "subjects_tied": sum(value == 0 for value in subject_delta),
    }


def gates(primary: dict[str, Any], comparison: dict[str, Any]) -> dict[str, bool]:
    delta = comparison["primary_minus_secondary"]
    values = {
        "primary_scan_at_least_0p70": float(primary["locator_dsc_mean"]) >= 0.70,
        "primary_subject_at_least_0p70": float(primary["locator_subject_dsc_mean"]) >= 0.70,
        "primary_positive_point_at_least_0p90": float(primary["automatic_point_hit_rate_positive_only"] or 0.0) >= 0.90,
        "primary_scan_not_below_secondary": float(delta["scan_dsc_mean"]) >= 0.0,
        "primary_subject_not_below_secondary": float(delta["subject_dsc_mean"]) >= 0.0,
        "paired_scan_ci_lower_at_least_minus_0p003": float(comparison["paired_scan_dsc_mean_95ci"][0]) >= -0.003,
        "paired_subject_ci_lower_at_least_minus_0p003": float(comparison["paired_subject_dsc_mean_95ci"][0]) >= -0.003,
        "primary_positive_point_loses_at_most_one": int(delta["positive_point_hits"]) >= -1,
    }
    values["all_performance_gates_passed"] = all(values.values())
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--expected-cache-index-sha256", required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    parser.add_argument("--preaccess-receipt", type=Path, required=True)
    parser.add_argument("--expected-preaccess-receipt-sha256", required=True)
    parser.add_argument("--primary-checkpoint", type=Path, required=True)
    parser.add_argument("--secondary-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    started_path = output.with_suffix(output.suffix + ".started.json")
    if output.exists() or started_path.exists():
        raise FileExistsError("refusing repeated external confirmation evaluation")
    if sha256_file(args.inventory.resolve()) != args.expected_inventory_sha256:
        raise ExternalEvaluationError("inventory SHA-256 mismatch")
    if sha256_file(args.preaccess_receipt.resolve()) != args.expected_preaccess_receipt_sha256:
        raise ExternalEvaluationError("preaccess receipt SHA-256 mismatch")
    preaccess = json.loads(args.preaccess_receipt.read_text())
    if preaccess.get("status") != "frozen_before_eligible_external_label_download" or preaccess.get("access", {}).get("eligible_label_voxel_access") is not False:
        raise ExternalEvaluationError("invalid preaccess receipt")
    data = ExternalCacheDataset(args.cache, args.expected_cache_index_sha256)
    if data.index.get("manifest_sha256") != args.expected_inventory_sha256:
        raise ExternalEvaluationError("cache is bound to a different inventory")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ExternalEvaluationError("formal external confirmation requires CUDA")
    torch.cuda.set_device(device)
    loader = DataLoader(data, batch_size=min(args.batch_size, len(data)), shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    primary_model, primary_payload = load_model(args.primary_checkpoint.resolve(), PRIMARY_SHA256, "primary", device)
    secondary_model, secondary_payload = load_model(args.secondary_checkpoint.resolve(), SECONDARY_SHA256, "secondary", device)
    write_json_atomic(started_path, {
        "schema": SCHEMA + ".started", "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inventory_sha256": args.expected_inventory_sha256,
        "cache_index_sha256": args.expected_cache_index_sha256,
        "preaccess_receipt_sha256": args.expected_preaccess_receipt_sha256,
        "primary_checkpoint_sha256": PRIMARY_SHA256,
        "secondary_checkpoint_sha256": SECONDARY_SHA256,
        "threshold": 0.0, "tta": False, "sam": False, "postprocessing": False,
        "external_confirmation_access": True,
    })
    primary = evaluate_model(primary_model, loader, device)
    del primary_model
    torch.cuda.empty_cache()
    secondary = evaluate_model(secondary_model, loader, device)
    comparison = compare(primary, secondary)
    performance_gates = gates(primary, comparison)
    report = {
        "schema": SCHEMA, "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "one-time patient-disjoint UCSD-PTGBM external confirmation; 75 scans/53 subjects; excludes every subject in existing BraTS 1,350-case union and excludes separate challenge test package",
        "preregistered": True,
        "not_test_tuned": True,
        "not_pristine_blind": False,
        "external_confirmation": True,
        "protocol": {
            "input": "T1 post-contrast only",
            "target": "WT={1,2,3}; resection cavity label4 excluded",
            "threshold": 0.0, "forwards_per_model": 1,
            "tta": False, "sam": False, "postprocessing": False, "ensemble": False,
            "point_denominator": "source-WT-positive scans only",
            "empty_target_policy": "retain all cases; Dice=1 only if both target and prediction are empty; separately report false positives",
            "bootstrap": {"method": "percentile", "samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED},
        },
        "data": {
            "inventory": str(args.inventory.resolve()), "inventory_sha256": args.expected_inventory_sha256,
            "cache": str(args.cache.resolve()), "cache_index_sha256": args.expected_cache_index_sha256,
            "images_sha256": data.index["images_sha256"], "labels_sha256": data.index["labels_sha256"],
            "cases": len(data), "subjects": len(set(data.subject_ids)),
            "existing_case_overlap": 0, "existing_subject_overlap": 0,
            "separate_brats_challenge_test_package_used": False,
        },
        "primary": {
            "identifier": "full972_random_init_voxel_kd_epoch168",
            "checkpoint": str(args.primary_checkpoint.resolve()), "checkpoint_sha256": PRIMARY_SHA256,
            "checkpoint_epoch": int(primary_payload["epoch"]), "summary": primary,
            "bootstrap": bootstrap_absolute(primary, BOOTSTRAP_SEED),
        },
        "secondary": {
            "identifier": "gcsam_fit864_inner_selected_epoch184",
            "checkpoint": str(args.secondary_checkpoint.resolve()), "checkpoint_sha256": SECONDARY_SHA256,
            "checkpoint_epoch": int(secondary_payload["epoch"]), "summary": secondary,
            "bootstrap": bootstrap_absolute(secondary, BOOTSTRAP_SEED + 1),
        },
        "comparison": comparison,
        "performance_gates": performance_gates,
        "data_engineering_gates": {"all_passed_before_inference": True},
        "result_must_not_trigger_tuning_or_repeat_external_access": True,
        "access": {"train": False, "fixed_validation": False, "outer": False, "internal_confirmation": False, "reused_test270": False, "external_confirmation": True},
        "evaluator": str(Path(__file__).resolve().relative_to(ROOT)),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_atomic(output, report)
    print(json.dumps({
        "primary": {"scan": primary["locator_dsc_mean"], "subject": primary["locator_subject_dsc_mean"], "point_positive_only": primary["automatic_point_hit_rate_positive_only"]},
        "secondary": {"scan": secondary["locator_dsc_mean"], "subject": secondary["locator_subject_dsc_mean"], "point_positive_only": secondary["automatic_point_hit_rate_positive_only"]},
        "comparison": comparison["primary_minus_secondary"], "gates": performance_gates,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
