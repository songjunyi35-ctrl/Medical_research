#!/usr/bin/env python3
"""One-time readout for two fixed-epoch168 percentile-clipping replications."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator  # noqa: E402
from scripts.train_t1ce_wt_capg3d import BrainCropDataset, read_case_ids, sha256_file, validate, write_json_atomic  # noqa: E402
import scripts.train_t1ce_wt_percentile_clip_replication as train  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_percentile_clip_replication_inner"
ARMS = train.ARMS
BOOTSTRAP_SAMPLES = 10_000
EXPECTED_ACCESS = {"fit": True, "inner_metrics": False, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False, "ucsd_external75": False}


def load_split(path: Path) -> dict[str, Any]:
    split = json.loads(path.read_text())
    if split.get("schema") != "aflow.brats2024.t1ce_wt_percentile_clip_replication_splits":
        raise ValueError("split schema mismatch")
    if split.get("documents") != {"idea_sha256": train.IDEA_SHA256, "prereg_sha256": train.PREREG_SHA256}:
        raise ValueError("split document binding mismatch")
    return split


def load_training_report(path: Path, replicate: str, arm: str, expected_epoch: int = train.FORMAL_EPOCHS) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema") != train.SCHEMA or value.get("status") != "complete":
        raise ValueError(f"invalid training report: {path}")
    if value.get("replicate") != replicate or value.get("arm") != arm or value.get("fixed_stop_epoch") != expected_epoch:
        raise ValueError(f"training identity mismatch: {path}")
    if value.get("all_gates_passed") is not True or len(value.get("history", [])) != expected_epoch:
        raise ValueError(f"training gates/epochs failed: {path}")
    if value.get("runner_sha256") != sha256_file(Path(train.__file__).resolve()):
        raise ValueError("training runner SHA-256 mismatch")
    if value.get("access") != EXPECTED_ACCESS or value.get("data_roles", {}).get("inner_dataset_constructed") is not False:
        raise ValueError("training crossed evaluation boundary")
    if any(row.get("inner_metrics") is not None for row in value["history"]):
        raise ValueError("training history contains inner metrics")
    checkpoint_path = Path(value["checkpoint"]["path"])
    if sha256_file(checkpoint_path) != value["checkpoint"]["sha256"]:
        raise ValueError("checkpoint SHA-256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != train.CHECKPOINT_SCHEMA or checkpoint.get("replicate") != replicate or checkpoint.get("arm") != arm or checkpoint.get("epoch") != expected_epoch:
        raise ValueError("checkpoint identity mismatch")
    if checkpoint.get("model_state_sha256") != train.base.state_dict_sha256(checkpoint["model_state_dict"]):
        raise ValueError("checkpoint model-state SHA-256 mismatch")
    if checkpoint.get("access") != EXPECTED_ACCESS:
        raise ValueError("checkpoint crossed evaluation boundary")
    return value


def verify_pair(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if control["replicate"] != candidate["replicate"]:
        raise ValueError("pair replicate mismatch")
    if control["initial_state_sha256"] != candidate["initial_state_sha256"]:
        raise ValueError("pair initial state mismatch")
    if control["sampling_weight_sha256"] != candidate["sampling_weight_sha256"]:
        raise ValueError("pair sampling weights mismatch")
    left_cache = control["data_roles"]["input_cache"]
    right_cache = candidate["data_roles"]["input_cache"]
    if not (
        left_cache["labels_sha256"]
        == right_cache["labels_sha256"]
        == train.CONTROL_LABELS_SHA256
    ):
        raise ValueError("pair cache labels mismatch")
    if left_cache["images_sha256"] == right_cache["images_sha256"]:
        raise ValueError("candidate image cache unexpectedly equals control")
    if control["first_batch_sha256"]["label"] != candidate["first_batch_sha256"]["label"]:
        raise ValueError("pair first augmented label mismatch")
    if control["first_batch_sha256"]["image"] == candidate["first_batch_sha256"]["image"]:
        raise ValueError("pair first image did not express preprocessing difference")
    left_history = control["history"]
    right_history = candidate["history"]
    if len(left_history) != len(right_history):
        raise ValueError("pair epoch count mismatch")
    mismatches = []
    for left, right in zip(left_history, right_history):
        if (
            left["epoch"] != right["epoch"]
            or left["sampler_index_sha256"] != right["sampler_index_sha256"]
            or left["augmentation_rng_before_sha256"] != right["augmentation_rng_before_sha256"]
            or left["augmentation_rng_after_sha256"] != right["augmentation_rng_after_sha256"]
        ):
            mismatches.append(left["epoch"])
    if mismatches:
        raise ValueError(f"pair sampler/augmentation mismatch: {mismatches[:5]}")
    return {
        "initial_state_exact": True,
        "sampling_weight_exact": True,
        "cache_labels_exact": True,
        "cache_images_intentionally_different": True,
        "first_augmented_label_exact": True,
        "all_epoch_sampler_indices_exact": True,
        "all_epoch_augmentation_rng_exact": True,
        "epochs_compared": len(left_history),
    }


def preflight_training_artifacts(work: Path) -> dict[str, Any]:
    reports: dict[str, dict[str, Any]] = {}
    pairing: dict[str, Any] = {}
    for replicate in ("a", "b"):
        for arm in ARMS:
            key = f"{replicate}_{arm}"
            reports[key] = load_training_report(
                work / f"formal_{replicate}_{arm}/TRAIN_DONE.json", replicate, arm
            )
        pairing[replicate] = verify_pair(
            reports[f"{replicate}_control"], reports[f"{replicate}_percentile_clip"]
        )
    if reports["a_control"]["initial_state_sha256"] == reports["b_control"]["initial_state_sha256"]:
        raise ValueError("replications unexpectedly share initial state")
    return {"reports": reports, "pairing": pairing}


def load_model(report: dict[str, Any], device: torch.device) -> AutomaticPromptLocator:
    checkpoint = torch.load(report["checkpoint"]["path"], map_location="cpu", weights_only=False)
    model = AutomaticPromptLocator(center_aware=False, init_filters=16, dropout=0.1)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def _bootstrap(values: list[float], seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(BOOTSTRAP_SAMPLES, len(array)))
    return [float(value) for value in np.quantile(array[draws].mean(axis=1), (0.025, 0.975))]


def compare(control: dict[str, Any], candidate: dict[str, Any], seed: int) -> dict[str, Any]:
    left = {row["case_id"]: row for row in control["per_case"]}
    right = {row["case_id"]: row for row in candidate["per_case"]}
    if list(left) != list(right):
        raise ValueError("paired reports have different case order")
    scan_delta = [float(right[key]["locator_dsc"]) - float(left[key]["locator_dsc"]) for key in left]
    by_subject: dict[str, list[float]] = defaultdict(list)
    for key in left:
        if left[key]["subject_id"] != right[key]["subject_id"]:
            raise ValueError("paired subject mismatch")
        by_subject[left[key]["subject_id"]].append(float(right[key]["locator_dsc"]) - float(left[key]["locator_dsc"]))
    subject_delta = [statistics.fmean(values) for values in by_subject.values()]
    point_delta = sum(bool(row["automatic_point_hits_gt"]) for row in candidate["per_case"]) - sum(bool(row["automatic_point_hits_gt"]) for row in control["per_case"])
    return {
        "candidate_minus_control": {"scan_dice": statistics.fmean(scan_delta), "subject_dice": statistics.fmean(subject_delta), "point_hits": point_delta},
        "scan_dice_95ci": _bootstrap(scan_delta, seed),
        "subject_dice_95ci": _bootstrap(subject_delta, seed + 100),
        "scan_improved": sum(value > 0 for value in scan_delta),
        "scan_worse": sum(value < 0 for value in scan_delta),
        "scan_tied": sum(value == 0 for value in scan_delta),
        "subjects_improved": sum(value > 0 for value in subject_delta),
        "subjects_worse": sum(value < 0 for value in subject_delta),
        "subjects_tied": sum(value == 0 for value in subject_delta),
    }


def replication_gates(comparison: dict[str, Any]) -> dict[str, bool]:
    delta = comparison["candidate_minus_control"]
    gates = {
        "scan_strictly_positive": float(delta["scan_dice"]) > 0,
        "subject_strictly_positive": float(delta["subject_dice"]) > 0,
        "point_loses_at_most_one": int(delta["point_hits"]) >= -1,
        "scan_ci_lower_at_least_minus_0p003": float(comparison["scan_dice_95ci"][0]) >= -0.003,
        "subject_ci_lower_at_least_minus_0p003": float(comparison["subject_dice_95ci"][0]) >= -0.003,
    }
    gates["all_passed"] = all(gates.values())
    return gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--control-cache", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--preaccess", type=Path, required=True)
    parser.add_argument("--expected-preaccess-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    started_output = output.with_suffix(output.suffix + ".started.json")
    if output.exists() or started_output.exists():
        raise FileExistsError("refusing repeated inner evaluation")
    if sha256_file(args.preaccess) != args.expected_preaccess_sha256:
        raise ValueError("preaccess SHA-256 mismatch")
    receipt = json.loads(args.preaccess.read_text())
    if receipt.get("status") != "frozen_after_four_trainings_before_inner_access" or receipt.get("access", {}).get("inner_metrics") is not False:
        raise ValueError("invalid preaccess receipt")
    if receipt.get("evaluator_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("preaccess evaluator SHA-256 mismatch")
    split = load_split(args.split_manifest.resolve())
    if sha256_file(args.split_manifest) != receipt["split_manifest_sha256"]:
        raise ValueError("split manifest changed after preaccess")
    caches = {"control": args.control_cache.resolve(), "percentile_clip": args.candidate_cache.resolve()}
    for arm, cache in caches.items():
        if sha256_file(cache / "index.json") != receipt["caches"][arm]["index_sha256"]:
            raise ValueError(f"{arm} cache index changed after preaccess")
    preflight = preflight_training_artifacts(args.work_dir.resolve())
    for key, report in preflight["reports"].items():
        frozen = receipt["training_artifacts"][key]
        report_path = args.work_dir.resolve() / f"formal_{key}/TRAIN_DONE.json"
        if sha256_file(report_path) != frozen["report_sha256"] or report["checkpoint"]["sha256"] != frozen["checkpoint_sha256"]:
            raise ValueError(f"training artifact changed after preaccess: {key}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("inner evaluation requires CUDA")
    torch.cuda.set_device(device)
    write_json_atomic(started_output, {
        "schema": SCHEMA + ".started", "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preaccess_sha256": args.expected_preaccess_sha256,
        "inner_access": True, "fixed_validation108": False, "outer": False,
        "confirmation": False, "test270": False, "ucsd_external75": False,
    })
    results: dict[str, Any] = {}
    scan_deltas: list[float] = []
    subject_deltas: list[float] = []
    all_replication_gates = True
    for replicate in ("a", "b"):
        inner = split["replications"][replicate]["inner"]
        inner_path = Path(inner["path"])
        if sha256_file(inner_path) != inner["sha256"]:
            raise ValueError(f"replicate {replicate} inner manifest changed")
        case_ids = read_case_ids(inner_path)
        arm_results: dict[str, Any] = {}
        for arm in ARMS:
            data = BrainCropDataset(caches[arm], expected_split="train", case_ids=case_ids)
            if len(data) != 108 or len({train.base.subject_id(case_id) for case_id in data.case_ids}) != int(inner["subjects"]):
                raise ValueError("inner case/subject count mismatch")
            loader = DataLoader(data, batch_size=8, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
            model = load_model(preflight["reports"][f"{replicate}_{arm}"], device)
            arm_results[arm] = validate(model, loader, device)
            del model, loader, data
            torch.cuda.empty_cache()
        comparison = compare(arm_results["control"], arm_results["percentile_clip"], int(split["replications"][replicate]["bootstrap_seed"]))
        gates = replication_gates(comparison)
        all_replication_gates = all_replication_gates and gates["all_passed"]
        scan_deltas.append(float(comparison["candidate_minus_control"]["scan_dice"]))
        subject_deltas.append(float(comparison["candidate_minus_control"]["subject_dice"]))
        results[replicate] = {"cohort": inner, "arms": arm_results, "comparison": comparison, "gates": gates}
    cross = {"mean_scan_delta": statistics.fmean(scan_deltas), "mean_subject_delta": statistics.fmean(subject_deltas)}
    cross_gates = {"mean_scan_delta_positive": cross["mean_scan_delta"] > 0, "mean_subject_delta_positive": cross["mean_subject_delta"] > 0}
    stage_b_authorized = all_replication_gates and all(cross_gates.values())
    report = {
        "schema": SCHEMA, "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "complete",
        "fixed_epoch": train.FORMAL_EPOCHS, "replications": results,
        "pairing": preflight["pairing"],
        "cross_replication": {**cross, "gates": cross_gates},
        "stage_b_authorized": stage_b_authorized,
        "result_must_not_trigger_percentile_epoch_or_test_tuning": True,
        "access": {"fit": True, "inner_metrics": True, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False, "ucsd_external75": False},
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_atomic(output, report)
    print(json.dumps({
        "replications": {key: {"comparison": value["comparison"]["candidate_minus_control"], "ci": {"scan": value["comparison"]["scan_dice_95ci"], "subject": value["comparison"]["subject_dice_95ci"]}, "gates": value["gates"]} for key, value in results.items()},
        "cross_replication": report["cross_replication"],
        "stage_b_authorized": stage_b_authorized,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
