#!/usr/bin/env python3
"""One-time readout for two fixed-epoch184 GCSAM train-only replications."""
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
import scripts.train_t1ce_wt_gcsam_fixed184_replication as train  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_gcsam_fixed184_replication_inner"
SPLIT_DIR = ROOT / "data/datasets/brats2024_t1ce_gcsam_fixed184_replications_20260903"
INNER = {
    "a": {"path": SPLIT_DIR / "replicate_a_inner_cases.jsonl", "sha256": "73068bcd2989bc84b87e99830e069620c0446a5c49c44e26010dd178bac6a3d3", "subjects": 47, "bootstrap_seed": 20261105},
    "b": {"path": SPLIT_DIR / "replicate_b_inner_cases.jsonl", "sha256": "6639a7c381296b17986fbac4ef25be981d378457412d2246bb87ff2a8cc2192c", "subjects": 45, "bootstrap_seed": 20261106},
}
ARMS = train.ARMS
BOOTSTRAP_SAMPLES = 10_000


def load_training_report(path: Path, replicate: str, arm: str, expected_epoch: int = 184) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema") != train.SCHEMA or value.get("status") != "complete":
        raise ValueError(f"invalid training report schema/status: {path}")
    if value.get("replicate") != replicate or value.get("arm") != arm or value.get("fixed_stop_epoch") != expected_epoch:
        raise ValueError(f"training report identity mismatch: {path}")
    if value.get("all_gates_passed") is not True or len(value.get("history", [])) != expected_epoch:
        raise ValueError(f"training report gates/epoch count failed: {path}")
    if value.get("runner_sha256") != sha256_file(Path(train.__file__).resolve()):
        raise ValueError("training runner SHA-256 mismatch")
    access = value.get("access", {})
    expected_access = {"fit": True, "inner_metrics": False, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False, "ucsd_external75": False}
    if access != expected_access or value.get("data_roles", {}).get("inner_dataset_constructed") is not False:
        raise ValueError("training report crossed inner/evaluation boundary")
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
    if checkpoint.get("access") != expected_access:
        raise ValueError("checkpoint crossed inner/evaluation boundary")
    return value


def verify_pair(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if control["replicate"] != candidate["replicate"]:
        raise ValueError("pair replicate mismatch")
    if control["initial_state_sha256"] != candidate["initial_state_sha256"]:
        raise ValueError("pair initial state mismatch")
    if control["sampling_weight_sha256"] != candidate["sampling_weight_sha256"]:
        raise ValueError("pair sampling weight mismatch")
    if control["first_batch_sha256"] != candidate["first_batch_sha256"] or control["first_batch_loss"] != candidate["first_batch_loss"]:
        raise ValueError("pair first augmented batch/loss mismatch")
    control_history = control["history"]
    candidate_history = candidate["history"]
    if len(control_history) != len(candidate_history):
        raise ValueError("pair epoch count mismatch")
    mismatched_epochs = [
        left["epoch"] for left, right in zip(control_history, candidate_history)
        if left["epoch"] != right["epoch"] or left["sampler_index_sha256"] != right["sampler_index_sha256"]
    ]
    if mismatched_epochs:
        raise ValueError(f"pair sampler mismatch at epochs: {mismatched_epochs[:5]}")
    return {
        "initial_state_exact": True,
        "sampling_weight_exact": True,
        "first_augmented_batch_exact": True,
        "first_pass_loss_exact": True,
        "all_epoch_sampler_indices_exact": True,
        "epochs_compared": len(control_history),
    }


def preflight_training_artifacts(work: Path, expected_epoch: int = 184) -> dict[str, Any]:
    reports: dict[str, dict[str, Any]] = {}
    pairs = {}
    for replicate in ("a", "b"):
        for arm in ARMS:
            key = f"{replicate}_{arm}"
            path = work / f"formal_{replicate}_{arm}/TRAIN_DONE.json"
            reports[key] = load_training_report(path, replicate, arm, expected_epoch=expected_epoch)
        pairs[replicate] = verify_pair(reports[f"{replicate}_adamw_control"], reports[f"{replicate}_gcsam_candidate"])
    if reports["a_adamw_control"]["initial_state_sha256"] == reports["b_adamw_control"]["initial_state_sha256"]:
        raise ValueError("replicate A/B unexpectedly share the same random initial state")
    return {"reports": reports, "pairing": pairs}


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
        raise ValueError("paired inner reports have different case order")
    scan_delta = [float(right[key]["locator_dsc"]) - float(left[key]["locator_dsc"]) for key in left]
    by_subject: dict[str, list[float]] = defaultdict(list)
    for key in left:
        if left[key]["subject_id"] != right[key]["subject_id"]:
            raise ValueError("paired inner subject mismatch")
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
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--preaccess", type=Path, required=True)
    parser.add_argument("--expected-preaccess-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.with_suffix(output.suffix + ".started.json").exists():
        raise FileExistsError("refusing repeated inner evaluation")
    if sha256_file(args.preaccess) != args.expected_preaccess_sha256:
        raise ValueError("inner preaccess receipt SHA-256 mismatch")
    receipt = json.loads(args.preaccess.read_text())
    if receipt.get("status") != "frozen_after_four_trainings_before_inner_access" or receipt.get("access", {}).get("inner_metrics") is not False:
        raise ValueError("invalid inner preaccess receipt")
    if receipt.get("evaluator_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("preaccess evaluator SHA-256 mismatch")
    preflight = preflight_training_artifacts(args.work_dir.resolve())
    for key, report in preflight["reports"].items():
        frozen = receipt["training_artifacts"][key]
        report_path = args.work_dir.resolve() / f"formal_{key}/TRAIN_DONE.json"
        if sha256_file(report_path) != frozen["report_sha256"] or report["checkpoint"]["sha256"] != frozen["checkpoint_sha256"]:
            raise ValueError(f"preaccess training artifact mismatch: {key}")
    if sha256_file(args.cache / "index.json") != train.CACHE_INDEX_SHA256:
        raise ValueError("train cache index mismatch")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("inner evaluation requires CUDA")
    torch.cuda.set_device(device)
    write_json_atomic(output.with_suffix(output.suffix + ".started.json"), {
        "schema": SCHEMA + ".started", "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preaccess_sha256": args.expected_preaccess_sha256,
        "inner_access": True, "fixed_validation108": False, "test270": False, "ucsd_external75": False,
    })
    results: dict[str, Any] = {}
    deltas = []
    subject_deltas = []
    all_replication_gates = True
    for replicate in ("a", "b"):
        inner = INNER[replicate]
        if sha256_file(inner["path"]) != inner["sha256"]:
            raise ValueError(f"replicate {replicate} inner manifest SHA mismatch")
        case_ids = read_case_ids(inner["path"])
        data = BrainCropDataset(args.cache, expected_split="train", case_ids=case_ids)
        if len(data) != 108 or len({train.base.subject_id(case_id) for case_id in data.case_ids}) != inner["subjects"]:
            raise ValueError("inner case/subject count mismatch")
        loader = DataLoader(data, batch_size=8, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
        arm_results = {}
        for arm in ARMS:
            report = preflight["reports"][f"{replicate}_{arm}"]
            model = load_model(report, device)
            arm_results[arm] = validate(model, loader, device)
            del model
            torch.cuda.empty_cache()
        comparison = compare(arm_results["adamw_control"], arm_results["gcsam_candidate"], inner["bootstrap_seed"])
        gates = replication_gates(comparison)
        all_replication_gates = all_replication_gates and gates["all_passed"]
        deltas.append(comparison["candidate_minus_control"]["scan_dice"])
        subject_deltas.append(comparison["candidate_minus_control"]["subject_dice"])
        results[replicate] = {"cohort": {"cases": 108, "subjects": inner["subjects"], "manifest": str(inner["path"]), "manifest_sha256": inner["sha256"]}, "arms": arm_results, "comparison": comparison, "gates": gates}
    cross = {"mean_scan_delta": statistics.fmean(deltas), "mean_subject_delta": statistics.fmean(subject_deltas)}
    cross_gates = {"mean_scan_delta_positive": cross["mean_scan_delta"] > 0, "mean_subject_delta_positive": cross["mean_subject_delta"] > 0}
    stage_b_authorized = all_replication_gates and all(cross_gates.values())
    report = {
        "schema": SCHEMA, "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "fixed_epoch": 184,
        "replications": results,
        "pairing": preflight["pairing"],
        "cross_replication": {**cross, "gates": cross_gates},
        "stage_b_authorized": stage_b_authorized,
        "result_must_not_trigger_rho_epoch_or_test_tuning": True,
        "access": {"fit": True, "inner_metrics": True, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False, "ucsd_external75": False},
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_atomic(output, report)
    print(json.dumps({"replications": {key: {"comparison": value["comparison"]["candidate_minus_control"], "ci": {"scan": value["comparison"]["scan_dice_95ci"], "subject": value["comparison"]["subject_dice_95ci"]}, "gates": value["gates"]} for key, value in results.items()}, "cross_replication": report["cross_replication"], "stage_b_authorized": stage_b_authorized}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
