#!/usr/bin/env python3
"""One-time C/D fresh-inner evaluation of paired flip-consistency training."""
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

import scripts.evaluate_t1ce_wt_flip8_tta_replication_inner as flip8  # noqa: E402
from scripts.train_t1ce_wt_capg3d import BrainCropDataset, read_case_ids, sha256_file, write_json_atomic  # noqa: E402
import scripts.train_t1ce_wt_flip_equivariance_consistency as train  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_flip_equivariance_consistency_inner"
BOOTSTRAP_SAMPLES = 10_000
EXPECTED_TRAIN_ACCESS = train.EXPECTED_ACCESS


def load_split(path: Path) -> dict[str, Any]:
    if sha256_file(path) != train.SPLIT_MANIFEST_SHA256:
        raise ValueError("split manifest SHA mismatch")
    split = json.loads(path.read_text())
    if split.get("schema") != train.SPLIT_SCHEMA:
        raise ValueError("split schema mismatch")
    if split.get("documents") != {"idea_sha256": train.IDEA_SHA256, "prereg_sha256": train.PREREG_SHA256}:
        raise ValueError("split document binding mismatch")
    if split.get("cross_replication") != {"inner_case_overlap": 0, "inner_subject_overlap": 0}:
        raise ValueError("C/D inner are not mutually disjoint")
    expected = {"train_only_inner_metrics": False, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False, "ucsd_external75": False}
    if split.get("access") != expected:
        raise ValueError("split access boundary changed")
    return split


def load_training_report(path: Path, replicate: str, arm: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema") != train.SCHEMA or value.get("status") != "complete":
        raise ValueError(f"invalid training report: {path}")
    if value.get("replicate") != replicate or value.get("arm") != arm:
        raise ValueError("training identity mismatch")
    if value.get("consistency_weight") != train.CONSISTENCY_WEIGHT[arm] or value.get("fixed_stop_epoch") != train.FORMAL_EPOCHS:
        raise ValueError("training method/epoch mismatch")
    history = value.get("history", [])
    if value.get("all_gates_passed") is not True or len(history) != train.FORMAL_EPOCHS:
        raise ValueError("training gates/epoch count failed")
    if value.get("runner_sha256") != sha256_file(Path(train.__file__).resolve()):
        raise ValueError("training runner SHA mismatch")
    if value.get("access") != EXPECTED_TRAIN_ACCESS or value.get("data_roles", {}).get("inner_dataset_constructed") is not False:
        raise ValueError("training crossed inner boundary")
    if any(row.get("inner_metrics") is not None for row in history):
        raise ValueError("training history contains inner metrics")
    checkpoint_path = Path(value["checkpoint"]["path"])
    if sha256_file(checkpoint_path) != value["checkpoint"]["sha256"]:
        raise ValueError("checkpoint SHA mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema") != train.CHECKPOINT_SCHEMA
        or checkpoint.get("replicate") != replicate
        or checkpoint.get("arm") != arm
        or checkpoint.get("epoch") != train.FORMAL_EPOCHS
        or checkpoint.get("consistency_weight") != train.CONSISTENCY_WEIGHT[arm]
    ):
        raise ValueError("checkpoint contract mismatch")
    if checkpoint.get("model_state_sha256") != train.base.state_dict_sha256(checkpoint["model_state_dict"]):
        raise ValueError("checkpoint model-state SHA mismatch")
    if checkpoint.get("documents") != {
        "idea_sha256": train.IDEA_SHA256,
        "prereg_sha256": train.PREREG_SHA256,
        "amendment_sha256": train.AMENDMENT_SHA256,
        "split_manifest_sha256": train.SPLIT_MANIFEST_SHA256,
    }:
        raise ValueError("checkpoint document binding mismatch")
    if checkpoint.get("access") != EXPECTED_TRAIN_ACCESS:
        raise ValueError("checkpoint access boundary mismatch")
    return value


def preflight_training_artifacts(work: Path) -> dict[str, dict[str, dict[str, Any]]]:
    reports = {
        replicate: {
            arm: load_training_report(work / f"formal_{replicate}_{arm}/TRAIN_DONE.json", replicate, arm)
            for arm in train.ARMS
        }
        for replicate in train.REPLICATES
    }
    if reports["c"]["control_w0"]["initial_state_sha256"] == reports["d"]["control_w0"]["initial_state_sha256"]:
        raise ValueError("C/D unexpectedly share initial state")
    return reports


def pairing_gates(pair: dict[str, dict[str, Any]]) -> dict[str, bool]:
    control = pair["control_w0"]
    candidate = pair["candidate_w01"]
    control_history = control["history"]
    candidate_history = candidate["history"]
    gates = {
        "initial_state_exact": control["initial_state_sha256"] == candidate["initial_state_sha256"],
        "sampling_weights_exact": control["sampling_weight_sha256"] == candidate["sampling_weight_sha256"],
        "first_augmented_batch_exact": control["first_batch_sha256"] == candidate["first_batch_sha256"],
        "first_axes_exact": control["first_step"]["axes"] == candidate["first_step"]["axes"],
        "first_existing_loss_exact": control["first_step"]["existing_loss"] == candidate["first_step"]["existing_loss"],
        "first_consistency_loss_exact": control["first_step"]["consistency_loss"] == candidate["first_step"]["consistency_loss"],
        "all_sampler_indices_exact": [row["sampler_index_sha256"] for row in control_history] == [row["sampler_index_sha256"] for row in candidate_history],
        "all_augmentation_rng_before_exact": [row["augmentation_rng_before_sha256"] for row in control_history] == [row["augmentation_rng_before_sha256"] for row in candidate_history],
        "all_augmentation_rng_after_exact": [row["augmentation_rng_after_sha256"] for row in control_history] == [row["augmentation_rng_after_sha256"] for row in candidate_history],
        "all_reflection_schedules_exact": [row["reflection_schedule_sha256"] for row in control_history] == [row["reflection_schedule_sha256"] for row in candidate_history],
        "all_two_forwards": all(row["student_forwards_per_step"] == 2 for report in pair.values() for row in report["history"]),
        "all_training_gates_passed": all(report["all_gates_passed"] for report in pair.values()),
    }
    gates["all_passed"] = all(gates.values())
    return gates


def load_model(report: dict[str, Any], device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(report["checkpoint"]["path"], map_location="cpu", weights_only=False)
    model = train.base.AutomaticPromptLocator(center_aware=False, init_filters=16, dropout=0.1)
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
        raise ValueError("control/candidate case order mismatch")
    scan = [float(right[key]["locator_dsc"]) - float(left[key]["locator_dsc"]) for key in left]
    by_subject: dict[str, list[float]] = defaultdict(list)
    for key in left:
        if left[key]["subject_id"] != right[key]["subject_id"]:
            raise ValueError("control/candidate subject mismatch")
        by_subject[left[key]["subject_id"]].append(float(right[key]["locator_dsc"]) - float(left[key]["locator_dsc"]))
    subject = [statistics.fmean(values) for values in by_subject.values()]
    point = sum(bool(row["automatic_point_hits_gt"]) for row in candidate["per_case"]) - sum(bool(row["automatic_point_hits_gt"]) for row in control["per_case"])
    return {
        "candidate_minus_control": {"scan_dice": statistics.fmean(scan), "subject_dice": statistics.fmean(subject), "point_hits": point},
        "scan_dice_95ci": _bootstrap(scan, seed),
        "subject_dice_95ci": _bootstrap(subject, seed + 100),
        "scans_improved_worse_tied": [sum(x > 0 for x in scan), sum(x < 0 for x in scan), sum(x == 0 for x in scan)],
        "subjects_improved_worse_tied": [sum(x > 0 for x in subject), sum(x < 0 for x in subject), sum(x == 0 for x in subject)],
    }


def primary_gates(comparison: dict[str, Any]) -> dict[str, bool]:
    delta = comparison["candidate_minus_control"]
    return {
        "scan_strictly_positive": float(delta["scan_dice"]) > 0,
        "subject_strictly_positive": float(delta["subject_dice"]) > 0,
        "point_loses_at_most_one": int(delta["point_hits"]) >= -1,
        "scan_ci_lower_at_least_minus_0p003": float(comparison["scan_dice_95ci"][0]) >= -0.003,
        "subject_ci_lower_at_least_minus_0p003": float(comparison["subject_dice_95ci"][0]) >= -0.003,
    }


def identity_gates(comparison: dict[str, Any]) -> dict[str, bool]:
    delta = comparison["candidate_minus_control"]
    return {
        "identity_scan_strictly_positive": float(delta["scan_dice"]) > 0,
        "identity_subject_strictly_positive": float(delta["subject_dice"]) > 0,
        "identity_point_loses_at_most_one": int(delta["point_hits"]) >= -1,
    }


def mechanism(pair: dict[str, dict[str, Any]]) -> dict[str, Any]:
    control = statistics.fmean(row["train_consistency_loss"] for row in pair["control_w0"]["history"][-5:])
    candidate = statistics.fmean(row["train_consistency_loss"] for row in pair["candidate_w01"]["history"][-5:])
    ratio = candidate / control
    return {"epochs": [164, 165, 166, 167, 168], "control_mean": control, "candidate_mean": candidate, "candidate_over_control": ratio, "at_most_0p90": ratio <= 0.90}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--preaccess", type=Path, required=True)
    parser.add_argument("--expected-preaccess-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    started = output.with_suffix(output.suffix + ".started.json")
    if output.exists() or started.exists():
        raise FileExistsError("refusing repeated C/D inner evaluation")
    if sha256_file(args.preaccess) != args.expected_preaccess_sha256:
        raise ValueError("preaccess SHA mismatch")
    receipt = json.loads(args.preaccess.read_text())
    if receipt.get("status") != "frozen_after_four_trainings_before_inner_access" or receipt.get("access", {}).get("inner_metrics") is not False:
        raise ValueError("invalid preaccess receipt")
    if receipt.get("evaluator_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("preaccess evaluator SHA mismatch")
    split = load_split(args.split_manifest.resolve())
    if receipt.get("split_manifest_sha256") != train.SPLIT_MANIFEST_SHA256:
        raise ValueError("preaccess split binding mismatch")
    if sha256_file(args.cache / "index.json") != prior_cache_sha():
        raise ValueError("train cache index mismatch")
    reports = preflight_training_artifacts(args.work_dir.resolve())
    engineering: dict[str, dict[str, bool]] = {}
    for replicate, pair in reports.items():
        engineering[replicate] = pairing_gates(pair)
        if not engineering[replicate]["all_passed"]:
            raise ValueError(f"replicate {replicate} pairing gate failed before inner access")
        for arm, report in pair.items():
            frozen = receipt["training_artifacts"][replicate][arm]
            report_path = args.work_dir.resolve() / f"formal_{replicate}_{arm}/TRAIN_DONE.json"
            if sha256_file(report_path) != frozen["report_sha256"] or report["checkpoint"]["sha256"] != frozen["checkpoint_sha256"]:
                raise ValueError("training artifact changed after preaccess")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("inner evaluation requires CUDA")
    torch.cuda.set_device(device)
    write_json_atomic(started, {
        "schema": SCHEMA + ".started", "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preaccess_sha256": args.expected_preaccess_sha256,
        "inner_access": True, "fixed_validation108": False, "outer": False,
        "confirmation": False, "test270": False, "ucsd_external75": False,
    })
    results: dict[str, Any] = {}
    cross: dict[str, list[float]] = {"flip8_scan": [], "flip8_subject": [], "identity_scan": [], "identity_subject": []}
    all_replicate_gates = True
    for replicate in train.REPLICATES:
        inner_record = split["replications"][replicate]["inner"]
        inner_path = Path(inner_record["path"])
        if sha256_file(inner_path) != inner_record["sha256"]:
            raise ValueError("inner manifest changed")
        case_ids = read_case_ids(inner_path)
        data = BrainCropDataset(args.cache, expected_split="train", case_ids=case_ids)
        if len(data) != 108 or len({train.base.subject_id(case_id) for case_id in data.case_ids}) != int(inner_record["subjects"]):
            raise ValueError("inner cohort mismatch")
        loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
        arms: dict[str, Any] = {}
        for arm in train.ARMS:
            model = load_model(reports[replicate][arm], device)
            arms[arm] = flip8.evaluate_model(model, loader, device)
            del model
            torch.cuda.empty_cache()
        seed = int(split["replications"][replicate]["bootstrap_seed"])
        identity_comparison = compare(arms["control_w0"]["identity"], arms["candidate_w01"]["identity"], seed)
        flip8_comparison = compare(arms["control_w0"]["flip8"], arms["candidate_w01"]["flip8"], seed + 1000)
        primary = primary_gates(flip8_comparison)
        identity = identity_gates(identity_comparison)
        mech = mechanism(reports[replicate])
        gates = {**primary, **identity, "mechanism_last5_ratio_at_most_0p90": mech["at_most_0p90"], "engineering_pairing": engineering[replicate]["all_passed"]}
        gates["all_passed"] = all(gates.values())
        all_replicate_gates = all_replicate_gates and gates["all_passed"]
        fdelta = flip8_comparison["candidate_minus_control"]
        idelta = identity_comparison["candidate_minus_control"]
        cross["flip8_scan"].append(float(fdelta["scan_dice"]))
        cross["flip8_subject"].append(float(fdelta["subject_dice"]))
        cross["identity_scan"].append(float(idelta["scan_dice"]))
        cross["identity_subject"].append(float(idelta["subject_dice"]))
        results[replicate] = {"cohort": inner_record, "arms": arms, "identity_comparison": identity_comparison, "flip8_comparison": flip8_comparison, "mechanism": mech, "engineering": engineering[replicate], "gates": gates}
        del loader, data
        torch.cuda.empty_cache()
    means = {key: statistics.fmean(values) for key, values in cross.items()}
    cross_gates = {
        "mean_flip8_scan_above_0p001": means["flip8_scan"] > 0.001,
        "mean_flip8_subject_positive": means["flip8_subject"] > 0,
        "mean_identity_scan_positive": means["identity_scan"] > 0,
        "mean_identity_subject_positive": means["identity_subject"] > 0,
    }
    all_passed = all_replicate_gates and all(cross_gates.values())
    report = {
        "schema": SCHEMA, "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "complete",
        "fixed_epoch": train.FORMAL_EPOCHS,
        "training_factor": {"control_weight": 0.0, "candidate_weight": 0.1, "amendment_sha256": train.AMENDMENT_SHA256},
        "inference": {"identity_forwards": 1, "flip8_forwards": 8, "threshold": 0.0, "postprocessing": False},
        "replications": results,
        "cross_replication": {"means": means, "gates": cross_gates},
        "all_gates_passed": all_passed,
        "fresh_external_preregistration_eligible": all_passed,
        "existing_validation_test_or_external_access_authorized": False,
        "result_must_not_trigger_weight_loss_view_epoch_threshold_or_seed_tuning": True,
        "access": {"fit": True, "inner_metrics": True, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False, "ucsd_external75": False},
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_atomic(output, report)
    print(json.dumps({"replications": {key: {"identity_delta": value["identity_comparison"]["candidate_minus_control"], "flip8_delta": value["flip8_comparison"]["candidate_minus_control"], "mechanism": value["mechanism"], "gates": value["gates"]} for key, value in results.items()}, "cross_replication": report["cross_replication"], "all_gates_passed": all_passed, "revealed_cohort_access": False}, indent=2, sort_keys=True))


def prior_cache_sha() -> str:
    return train.prior.CONTROL_CACHE_INDEX_SHA256


if __name__ == "__main__":
    main()
