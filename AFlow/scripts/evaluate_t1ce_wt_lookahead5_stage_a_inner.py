#!/usr/bin/env python3
"""One-shot fresh-inner evaluation of paired AdamW and Lookahead5 slow weights."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator  # noqa: E402
from scripts.evaluate_t1ce_wt_plain_ema_validation import compare  # noqa: E402
from scripts.train_t1ce_wt_capg3d import (  # noqa: E402
    BrainCropDataset,
    read_case_ids,
    sha256_file,
    subject_id,
    validate,
    write_json_atomic,
)
from scripts.train_t1ce_wt_voxel_kd_972_draw_budget import state_dict_sha256  # noqa: E402
import scripts.train_t1ce_wt_lookahead5_stage_a as train  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_lookahead5_stage_a_inner"
PREACCESS_SCHEMA = SCHEMA + ".preaccess"
FIT_SHA256 = "ee16a66d9a7efb8b3c5a14536d03bba7e5eb0eb984e4f333b314ec61570439b3"
INNER_SHA256 = "1bcffb7bc49f3199551510218419e1a7e074e055f5d2843a8731f4f349f08069"
SPLIT_SHA256 = "adf5d61280402d167d2eaa92f09b355d91385ee51d5b0f7e989cc7b7b21a096b"
CACHE_INDEX_SHA256 = "7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd"
EXPECTED_ACCESS = {"train_only_inner": False, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False}


def load_train_report(path: Path, arm: str, runner_sha: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if (
        value.get("schema") != train.SCHEMA
        or value.get("status") != "complete"
        or value.get("arm") != arm
        or value.get("seed") != train.SEED
        or value.get("smoke") is not False
        or value.get("all_gates_passed") is not True
        or value.get("runner_sha256") != runner_sha
        or value.get("access") != EXPECTED_ACCESS
        or len(value.get("history", [])) != 168
    ):
        raise ValueError("Lookahead Stage A train report contract mismatch")
    return value


def load_pairing(path: Path, arm: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if (
        value.get("schema") != train.SCHEMA + ".pairing_audit"
        or value.get("status") != "complete"
        or value.get("arm") != arm
        or value.get("seed") != train.SEED
        or value.get("smoke") is not False
        or value.get("all_gates_passed") is not True
        or value.get("access") != EXPECTED_ACCESS
        or value.get("sampler_epoch_count") != 168
        or len(value.get("sampler_epoch_sha256", [])) != 168
    ):
        raise ValueError("Lookahead Stage A pairing report contract mismatch")
    return value


def pairing_gates(pairings: dict[str, dict[str, Any]]) -> dict[str, bool]:
    control = pairings["adamw_control"]
    candidate = pairings["lookahead5_candidate"]
    optimizer = candidate["optimizer_audit"]
    return {
        "initial_state_exact": control["initial_state_sha256"] == candidate["initial_state_sha256"],
        "sampling_weights_exact": control["sampling_weight_sha256"] == candidate["sampling_weight_sha256"],
        "all_sampler_epochs_exact": control["sampler_epoch_sha256"] == candidate["sampler_epoch_sha256"],
        "first_augmented_batch_exact": control["first_augmented_batch"] == candidate["first_augmented_batch"],
        "candidate_fast_steps_exact": optimizer.get("fast_steps") == 20328,
        "candidate_slow_updates_exact": optimizer.get("slow_updates") == 4065,
        "candidate_trailing_fast_steps_exact": optimizer.get("trailing_fast_steps") == 3,
        "candidate_final_slow_loaded": optimizer.get("final_slow_loaded") is True,
        "candidate_optimizer_counts_passed": optimizer.get("all_counts_exact") is True,
    }


def load_model(path: Path, arm: str, report: dict[str, Any], device: torch.device) -> torch.nn.Module:
    if sha256_file(path) != report["checkpoints"]["raw"]["sha256"]:
        raise ValueError("Lookahead checkpoint SHA mismatch")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if (
        value.get("schema") != train.CHECKPOINT_SCHEMA
        or value.get("arm") != arm
        or value.get("epoch") != 168
        or value.get("kind") != "raw_epoch168"
        or value.get("access") != EXPECTED_ACCESS
        or value.get("model_state_sha256") != state_dict_sha256(value["model_state_dict"])
    ):
        raise ValueError("Lookahead checkpoint contract mismatch")
    factor = value.get("optimizer_factor", {})
    if arm == "lookahead5_candidate":
        if factor != {
            "outer": "Lookahead",
            "k": 5,
            "alpha": 0.5,
            "total_steps": 20328,
            "expected_slow_updates": 4065,
            "trailing_fast_steps": 3,
            "deployment_weights": "latest slow weights; trailing fast steps are not deployed",
        }:
            raise ValueError("candidate slow-weight deployment metadata mismatch")
    elif factor.get("outer") is not None:
        raise ValueError("control unexpectedly declares outer optimizer")
    model = AutomaticPromptLocator(center_aware=False, init_filters=16, dropout=0.1).to(device)
    model.load_state_dict(value["model_state_dict"], strict=True)
    return model.eval().requires_grad_(False)


def promotion_gates(control: dict[str, Any], candidate: dict[str, Any], comparison: dict[str, Any], engineering_ok: bool) -> dict[str, bool]:
    delta = comparison["ema_minus_raw"]
    return {
        "candidate_beats_control_scan": float(candidate["locator_dsc_mean"]) > float(control["locator_dsc_mean"]),
        "candidate_beats_control_subject": float(candidate["locator_subject_dsc_mean"]) > float(control["locator_subject_dsc_mean"]),
        "candidate_point_not_lower": float(candidate["automatic_point_hit_rate"]) >= float(control["automatic_point_hit_rate"]),
        "scan_ci_safe": float(delta["scan_95ci"][0]) >= -0.003,
        "subject_ci_safe": float(delta["subject_95ci"][0]) >= -0.003,
        "engineering_pairing_gates": engineering_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--inner-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    for arm in train.ARMS:
        parser.add_argument(f"--{arm}-report", type=Path, required=True)
        parser.add_argument(f"--{arm}-pairing", type=Path, required=True)
        parser.add_argument(f"--{arm}-checkpoint", type=Path, required=True)
    parser.add_argument("--preaccess-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = args.output.with_suffix(".started.json")
    if args.output.exists() or started.exists():
        raise FileExistsError("refusing repeated Lookahead Stage A inner evaluation")
    evaluator_sha = sha256_file(Path(__file__).resolve())
    for path, expected in {
        args.fit_manifest: FIT_SHA256,
        args.inner_manifest: INNER_SHA256,
        args.split_manifest: SPLIT_SHA256,
        args.cache / "index.json": CACHE_INDEX_SHA256,
    }.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen evaluation input SHA mismatch: {path}")
    runner_sha = sha256_file(ROOT / "scripts/train_t1ce_wt_lookahead5_stage_a.py")
    report_paths = {arm: getattr(args, f"{arm}_report") for arm in train.ARMS}
    pairing_paths = {arm: getattr(args, f"{arm}_pairing") for arm in train.ARMS}
    checkpoint_paths = {arm: getattr(args, f"{arm}_checkpoint") for arm in train.ARMS}
    reports = {arm: load_train_report(report_paths[arm], arm, runner_sha) for arm in train.ARMS}
    pairings = {arm: load_pairing(pairing_paths[arm], arm) for arm in train.ARMS}
    engineering = pairing_gates(pairings)
    if not all(engineering.values()):
        raise ValueError("paired training evidence failed before inner access")
    expected_hashes = {
        "report": {arm: sha256_file(path) for arm, path in report_paths.items()},
        "pairing": {arm: sha256_file(path) for arm, path in pairing_paths.items()},
        "checkpoint": {arm: sha256_file(path) for arm, path in checkpoint_paths.items()},
    }
    preaccess = json.loads(args.preaccess_receipt.read_text())
    if (
        preaccess.get("schema") != PREACCESS_SCHEMA
        or preaccess.get("status") != "frozen_before_train_only_inner"
        or preaccess.get("evaluator_sha256") != evaluator_sha
        or preaccess.get("runner_sha256") != runner_sha
        or preaccess.get("artifact_sha256") != expected_hashes
        or preaccess.get("access") != EXPECTED_ACCESS
    ):
        raise ValueError("Lookahead inner preaccess receipt mismatch")
    fit_ids = set(read_case_ids(args.fit_manifest))
    inner_ids = read_case_ids(args.inner_manifest)
    fit_subjects = {subject_id(case_id) for case_id in fit_ids}
    inner_subjects = {subject_id(case_id) for case_id in inner_ids}
    if fit_ids & set(inner_ids) or fit_subjects & inner_subjects:
        raise ValueError("fit/inner leakage")
    if len(fit_ids) != 864 or len(fit_subjects) != 392 or len(inner_ids) != 108 or len(inner_subjects) != 52:
        raise ValueError("fresh split counts mismatch")
    write_json_atomic(started, {
        "schema": SCHEMA + ".started",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preaccess_receipt_sha256": sha256_file(args.preaccess_receipt),
        "train_only_inner_access": True,
        "fixed_validation108_access": False,
        "outer_access": False,
        "confirmation_access": False,
        "test270_access": False,
    })
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Lookahead inner evaluation requires CUDA")
    torch.cuda.set_device(device)
    data = BrainCropDataset(args.cache, expected_split="train", case_ids=inner_ids)
    loader = DataLoader(data, batch_size=8, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    summaries: dict[str, Any] = {}
    for arm in train.ARMS:
        model = load_model(checkpoint_paths[arm], arm, reports[arm], device)
        summaries[arm] = validate(model, loader, device)
        del model
        torch.cuda.empty_cache()
    comparison = compare(summaries["adamw_control"], summaries["lookahead5_candidate"], seed=20261073, samples=10_000)
    gates = promotion_gates(summaries["adamw_control"], summaries["lookahead5_candidate"], comparison, all(engineering.values()))
    result = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summaries": summaries,
        "candidate_minus_control": comparison,
        "engineering_pairing_gates": engineering,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "full972_final_authorized": all(gates.values()),
        "inputs": {**expected_hashes, "preaccess_receipt_sha256": sha256_file(args.preaccess_receipt), "evaluator_sha256": evaluator_sha, "runner_sha256": runner_sha},
        "access": {"train_only_inner": True, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False},
    }
    write_json_atomic(args.output, result)
    print(json.dumps({
        "control": {"scan": summaries["adamw_control"]["locator_dsc_mean"], "subject": summaries["adamw_control"]["locator_subject_dsc_mean"], "point": summaries["adamw_control"]["automatic_point_hit_rate"]},
        "candidate": {"scan": summaries["lookahead5_candidate"]["locator_dsc_mean"], "subject": summaries["lookahead5_candidate"]["locator_subject_dsc_mean"], "point": summaries["lookahead5_candidate"]["automatic_point_hit_rate"]},
        "gates": gates,
        "full972_final_authorized": result["full972_final_authorized"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
