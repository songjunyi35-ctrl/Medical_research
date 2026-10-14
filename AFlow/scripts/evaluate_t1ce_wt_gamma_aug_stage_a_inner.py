#!/usr/bin/env python3
"""One-shot fresh-inner evaluation of paired baseline and retained-statistics Gamma."""
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
from scripts.train_t1ce_wt_capg3d import BrainCropDataset, read_case_ids, sha256_file, subject_id, validate, write_json_atomic  # noqa: E402
from scripts.train_t1ce_wt_voxel_kd_972_draw_budget import state_dict_sha256  # noqa: E402
import scripts.train_t1ce_wt_gamma_aug_stage_a as train  # noqa: E402

SCHEMA = "aflow.sammed3d.t1ce_wt_gamma_aug_stage_a_inner"
PREACCESS_SCHEMA = SCHEMA + ".preaccess"
FIT_SHA256 = "f8efe7eb10f1b66b7bd3fbfc3b5c1286e63594c6e8f51be3c89469f6848bf8bb"
INNER_SHA256 = "798bc25a12b57f772b41cd01600f993ad414a2c71bb5325598fd2f926bc6c317"
SPLIT_SHA256 = "3287acd890e64aea5ad301ff7f7bf525f957845486249f0ad8aeac8e8e5613fc"
CACHE_INDEX_SHA256 = "7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd"
EXPECTED_ACCESS = {"train_only_inner": False, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False}


def load_report(path: Path, arm: str, runner_sha: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema") != train.SCHEMA or value.get("status") != "complete" or value.get("arm") != arm or value.get("seed") != train.SEED or value.get("smoke") is not False or value.get("all_gates_passed") is not True or value.get("runner_sha256") != runner_sha or value.get("access") != EXPECTED_ACCESS or len(value.get("history", [])) != 168:
        raise ValueError("Gamma train report contract mismatch")
    return value


def load_pairing(path: Path, arm: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema") != train.SCHEMA + ".pairing_audit" or value.get("status") != "complete" or value.get("arm") != arm or value.get("seed") != train.SEED or value.get("smoke") is not False or value.get("all_gates_passed") is not True or value.get("access") != EXPECTED_ACCESS or value.get("sampler_epoch_count") != 168 or len(value.get("sampler_epoch_sha256", [])) != 168:
        raise ValueError("Gamma pairing report contract mismatch")
    return value


def pairing_gates(pairings: dict[str, dict[str, Any]]) -> dict[str, bool]:
    control = pairings["baseline_control"]
    candidate = pairings["gamma_candidate"]
    gamma = candidate.get("gamma_audit") or {}
    return {
        "initial_state_exact": control["initial_state_sha256"] == candidate["initial_state_sha256"],
        "sampling_weights_exact": control["sampling_weight_sha256"] == candidate["sampling_weight_sha256"],
        "all_sampler_epochs_exact": control["sampler_epoch_sha256"] == candidate["sampler_epoch_sha256"],
        "first_pre_gamma_batch_exact": control["first_pre_gamma_batch"] == candidate["first_pre_gamma_batch"],
        "gamma_total_samples_exact": gamma.get("total_samples") == 162624,
        "gamma_fraction_ok": 0.25 <= float(gamma.get("applied_fraction", -1)) <= 0.35,
        "gamma_range_ok": 0.7 <= float(gamma.get("gamma_min", -1)) <= 1.5 and 0.7 <= float(gamma.get("gamma_max", -1)) <= 1.5 and float(gamma.get("gamma_min", 1)) != 1.0 and float(gamma.get("gamma_max", 1)) != 1.0,
        "gamma_statistics_preserved": float(gamma.get("max_mean_abs_error", 1)) <= 1e-4 and float(gamma.get("max_population_std_abs_error", 1)) <= 1e-4,
        "label_and_global_rng_unchanged": candidate["gamma_gates"].get("first_label_unchanged") is True and candidate["gamma_gates"].get("global_rng_unchanged_by_gamma") is True,
    }


def load_model(path: Path, arm: str, report: dict[str, Any], device: torch.device) -> torch.nn.Module:
    if sha256_file(path) != report["checkpoints"]["raw"]["sha256"]:
        raise ValueError("Gamma checkpoint SHA mismatch")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("schema") != train.CHECKPOINT_SCHEMA or value.get("arm") != arm or value.get("epoch") != 168 or value.get("kind") != "raw_epoch168" or value.get("access") != EXPECTED_ACCESS or value.get("model_state_sha256") != state_dict_sha256(value["model_state_dict"]):
        raise ValueError("Gamma checkpoint contract mismatch")
    factor = value.get("augmentation_factor", {})
    if arm == "gamma_candidate":
        expected = {"gamma": True, "probability": 0.3, "range": [0.7, 1.5], "invert": False, "retain_mean_population_std": True, "generator_seed": 20261181, "position": "after existing augmentation"}
        if factor != expected:
            raise ValueError("candidate Gamma metadata mismatch")
    elif factor != {"gamma": False}:
        raise ValueError("control Gamma metadata mismatch")
    model = AutomaticPromptLocator(center_aware=False, init_filters=16, dropout=0.1).to(device)
    model.load_state_dict(value["model_state_dict"], strict=True)
    return model.eval().requires_grad_(False)


def promotion_gates(control: dict[str, Any], candidate: dict[str, Any], comparison: dict[str, Any], engineering_ok: bool) -> dict[str, bool]:
    delta = comparison["ema_minus_raw"]
    return {
        "candidate_beats_control_scan": candidate["locator_dsc_mean"] > control["locator_dsc_mean"],
        "candidate_beats_control_subject": candidate["locator_subject_dsc_mean"] > control["locator_subject_dsc_mean"],
        "candidate_point_not_lower": candidate["automatic_point_hit_rate"] >= control["automatic_point_hit_rate"],
        "scan_ci_safe": delta["scan_95ci"][0] >= -0.003,
        "subject_ci_safe": delta["subject_95ci"][0] >= -0.003,
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
        raise FileExistsError("refusing repeated Gamma inner evaluation")
    evaluator_sha = sha256_file(Path(__file__).resolve())
    for path, expected in {args.fit_manifest: FIT_SHA256, args.inner_manifest: INNER_SHA256, args.split_manifest: SPLIT_SHA256, args.cache / "index.json": CACHE_INDEX_SHA256}.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen evaluation input SHA mismatch: {path}")
    runner_sha = sha256_file(ROOT / "scripts/train_t1ce_wt_gamma_aug_stage_a.py")
    report_paths = {arm: getattr(args, f"{arm}_report") for arm in train.ARMS}
    pairing_paths = {arm: getattr(args, f"{arm}_pairing") for arm in train.ARMS}
    checkpoint_paths = {arm: getattr(args, f"{arm}_checkpoint") for arm in train.ARMS}
    reports = {arm: load_report(report_paths[arm], arm, runner_sha) for arm in train.ARMS}
    pairings = {arm: load_pairing(pairing_paths[arm], arm) for arm in train.ARMS}
    engineering = pairing_gates(pairings)
    if not all(engineering.values()):
        raise ValueError("Gamma engineering pairing failed before inner access")
    hashes = {"report": {arm: sha256_file(path) for arm, path in report_paths.items()}, "pairing": {arm: sha256_file(path) for arm, path in pairing_paths.items()}, "checkpoint": {arm: sha256_file(path) for arm, path in checkpoint_paths.items()}}
    preaccess = json.loads(args.preaccess_receipt.read_text())
    if preaccess.get("schema") != PREACCESS_SCHEMA or preaccess.get("status") != "frozen_before_train_only_inner" or preaccess.get("evaluator_sha256") != evaluator_sha or preaccess.get("runner_sha256") != runner_sha or preaccess.get("artifact_sha256") != hashes or preaccess.get("access") != EXPECTED_ACCESS:
        raise ValueError("Gamma preaccess receipt mismatch")
    fit_ids = set(read_case_ids(args.fit_manifest)); inner_ids = read_case_ids(args.inner_manifest)
    fit_subjects = {subject_id(case_id) for case_id in fit_ids}; inner_subjects = {subject_id(case_id) for case_id in inner_ids}
    if fit_ids & set(inner_ids) or fit_subjects & inner_subjects or (len(fit_ids), len(fit_subjects), len(inner_ids), len(inner_subjects)) != (864, 392, 108, 52):
        raise ValueError("Gamma split count/leakage failed")
    write_json_atomic(started, {"schema": SCHEMA + ".started", "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "preaccess_receipt_sha256": sha256_file(args.preaccess_receipt), "train_only_inner_access": True, "fixed_validation108_access": False, "outer_access": False, "confirmation_access": False, "test270_access": False})
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("Gamma inner evaluation requires CUDA")
    torch.cuda.set_device(device)
    loader = DataLoader(BrainCropDataset(args.cache, expected_split="train", case_ids=inner_ids), batch_size=8, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    summaries = {}
    for arm in train.ARMS:
        model = load_model(checkpoint_paths[arm], arm, reports[arm], device)
        summaries[arm] = validate(model, loader, device)
        del model; torch.cuda.empty_cache()
    comparison = compare(summaries["baseline_control"], summaries["gamma_candidate"], seed=20261083, samples=10_000)
    gates = promotion_gates(summaries["baseline_control"], summaries["gamma_candidate"], comparison, all(engineering.values()))
    result = {"schema": SCHEMA, "schema_version": 1, "status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(), "summaries": summaries, "candidate_minus_control": comparison, "engineering_pairing_gates": engineering, "gamma_audit": pairings["gamma_candidate"]["gamma_audit"], "gates": gates, "all_gates_passed": all(gates.values()), "full972_final_authorized": all(gates.values()), "inputs": {**hashes, "preaccess_receipt_sha256": sha256_file(args.preaccess_receipt), "evaluator_sha256": evaluator_sha, "runner_sha256": runner_sha}, "access": {"train_only_inner": True, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False}}
    write_json_atomic(args.output, result)
    print(json.dumps({"control": {"scan": summaries["baseline_control"]["locator_dsc_mean"], "subject": summaries["baseline_control"]["locator_subject_dsc_mean"], "point": summaries["baseline_control"]["automatic_point_hit_rate"]}, "candidate": {"scan": summaries["gamma_candidate"]["locator_dsc_mean"], "subject": summaries["gamma_candidate"]["locator_subject_dsc_mean"], "point": summaries["gamma_candidate"]["automatic_point_hit_rate"]}, "gates": gates, "full972_final_authorized": result["full972_final_authorized"]}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
