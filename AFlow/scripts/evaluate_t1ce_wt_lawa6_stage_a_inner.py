#!/usr/bin/env python3
"""One-shot fresh-inner evaluation of two raw168/LAWA6 trajectory pairs."""
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
from scripts.evaluate_t1ce_wt_consensus_teachers_inner import cases_from_logits  # noqa: E402
from scripts.evaluate_t1ce_wt_plain_ema_validation import compare  # noqa: E402
from scripts.train_t1ce_wt_capg3d import (  # noqa: E402
    BrainCropDataset,
    read_case_ids,
    sha256_file,
    subject_id,
    summarize_cases,
    write_json_atomic,
)
from scripts.train_t1ce_wt_voxel_kd_972_draw_budget import state_dict_sha256  # noqa: E402
import scripts.train_t1ce_wt_lawa6_stage_a as train  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_lawa6_stage_a_inner"
PREACCESS_SCHEMA = SCHEMA + ".preaccess"
INNER_SHA256 = "331c0c79baad1c459a635ab27088c1210512593959851c2faa8ad8d7e1955f60"
FIT_SHA256 = "bf741b71121a9523047926fab6b6e3e7b880f7696adeec349cb6fbf01dbf1a7f"
SPLIT_SHA256 = "5e6f938ff40c32682748e249a7f355a90ad75ecb1d7a3a77038b0614fc781cdc"
CACHE_INDEX_SHA256 = "7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd"
EXPECTED_ACCESS = {
    "train_only_inner": False,
    "fixed_validation108": False,
    "outer": False,
    "confirmation": False,
    "test270": False,
}


def load_train_report(path: Path, arm: str, runner_sha: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if (
        value.get("schema") != train.SCHEMA
        or value.get("status") != "complete"
        or value.get("arm") != arm
        or value.get("smoke") is not False
        or value.get("all_gates_passed") is not True
        or value.get("runner_sha256") != runner_sha
        or value.get("access") != EXPECTED_ACCESS
    ):
        raise ValueError("LAWA6 Stage A train report contract mismatch")
    if value.get("lawa", {}).get("captured_epochs") != [163, 164, 165, 166, 167, 168]:
        raise ValueError("formal LAWA6 capture epochs mismatch")
    return value


def load_pair_model(
    path: Path,
    *,
    arm: str,
    kind: str,
    report: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    report_key = "raw" if kind == "raw_epoch168" else "lawa6"
    if sha256_file(path) != report["checkpoints"][report_key]["sha256"]:
        raise ValueError("checkpoint SHA mismatch")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if (
        value.get("schema") != train.CHECKPOINT_SCHEMA
        or value.get("arm") != arm
        or value.get("kind") != kind
        or value.get("epoch") != 168
        or value.get("access") != EXPECTED_ACCESS
    ):
        raise ValueError("checkpoint metadata/access mismatch")
    if value.get("lawa_contract", {}).get("captured_epochs") != [163, 164, 165, 166, 167, 168]:
        raise ValueError("checkpoint LAWA capture contract mismatch")
    if value.get("model_state_sha256") != state_dict_sha256(value["model_state_dict"]):
        raise ValueError("checkpoint state hash mismatch")
    model = AutomaticPromptLocator(center_aware=False, init_filters=16, dropout=0.1).to(device)
    model.load_state_dict(value["model_state_dict"], strict=True)
    return model.eval().requires_grad_(False)


def promotion_gates(
    summaries: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
) -> dict[str, bool]:
    gates: dict[str, bool] = {}
    for arm in train.ARMS:
        raw_key = f"{arm}_raw"
        lawa_key = f"{arm}_lawa6"
        delta = comparisons[arm]["ema_minus_raw"]
        gates[f"{arm}_lawa_beats_raw_scan"] = float(delta["scan"]) > 0.0
        gates[f"{arm}_lawa_beats_raw_subject"] = float(delta["subject"]) > 0.0
        gates[f"{arm}_lawa_point_not_lower"] = float(summaries[lawa_key]["automatic_point_hit_rate"]) >= float(
            summaries[raw_key]["automatic_point_hit_rate"]
        )
        gates[f"{arm}_scan_ci_safe"] = float(delta["scan_95ci"][0]) >= -0.003
        gates[f"{arm}_subject_ci_safe"] = float(delta["subject_95ci"][0]) >= -0.003
    return gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--inner-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    for arm in train.ARMS:
        parser.add_argument(f"--{arm}-report", type=Path, required=True)
        parser.add_argument(f"--{arm}-raw-checkpoint", type=Path, required=True)
        parser.add_argument(f"--{arm}-lawa-checkpoint", type=Path, required=True)
    parser.add_argument("--preaccess-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    started = args.output.with_suffix(".started.json")
    if args.output.exists() or started.exists():
        raise FileExistsError("refusing repeated LAWA6 Stage A inner evaluation")
    evaluator_sha = sha256_file(Path(__file__).resolve())
    frozen = {
        args.fit_manifest: FIT_SHA256,
        args.inner_manifest: INNER_SHA256,
        args.split_manifest: SPLIT_SHA256,
        args.cache / "index.json": CACHE_INDEX_SHA256,
    }
    for path, expected in frozen.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen evaluation input SHA mismatch: {path}")

    report_paths = {arm: getattr(args, f"{arm}_report") for arm in train.ARMS}
    raw_paths = {arm: getattr(args, f"{arm}_raw_checkpoint") for arm in train.ARMS}
    lawa_paths = {arm: getattr(args, f"{arm}_lawa_checkpoint") for arm in train.ARMS}
    runner_sha = sha256_file(ROOT / "scripts/train_t1ce_wt_lawa6_stage_a.py")
    reports = {arm: load_train_report(report_paths[arm], arm, runner_sha) for arm in train.ARMS}
    preaccess = json.loads(args.preaccess_receipt.read_text())
    if preaccess.get("schema") != PREACCESS_SCHEMA or preaccess.get("status") != "frozen_before_train_only_inner":
        raise ValueError("invalid LAWA6 Stage A preaccess receipt")
    if preaccess.get("evaluator_sha256") != evaluator_sha or preaccess.get("runner_sha256") != runner_sha:
        raise ValueError("preaccess code hash mismatch")
    expected_report_hashes = {arm: sha256_file(path) for arm, path in report_paths.items()}
    expected_checkpoint_hashes = {
        arm: {"raw": sha256_file(raw_paths[arm]), "lawa6": sha256_file(lawa_paths[arm])}
        for arm in train.ARMS
    }
    if preaccess.get("report_sha256") != expected_report_hashes:
        raise ValueError("preaccess report hash mismatch")
    if preaccess.get("checkpoint_sha256") != expected_checkpoint_hashes:
        raise ValueError("preaccess checkpoint hash mismatch")
    if preaccess.get("access") != EXPECTED_ACCESS:
        raise ValueError("preaccess role seal mismatch")

    fit_ids = set(read_case_ids(args.fit_manifest))
    inner_ids = read_case_ids(args.inner_manifest)
    fit_subjects = {subject_id(case_id) for case_id in fit_ids}
    inner_subjects = {subject_id(case_id) for case_id in inner_ids}
    if fit_ids & set(inner_ids) or fit_subjects & inner_subjects:
        raise ValueError("fit/inner case or subject leakage")
    if len(fit_ids) != 864 or len(fit_subjects) != 393 or len(inner_ids) != 108 or len(inner_subjects) != 51:
        raise ValueError("fresh split case/subject count mismatch")

    write_json_atomic(started, {
        "schema": SCHEMA + ".started",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preaccess_receipt_sha256": sha256_file(args.preaccess_receipt),
        "evaluator_sha256": evaluator_sha,
        "comparison": "within-run raw epoch168 versus uniform epoch163-168 LAWA6",
        "train_only_inner_access": True,
        "fixed_validation108_access": False,
        "outer_access": False,
        "confirmation_access": False,
        "test270_access": False,
    })

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("LAWA6 Stage A inner evaluation requires CUDA")
    torch.cuda.set_device(device)
    models: dict[str, torch.nn.Module] = {}
    for arm in train.ARMS:
        models[f"{arm}_raw"] = load_pair_model(
            raw_paths[arm], arm=arm, kind="raw_epoch168", report=reports[arm], device=device
        )
        models[f"{arm}_lawa6"] = load_pair_model(
            lawa_paths[arm], arm=arm, kind="lawa6_epochs163_168", report=reports[arm], device=device
        )
    data = BrainCropDataset(args.cache, expected_split="train", case_ids=inner_ids)
    loader = DataLoader(data, batch_size=8, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    all_cases: dict[str, list[dict[str, Any]]] = {name: [] for name in models}
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            for name, model in models.items():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(image).segmentation_logits.float()
                all_cases[name].extend(
                    cases_from_logits(logits, label, list(batch["case_id"]), list(batch["subject_id"]))
                )

    summaries = {name: summarize_cases(cases) for name, cases in all_cases.items()}
    comparisons = {
        arm: compare(summaries[f"{arm}_raw"], summaries[f"{arm}_lawa6"], seed=20261064, samples=10_000)
        for arm in train.ARMS
    }
    gates = promotion_gates(summaries, comparisons)
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summaries": summaries,
        "lawa6_minus_raw": comparisons,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "full972_final_authorized": all(gates.values()),
        "inputs": {
            "report_sha256": expected_report_hashes,
            "checkpoint_sha256": expected_checkpoint_hashes,
            "preaccess_receipt_sha256": sha256_file(args.preaccess_receipt),
            "evaluator_sha256": evaluator_sha,
            "runner_sha256": runner_sha,
        },
        "access": {
            "train_only_inner": True,
            "fixed_validation108": False,
            "outer": False,
            "confirmation": False,
            "test270": False,
        },
    }
    write_json_atomic(args.output, report)
    print(json.dumps({
        "metrics": {
            name: {
                "scan": value["locator_dsc_mean"],
                "subject": value["locator_subject_dsc_mean"],
                "point": value["automatic_point_hit_rate"],
            }
            for name, value in summaries.items()
        },
        "gates": gates,
        "full972_final_authorized": report["full972_final_authorized"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
