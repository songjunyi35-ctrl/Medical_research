#!/usr/bin/env python3
"""One-shot fresh-inner evaluation of three seed teachers and fixed consensus."""
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
    dice_per_case,
    read_case_ids,
    sha256_file,
    subject_id,
    summarize_cases,
    write_json_atomic,
)
from scripts.train_t1ce_wt_voxel_kd_972_draw_budget import state_dict_sha256  # noqa: E402
import scripts.train_t1ce_wt_consensus_seed_teacher as train  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_consensus_teachers_inner"
PREACCESS_SCHEMA = SCHEMA + ".preaccess"
INNER_SHA256 = "6f3cd273e3605e4845fec1d5281a0baa71cfda6db0afbdcd0f4bff18c025ea5c"
FIT_SHA256 = "f46a2158fb9cbc7a797321cee13bdd97bf372750c4146b4a0dbe89e8f0e7edb8"
SPLIT_SHA256 = "b88fe33fd353035d271afbea0fb8e3c349f78b3206a60fcdacae2654c7cc6dae"
CACHE_INDEX_SHA256 = "7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd"
TEACHER_ARMS = train.ARMS


def consensus_logits(logits: list[torch.Tensor]) -> torch.Tensor:
    if len(logits) != 3:
        raise ValueError("consensus requires exactly three teacher logits")
    if not all(value.shape == logits[0].shape for value in logits):
        raise ValueError("teacher logit shapes differ")
    if not all(bool(torch.isfinite(value).all()) for value in logits):
        raise FloatingPointError("non-finite teacher logits")
    return torch.stack(logits, dim=0).mean(dim=0)


def cases_from_logits(
    logits: torch.Tensor,
    label: torch.Tensor,
    case_ids: list[str],
    subject_ids: list[str],
) -> list[dict[str, Any]]:
    dsc = dice_per_case(logits.float(), label)
    flat = logits[:, 0].flatten(1)
    indices = flat.argmax(dim=1)
    depth, height, width = logits.shape[-3:]
    cases: list[dict[str, Any]] = []
    for batch_index, flat_index in enumerate(indices.tolist()):
        z = int(flat_index // (height * width))
        remainder = int(flat_index % (height * width))
        y = int(remainder // width)
        x = int(remainder % width)
        cases.append({
            "case_id": str(case_ids[batch_index]),
            "subject_id": str(subject_ids[batch_index]),
            "locator_dsc": float(dsc[batch_index]),
            "automatic_point": [z, y, x],
            "automatic_point_hits_gt": bool(label[batch_index, 0, z, y, x] > 0.5),
        })
    return cases


def load_report(path: Path, arm: str, runner_sha: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if (
        value.get("schema") != train.SCHEMA
        or value.get("status") != "complete"
        or value.get("arm") != arm
        or value.get("all_gates_passed") is not True
        or value.get("runner_sha256") != runner_sha
    ):
        raise ValueError("teacher report contract mismatch")
    if value.get("access") != {"train_only_inner": False, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False}:
        raise ValueError("teacher report access seal mismatch")
    return value


def load_model(path: Path, arm: str, report: dict[str, Any], device: torch.device) -> torch.nn.Module:
    if sha256_file(path) != report["checkpoint"]["sha256"]:
        raise ValueError("teacher checkpoint SHA mismatch")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if (
        value.get("schema") != train.CHECKPOINT_SCHEMA
        or value.get("arm") != arm
        or value.get("epoch") != train.STOP_EPOCH
    ):
        raise ValueError("teacher checkpoint metadata mismatch")
    if value.get("model_state_sha256") != state_dict_sha256(value["model_state_dict"]):
        raise ValueError("teacher checkpoint state hash mismatch")
    if value.get("access") != {"train_only_inner": False, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False}:
        raise ValueError("teacher checkpoint access seal mismatch")
    model = AutomaticPromptLocator(center_aware=False, init_filters=16, dropout=0.1).to(device)
    model.load_state_dict(value["model_state_dict"], strict=True)
    return model.eval().requires_grad_(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--inner-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    for arm in TEACHER_ARMS:
        parser.add_argument(f"--{arm}-report", type=Path, required=True)
        parser.add_argument(f"--{arm}-checkpoint", type=Path, required=True)
    parser.add_argument("--preaccess-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    started = args.output.with_suffix(".started.json")
    if args.output.exists() or started.exists():
        raise FileExistsError("refusing repeated consensus teacher inner evaluation")
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

    report_paths = {arm: getattr(args, f"{arm}_report") for arm in TEACHER_ARMS}
    checkpoint_paths = {arm: getattr(args, f"{arm}_checkpoint") for arm in TEACHER_ARMS}
    runner_sha = sha256_file(ROOT / "scripts/train_t1ce_wt_consensus_seed_teacher.py")
    reports = {arm: load_report(report_paths[arm], arm, runner_sha) for arm in TEACHER_ARMS}
    preaccess = json.loads(args.preaccess_receipt.read_text())
    if preaccess.get("schema") != PREACCESS_SCHEMA or preaccess.get("status") != "frozen_before_train_only_inner":
        raise ValueError("invalid consensus inner preaccess receipt")
    if preaccess.get("evaluator_sha256") != evaluator_sha or preaccess.get("runner_sha256") != runner_sha:
        raise ValueError("preaccess code hash mismatch")
    if preaccess.get("report_sha256") != {arm: sha256_file(path) for arm, path in report_paths.items()}:
        raise ValueError("preaccess report hash mismatch")
    if preaccess.get("checkpoint_sha256") != {arm: sha256_file(path) for arm, path in checkpoint_paths.items()}:
        raise ValueError("preaccess checkpoint hash mismatch")
    if preaccess.get("access") != {"train_only_inner": False, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False}:
        raise ValueError("preaccess role seal mismatch")

    fit_ids = set(read_case_ids(args.fit_manifest))
    inner_ids = read_case_ids(args.inner_manifest)
    if fit_ids & set(inner_ids):
        raise ValueError("fit/inner case leakage")
    fit_subjects = {subject_id(case_id) for case_id in fit_ids}
    inner_subjects = {subject_id(case_id) for case_id in inner_ids}
    if fit_subjects & inner_subjects:
        raise ValueError("fit/inner subject leakage")
    if len(inner_ids) != 108 or len(inner_subjects) != 43:
        raise ValueError("fresh inner case/subject count mismatch")

    write_json_atomic(started, {
        "schema": SCHEMA + ".started",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preaccess_receipt_sha256": sha256_file(args.preaccess_receipt),
        "evaluator_sha256": evaluator_sha,
        "consensus": "equal raw-logit mean of three fixed teachers",
        "train_only_inner_access": True,
        "fixed_validation108_access": False,
        "outer_access": False,
        "confirmation_access": False,
        "test270_access": False,
    })

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("consensus inner evaluation requires CUDA")
    torch.cuda.set_device(device)
    models = {arm: load_model(checkpoint_paths[arm], arm, reports[arm], device) for arm in TEACHER_ARMS}
    data = BrainCropDataset(args.cache, expected_split="train", case_ids=inner_ids)
    loader = DataLoader(data, batch_size=8, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    all_cases: dict[str, list[dict[str, Any]]] = {arm: [] for arm in TEACHER_ARMS}
    all_cases["consensus"] = []
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            teacher_logits: list[torch.Tensor] = []
            for arm in TEACHER_ARMS:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = models[arm](image).segmentation_logits.float()
                teacher_logits.append(logits)
                all_cases[arm].extend(cases_from_logits(logits, label, list(batch["case_id"]), list(batch["subject_id"])))
            mean_logits = consensus_logits(teacher_logits)
            all_cases["consensus"].extend(cases_from_logits(mean_logits, label, list(batch["case_id"]), list(batch["subject_id"])))

    summaries = {name: summarize_cases(cases) for name, cases in all_cases.items()}
    comparisons = {
        arm: compare(summaries[arm], summaries["consensus"], seed=20261036, samples=10_000)
        for arm in TEACHER_ARMS
    }
    ensemble = summaries["consensus"]
    gates: dict[str, bool] = {}
    for arm in TEACHER_ARMS:
        delta = comparisons[arm]["ema_minus_raw"]
        gates[f"consensus_beats_{arm}_scan"] = float(delta["scan"]) > 0.0
        gates[f"consensus_beats_{arm}_subject"] = float(delta["subject"]) > 0.0
        gates[f"consensus_vs_{arm}_scan_ci_safe"] = float(delta["scan_95ci"][0]) >= -0.003
        gates[f"consensus_vs_{arm}_subject_ci_safe"] = float(delta["subject_95ci"][0]) >= -0.003
    gates["consensus_point_at_least_best_individual"] = float(ensemble["automatic_point_hit_rate"]) >= max(
        float(summaries[arm]["automatic_point_hit_rate"]) for arm in TEACHER_ARMS
    )
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summaries": summaries,
        "consensus_minus_individual": comparisons,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "student_stage_authorized": all(gates.values()),
        "inputs": {
            "report_sha256": {arm: sha256_file(path) for arm, path in report_paths.items()},
            "checkpoint_sha256": {arm: sha256_file(path) for arm, path in checkpoint_paths.items()},
            "preaccess_receipt_sha256": sha256_file(args.preaccess_receipt),
            "evaluator_sha256": evaluator_sha,
        },
        "access": {"train_only_inner": True, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False},
    }
    write_json_atomic(args.output, report)
    print(json.dumps({
        "metrics": {name: {"scan": value["locator_dsc_mean"], "subject": value["locator_subject_dsc_mean"], "point": value["automatic_point_hit_rate"]} for name, value in summaries.items()},
        "gates": gates,
        "student_stage_authorized": report["student_stage_authorized"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

