#!/usr/bin/env python3
"""One-shot fixed-validation evaluation for GT-best teacher routing students."""
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
    validate,
    write_json_atomic,
)
from scripts.train_t1ce_wt_voxel_kd_972_draw_budget import state_dict_sha256  # noqa: E402
import scripts.train_t1ce_wt_gt_best_teacher_routing as train  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_gt_best_teacher_routing_validation"
PREACCESS_SCHEMA = SCHEMA + ".preaccess"
VALIDATION_MANIFEST_SHA256 = "f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb"
VALIDATION_CACHE_INDEX_SHA256 = "9339e4575561447dd5bd99bc29e909ae1e9f566721d351a244d087a40622f322"
ARCHIVED_SEED32_REPORT_SHA256 = "30cd65c78ad1ed2a2b5748652f48ec05cde3b54468b44a66eba77d062d3ce129"
ARMS = train.ARMS


def load_report(path: Path, arm: str, runner_sha: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if (
        value.get("schema") != train.SCHEMA
        or value.get("status") != "complete"
        or value.get("arm") != arm
        or value.get("all_gates_passed") is not True
        or value.get("runner_sha256") != runner_sha
        or len(value.get("history", [])) != 168
    ):
        raise ValueError("routing training report contract mismatch")
    if value.get("access") != {"train_only_inner": False, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False}:
        raise ValueError("routing report access seal mismatch")
    return value


def load_model(path: Path, arm: str, report: dict[str, Any], device: torch.device) -> torch.nn.Module:
    if sha256_file(path) != report["checkpoint"]["sha256"]:
        raise ValueError("routing checkpoint file SHA mismatch")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("schema") != train.CHECKPOINT_SCHEMA or value.get("arm") != arm or value.get("epoch") != 168:
        raise ValueError("routing checkpoint metadata mismatch")
    if value.get("model_state_sha256") != state_dict_sha256(value["model_state_dict"]):
        raise ValueError("routing checkpoint state hash mismatch")
    if value.get("access") != {"train_only_inner": False, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False}:
        raise ValueError("routing checkpoint access seal mismatch")
    model = AutomaticPromptLocator(center_aware=False, init_filters=16, dropout=0.1).to(device)
    model.load_state_dict(value["model_state_dict"], strict=True)
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--control-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--archived-seed32-report", type=Path, required=True)
    parser.add_argument("--implementation-receipt", type=Path, required=True)
    parser.add_argument("--preaccess-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    started = args.output.with_suffix(".started.json")
    if args.output.exists() or started.exists():
        raise FileExistsError("refusing repeated routing validation")
    evaluator_sha = sha256_file(Path(__file__).resolve())
    if sha256_file(args.validation_manifest) != VALIDATION_MANIFEST_SHA256:
        raise ValueError("validation manifest SHA mismatch")
    if sha256_file(args.cache / "index.json") != VALIDATION_CACHE_INDEX_SHA256:
        raise ValueError("validation cache index SHA mismatch")
    if sha256_file(args.archived_seed32_report) != ARCHIVED_SEED32_REPORT_SHA256:
        raise ValueError("archived seed32 validation report SHA mismatch")
    runner_sha = sha256_file(ROOT / "scripts/train_t1ce_wt_gt_best_teacher_routing.py")
    reports = {
        "anchor_repeat3_control": load_report(args.control_report, "anchor_repeat3_control", runner_sha),
        "gt_best_of3_candidate": load_report(args.candidate_report, "gt_best_of3_candidate", runner_sha),
    }
    if reports["anchor_repeat3_control"]["initial_state_sha256"] != reports["gt_best_of3_candidate"]["initial_state_sha256"]:
        raise ValueError("paired initial states differ")
    if reports["anchor_repeat3_control"]["sampling_weight_sha256"] != reports["gt_best_of3_candidate"]["sampling_weight_sha256"]:
        raise ValueError("paired sampling weights differ")
    if [row["sampler_index_sha256"] for row in reports["anchor_repeat3_control"]["history"]] != [row["sampler_index_sha256"] for row in reports["gt_best_of3_candidate"]["history"]]:
        raise ValueError("paired sampler indices differ")
    if reports["anchor_repeat3_control"]["history"][0]["first_batch_sha256"] != reports["gt_best_of3_candidate"]["history"][0]["first_batch_sha256"]:
        raise ValueError("paired first augmented batch differs")

    checkpoint_paths = {
        "anchor_repeat3_control": args.control_checkpoint,
        "gt_best_of3_candidate": args.candidate_checkpoint,
    }
    report_paths = {
        "anchor_repeat3_control": args.control_report,
        "gt_best_of3_candidate": args.candidate_report,
    }
    preaccess = json.loads(args.preaccess_receipt.read_text())
    if preaccess.get("schema") != PREACCESS_SCHEMA or preaccess.get("status") != "frozen_before_fixed_validation108":
        raise ValueError("invalid routing validation preaccess receipt")
    if preaccess.get("evaluator_sha256") != evaluator_sha or preaccess.get("runner_sha256") != runner_sha:
        raise ValueError("preaccess code hash mismatch")
    if preaccess.get("report_sha256") != {arm: sha256_file(path) for arm, path in report_paths.items()}:
        raise ValueError("preaccess report hash mismatch")
    if preaccess.get("checkpoint_sha256") != {arm: sha256_file(path) for arm, path in checkpoint_paths.items()}:
        raise ValueError("preaccess checkpoint hash mismatch")
    if preaccess.get("implementation_receipt_sha256") != sha256_file(args.implementation_receipt):
        raise ValueError("preaccess implementation receipt mismatch")
    if preaccess.get("access") != {"fixed_validation108": False, "outer": False, "confirmation": False, "test270": False}:
        raise ValueError("preaccess role seal mismatch")

    archived_report = json.loads(args.archived_seed32_report.read_text())
    archived = archived_report["summaries"]["full972_candidate"]
    write_json_atomic(started, {
        "schema": SCHEMA + ".started",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preaccess_receipt_sha256": sha256_file(args.preaccess_receipt),
        "evaluator_sha256": evaluator_sha,
        "threshold": 0.0,
        "tta": False,
        "sam": False,
        "postprocessing": False,
        "fixed_validation108_access": True,
        "outer_access": False,
        "confirmation_access": False,
        "test270_access": False,
    })

    case_ids = read_case_ids(args.validation_manifest)
    data = BrainCropDataset(args.cache, expected_split="validation", case_ids=case_ids)
    if len(data) != 108 or len({subject_id.rsplit("-", 1)[0] for subject_id in data.case_ids}) != 50:
        raise ValueError("validation cohort count mismatch")
    for filename, field in (("images.f32", "images_sha256"), ("labels.u8", "labels_sha256")):
        if sha256_file(args.cache / filename) != data.index[field]:
            raise ValueError("validation cache tensor SHA mismatch")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("routing validation requires CUDA")
    torch.cuda.set_device(device)
    loader = DataLoader(data, batch_size=8, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    models = {arm: load_model(checkpoint_paths[arm], arm, reports[arm], device) for arm in ARMS}
    summaries = {arm: validate(model, loader, device) for arm, model in models.items()}
    candidate = summaries["gt_best_of3_candidate"]
    control = summaries["anchor_repeat3_control"]
    versus_control = compare(control, candidate, seed=20261041, samples=10_000)
    versus_archived = compare(archived, candidate, seed=20261041, samples=10_000)
    delta_control = versus_control["ema_minus_raw"]
    delta_archived = versus_archived["ema_minus_raw"]
    gates = {
        "candidate_beats_control_scan": float(delta_control["scan"]) > 0.0,
        "candidate_beats_control_subject": float(delta_control["subject"]) > 0.0,
        "candidate_beats_archived_scan": float(delta_archived["scan"]) > 0.0,
        "candidate_beats_archived_subject": float(delta_archived["subject"]) > 0.0,
        "candidate_point_at_least_103": float(candidate["automatic_point_hit_rate"]) >= 103.0 / 108.0,
        "control_scan_ci_safe": float(delta_control["scan_95ci"][0]) >= -0.003,
        "control_subject_ci_safe": float(delta_control["subject_95ci"][0]) >= -0.003,
        "archived_scan_ci_safe": float(delta_archived["scan_95ci"][0]) >= -0.003,
        "archived_subject_ci_safe": float(delta_archived["subject_95ci"][0]) >= -0.003,
    }
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summaries": summaries,
        "archived_full972_seed32": archived,
        "candidate_minus_control": versus_control,
        "candidate_minus_archived": versus_archived,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "candidate_authorized_for_test": all(gates.values()),
        "routing_total_counts": reports["gt_best_of3_candidate"]["routing"]["total_counts"],
        "inputs": {
            "report_sha256": {arm: sha256_file(path) for arm, path in report_paths.items()},
            "checkpoint_sha256": {arm: sha256_file(path) for arm, path in checkpoint_paths.items()},
            "preaccess_receipt_sha256": sha256_file(args.preaccess_receipt),
            "evaluator_sha256": evaluator_sha,
        },
        "access": {"fixed_validation108": True, "outer": False, "confirmation": False, "test270": False},
    }
    write_json_atomic(args.output, report)
    print(json.dumps({
        "metrics": {name: {"scan": value["locator_dsc_mean"], "subject": value["locator_subject_dsc_mean"], "point": value["automatic_point_hit_rate"]} for name, value in {**summaries, "archived_full972_seed32": archived}.items()},
        "candidate_minus_control": delta_control,
        "candidate_minus_archived": delta_archived,
        "gates": gates,
        "candidate_authorized_for_test": report["candidate_authorized_for_test"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
