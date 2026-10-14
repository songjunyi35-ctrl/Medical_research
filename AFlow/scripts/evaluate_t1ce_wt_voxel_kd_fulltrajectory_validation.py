#!/usr/bin/env python3
"""Evaluate frozen full756/full972 random-init voxel-KD checkpoints on validation108."""
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

from scripts.evaluate_t1ce_wt_plain_ema_validation import compare  # noqa: E402
from scripts.train_t1ce_wt_capg3d import BrainCropDataset, read_case_ids, sha256_file, validate, write_json_atomic  # noqa: E402
from scripts.train_t1ce_wt_plain_voxel_kd_continuation import _model  # noqa: E402
import scripts.train_t1ce_wt_voxel_kd_fulltrajectory_final as train  # noqa: E402

SCHEMA = "aflow.sammed3d.t1ce_wt_voxel_kd_fulltrajectory_validation"
PREACCESS_SCHEMA = SCHEMA + ".preaccess_receipt"
IMPLEMENTATION_SCHEMA = "aflow.brats.t1ce_wt_voxel_kd_fulltrajectory_implementation_receipt"
VALIDATION_MANIFEST_SHA256 = "f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb"
VALIDATION_CACHE_INDEX_SHA256 = "9339e4575561447dd5bd99bc29e909ae1e9f566721d351a244d087a40622f322"
KD_VALIDATION_SHA256 = "db7c3bc6b5ff9e1076827d25c9f4de0dea88781bfc5de45da0fb2f79a09bfb57"
ARMS = ("full756_control", "full972_candidate")
CHECKPOINT_NAMES = {arm: f"plain_fulltrajectory_{arm}_epoch168.pth" for arm in ARMS}


def require_access_false(value: dict[str, Any]) -> None:
    for key in ("validation_access", "outer_selector_access", "confirmation_access", "test_access"):
        if value.get(key) is not False:
            raise ValueError(f"artifact must declare {key}=false")


def load_training_report(path: Path, arm: str, runner_sha: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema") != train.SCHEMA or value.get("status") != "complete":
        raise ValueError("unexpected final report schema/status")
    if value.get("arm") != arm or value.get("runner_sha256") != runner_sha:
        raise ValueError("final report arm/runner mismatch")
    if value.get("all_gates_passed") is not True or len(value.get("history", [])) != 168:
        raise ValueError("final training gates/epoch count failed")
    if value.get("fixed_epoch") != 168 or value.get("authorized_final_epoch") != 168:
        raise ValueError("final epoch contract mismatch")
    sampling = value.get("sampling_contract", {})
    if (sampling.get("requested_draws"), sampling.get("consumed_draws"), sampling.get("optimizer_steps")) != (972, 968, 121):
        raise ValueError("final compute contract mismatch")
    require_access_false(value)
    return value


def load_checkpoint(path: Path, arm: str, report: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    if path.resolve() != Path(report["checkpoint"]["path"]).resolve() or sha256_file(path) != report["checkpoint"]["sha256"]:
        raise ValueError("checkpoint path/SHA mismatch")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("schema") != train.CHECKPOINT_SCHEMA or value.get("arm") != arm or value.get("epoch") != 168:
        raise ValueError("checkpoint schema/arm/epoch mismatch")
    if value.get("model_state_sha256") != train.state_dict_sha256(value["model_state_dict"]):
        raise ValueError("checkpoint model-state hash mismatch")
    if value.get("initial_state_sha256") != report.get("initial_state_sha256"):
        raise ValueError("checkpoint initial-state mismatch")
    if value.get("method_documents") != report.get("method_documents"):
        raise ValueError("checkpoint method-document mismatch")
    require_access_false(value)
    return _model(value, device).eval(), value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--control-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--kd-validation", type=Path, required=True)
    parser.add_argument("--implementation-receipt", type=Path, required=True)
    parser.add_argument("--preaccess-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    started = args.output.with_suffix(".started.json")
    if args.output.exists() or started.exists():
        raise FileExistsError("refusing repeated fulltrajectory validation")
    if sha256_file(args.validation_manifest) != VALIDATION_MANIFEST_SHA256 or sha256_file(args.cache / "index.json") != VALIDATION_CACHE_INDEX_SHA256:
        raise ValueError("validation manifest/cache SHA mismatch")
    if sha256_file(args.kd_validation) != KD_VALIDATION_SHA256:
        raise ValueError("KD200 validation SHA mismatch")
    receipt = json.loads(args.implementation_receipt.read_text())
    if receipt.get("schema") != IMPLEMENTATION_SCHEMA or receipt.get("status") != "passed_prevalidation" or receipt.get("all_gates_passed") is not True:
        raise ValueError("implementation receipt not validation-ready")
    require_access_false(receipt)
    evaluator_sha = sha256_file(Path(__file__).resolve())
    if receipt["validation_evaluator_sha256"] != evaluator_sha:
        raise ValueError("evaluator SHA mismatch")
    runner_sha = receipt["final_implementation"]["trainer_sha256"]
    if runner_sha != sha256_file(ROOT / "scripts/train_t1ce_wt_voxel_kd_fulltrajectory_final.py"):
        raise ValueError("final runner changed")
    preaccess = json.loads(args.preaccess_receipt.read_text())
    if preaccess.get("schema") != PREACCESS_SCHEMA or preaccess.get("status") != "frozen_before_validation":
        raise ValueError("preaccess receipt invalid")
    if preaccess.get("implementation_receipt_sha256") != sha256_file(args.implementation_receipt) or preaccess.get("evaluator_sha256") != evaluator_sha:
        raise ValueError("preaccess receipt hash mismatch")
    require_access_false(preaccess)
    reports = {
        "full756_control": load_training_report(args.control_report, "full756_control", runner_sha),
        "full972_candidate": load_training_report(args.candidate_report, "full972_candidate", runner_sha),
    }
    if reports[ARMS[0]]["initial_state_sha256"] != reports[ARMS[1]]["initial_state_sha256"]:
        raise ValueError("final initial states differ")
    if preaccess["training_report_sha256"] != {"full756_control": sha256_file(args.control_report), "full972_candidate": sha256_file(args.candidate_report)}:
        raise ValueError("training report SHA receipt mismatch")
    if preaccess["checkpoint_sha256"] != {arm: reports[arm]["checkpoint"]["sha256"] for arm in ARMS}:
        raise ValueError("checkpoint SHA receipt mismatch")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("validation requires CUDA")
    torch.cuda.set_device(device)
    case_ids = read_case_ids(args.validation_manifest)
    data = BrainCropDataset(args.cache, expected_split="validation", case_ids=case_ids)
    if len(data) != 108 or len({case.rsplit("-", 1)[0] for case in data.case_ids}) != 50:
        raise ValueError("validation cohort mismatch")
    write_json_atomic(started, {"schema": SCHEMA + ".started", "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "preaccess_receipt_sha256": sha256_file(args.preaccess_receipt), "validation_access": True, "outer_selector_access": False, "confirmation_access": False, "test_access": False})
    loader = DataLoader(data, batch_size=8, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    summaries, checkpoints = {}, {}
    for arm, directory in (("full756_control", args.control_dir), ("full972_candidate", args.candidate_dir)):
        path = directory / CHECKPOINT_NAMES[arm]
        model, checkpoint = load_checkpoint(path, arm, reports[arm], device)
        summaries[arm] = validate(model, loader, device)
        checkpoints[arm] = {"path": str(path.resolve()), "sha256": sha256_file(path), "model_state_sha256": checkpoint["model_state_sha256"]}
        del model
        torch.cuda.empty_cache()
    kd = json.loads(args.kd_validation.read_text())["summaries"]["kd"]["200"]
    vs_control = compare(summaries["full756_control"], summaries["full972_candidate"], seed=20261500)
    vs_kd = compare(kd, summaries["full972_candidate"], seed=20261501)
    candidate, control = summaries["full972_candidate"], summaries["full756_control"]
    dc, dk = vs_control["ema_minus_raw"], vs_kd["ema_minus_raw"]
    hits = int(round(candidate["automatic_point_hit_rate"] * 108))
    gates = {
        "beats_full756_scan": candidate["locator_dsc_mean"] > control["locator_dsc_mean"],
        "beats_full756_subject": candidate["locator_subject_dsc_mean"] > control["locator_subject_dsc_mean"],
        "beats_kd200_scan": candidate["locator_dsc_mean"] > kd["locator_dsc_mean"],
        "beats_kd200_subject": candidate["locator_subject_dsc_mean"] > kd["locator_subject_dsc_mean"],
        "point_hits_at_least_101": hits >= 101,
        "scan_ci_vs_control": dc["scan_95ci"][0] >= -0.003,
        "subject_ci_vs_control": dc["subject_95ci"][0] >= -0.003,
        "scan_ci_vs_kd200": dk["scan_95ci"][0] >= -0.003,
        "subject_ci_vs_kd200": dk["subject_95ci"][0] >= -0.003,
        "engineering_gates": all(report["all_gates_passed"] for report in reports.values()),
    }
    report = {"schema": SCHEMA, "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "summaries": summaries, "checkpoints": checkpoints, "archived_kd200": kd, "candidate_minus_control": vs_control, "candidate_minus_kd200": vs_kd, "candidate_point_hits": hits, "gates": gates, "all_gates_passed": all(gates.values()), "candidate_authorized_for_test": all(gates.values()), "preaccess_receipt_sha256": sha256_file(args.preaccess_receipt), "validation_access": True, "outer_selector_access": False, "confirmation_access": False, "test_access": False}
    write_json_atomic(args.output, report)
    print(json.dumps({"candidate": {"scan": candidate["locator_dsc_mean"], "subject": candidate["locator_subject_dsc_mean"], "point_hits": hits}, "control": {"scan": control["locator_dsc_mean"], "subject": control["locator_subject_dsc_mean"], "point": control["automatic_point_hit_rate"]}, "kd200": {"scan": kd["locator_dsc_mean"], "subject": kd["locator_subject_dsc_mean"], "point": kd["automatic_point_hit_rate"]}, "gates": gates, "candidate_authorized_for_test": report["candidate_authorized_for_test"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
