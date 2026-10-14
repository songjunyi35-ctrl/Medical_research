#!/usr/bin/env python3
"""Fixed validation108 decision for native-volume-weighted 756/972 voxel-KD."""
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
from scripts.train_t1ce_wt_capg3d import (  # noqa: E402
    BrainCropDataset, read_case_ids, sha256_file, validate, write_json_atomic,
)
from scripts.train_t1ce_wt_plain_voxel_kd_continuation import _model  # noqa: E402
import scripts.train_t1ce_wt_plain_voxel_kd_data972_native_sampling as train  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_voxel_kd_data972_native_validation"
VALIDATION_MANIFEST_SHA256 = "f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb"
VALIDATION_CACHE_INDEX_SHA256 = "9339e4575561447dd5bd99bc29e909ae1e9f566721d351a244d087a40622f322"
KD_VALIDATION_SHA256 = "db7c3bc6b5ff9e1076827d25c9f4de0dea88781bfc5de45da0fb2f79a09bfb57"
IMPLEMENTATION_SCHEMA = "aflow.sammed3d.t1ce_wt_voxel_kd_data972_native_implementation_receipt"
SAVE_EPOCHS = (176, 184, 192, 200)
EXPECTED_WEIGHT_SHA = {
    "native756_control": train.CONTROL_WEIGHT_SHA256,
    "native972_candidate": train.CANDIDATE_WEIGHT_SHA256,
}


def require_access_false(value: dict[str, Any]) -> None:
    for key in ("validation_access", "outer_selector_access", "confirmation_access", "test_access"):
        if value.get(key) is not False:
            raise ValueError(f"artifact must declare {key}=false")


def load_training_report(path: Path, arm: str, runner_sha: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != train.SCHEMA or value.get("status") != "complete":
        raise ValueError("unexpected/incomplete native training report")
    if value.get("arm") != arm or value.get("runner_sha256") != runner_sha:
        raise ValueError("native training arm/runner mismatch")
    if value.get("all_gates_passed") is not True:
        raise ValueError("native formal gates failed")
    if float(value.get("elapsed_seconds", float("inf"))) > train.base.MAX_FORMAL_SECONDS:
        raise ValueError("native formal elapsed gate failed")
    if int(value.get("peak_gpu_memory_bytes", train.base.MAX_PEAK_BYTES + 1)) > train.base.MAX_PEAK_BYTES:
        raise ValueError("native formal memory gate failed")
    if set(value.get("checkpoints", {})) != {str(e) for e in SAVE_EPOCHS}:
        raise ValueError("native formal checkpoint grid mismatch")
    if any(float(x) != 0 for x in value.get("reload_max_abs_difference_by_epoch", {}).values()):
        raise ValueError("native checkpoint reload not exact")
    count, subjects = train.ARM_COUNTS[arm]
    if value.get("data_role", {}).get("cases") != count or value.get("data_role", {}).get("subjects") != subjects:
        raise ValueError("native formal data role mismatch")
    sampling = value.get("sampling_contract", {})
    expected = {
        "population_count": count,
        "sampler_num_samples": 756,
        "consumed_draws_per_epoch": 752,
        "optimizer_steps_per_epoch": 94,
        "native_volume_artifact_sha256": train.NATIVE_ARTIFACT_SHA256,
        "native_weight_sha256": EXPECTED_WEIGHT_SHA[arm],
    }
    for key, expected_value in expected.items():
        if sampling.get(key) != expected_value:
            raise ValueError(f"native formal sampling mismatch: {key}")
    require_access_false(value)
    return value


def load_checkpoint(
    path: Path, arm: str, epoch: int, report: dict[str, Any], device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    item = report["checkpoints"][str(epoch)]
    if sha256_file(path) != item["sha256"]:
        raise ValueError("native checkpoint file SHA mismatch")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("schema") != train.CHECKPOINT_SCHEMA:
        raise ValueError("unexpected native checkpoint schema")
    if value.get("arm") != arm or value.get("global_epoch") != epoch:
        raise ValueError("native checkpoint arm/epoch mismatch")
    if value.get("method_documents") != train.method_documents():
        raise ValueError("native checkpoint method-document mismatch")
    native_sampling = value.get("native_sampling", {})
    if native_sampling.get("native_volume_artifact_sha256") != train.NATIVE_ARTIFACT_SHA256:
        raise ValueError("native checkpoint artifact mismatch")
    if native_sampling.get("native_weight_sha256") != EXPECTED_WEIGHT_SHA[arm]:
        raise ValueError("native checkpoint weight mismatch")
    if value.get("architecture", {}).get("parameter_count") != 4_700_897:
        raise ValueError("native checkpoint parameter count mismatch")
    if value.get("model_state_sha256") != train.base.state_dict_sha256(value["model_state_dict"]):
        raise ValueError("native checkpoint model-state SHA mismatch")
    require_access_false(value)
    return _model(value, device).eval(), value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--control-training-report", type=Path, required=True)
    parser.add_argument("--candidate-training-report", type=Path, required=True)
    parser.add_argument("--kd-validation", type=Path, required=True)
    parser.add_argument("--implementation-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    started = args.output.with_suffix(".started.json")
    if args.output.exists() or started.exists():
        raise FileExistsError("refusing repeated native validation")
    if sha256_file(args.validation_manifest) != VALIDATION_MANIFEST_SHA256:
        raise ValueError("validation manifest SHA mismatch")
    if sha256_file(args.cache / "index.json") != VALIDATION_CACHE_INDEX_SHA256:
        raise ValueError("validation cache index SHA mismatch")
    if sha256_file(args.kd_validation) != KD_VALIDATION_SHA256:
        raise ValueError("archived KD validation SHA mismatch")
    receipt = json.loads(args.implementation_receipt.read_text(encoding="utf-8"))
    if receipt.get("schema") != IMPLEMENTATION_SCHEMA or receipt.get("status") != "passed":
        raise ValueError("native implementation receipt not passed")
    if receipt.get("all_gates_passed") is not True:
        raise ValueError("native implementation gates failed")
    require_access_false(receipt)
    implementation = receipt.get("implementation", {})
    if implementation.get("validation_evaluator_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("native evaluator SHA mismatch")
    runner_sha = implementation.get("trainer_sha256")
    control_report = load_training_report(
        args.control_training_report, "native756_control", runner_sha
    )
    candidate_report = load_training_report(
        args.candidate_training_report, "native972_candidate", runner_sha
    )
    if control_report["initial_state_sha256"] != candidate_report["initial_state_sha256"]:
        raise ValueError("native arms initial states differ")
    kd_report = json.loads(args.kd_validation.read_text(encoding="utf-8"))
    kd200 = kd_report["summaries"]["kd"]["200"]

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("native validation requires CUDA")
    torch.cuda.set_device(device)
    case_ids = read_case_ids(args.validation_manifest)
    if case_ids is None:
        raise ValueError("validation manifest lacks IDs")
    data = BrainCropDataset(args.cache, expected_split="validation", case_ids=case_ids)
    if len(data) != 108 or len({case.rsplit("-", 1)[0] for case in data.case_ids}) != 50:
        raise ValueError("unexpected fixed validation cohort")
    write_json_atomic(started, {
        "schema": SCHEMA + ".started",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "candidate maximum Scan, Subject exact tie, earlier epoch exact tie",
        "epochs": list(SAVE_EPOCHS),
        "implementation_receipt_sha256": sha256_file(args.implementation_receipt),
        "validation_access": True,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    })
    loader = DataLoader(
        data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    summaries = {"native756_control": {}, "native972_candidate": {}}
    checkpoints = {"native756_control": {}, "native972_candidate": {}}
    for arm, directory, report in (
        ("native756_control", args.control_dir, control_report),
        ("native972_candidate", args.candidate_dir, candidate_report),
    ):
        for epoch in SAVE_EPOCHS:
            path = directory / f"plain_128_{arm}_global_epoch{epoch}.pth"
            model, checkpoint = load_checkpoint(path, arm, epoch, report, device)
            summaries[arm][str(epoch)] = validate(model, loader, device)
            checkpoints[arm][str(epoch)] = {
                "path": str(path.resolve()), "sha256": sha256_file(path),
                "model_state_sha256": checkpoint["model_state_sha256"],
            }
            del model
            torch.cuda.empty_cache()
    selected_epoch = max(
        SAVE_EPOCHS,
        key=lambda epoch: (
            summaries["native972_candidate"][str(epoch)]["locator_dsc_mean"],
            summaries["native972_candidate"][str(epoch)]["locator_subject_dsc_mean"],
            -epoch,
        ),
    )
    vs_control, vs_kd = {}, {}
    for index, epoch in enumerate(SAVE_EPOCHS):
        key = str(epoch)
        vs_control[key] = compare(
            summaries["native756_control"][key], summaries["native972_candidate"][key],
            seed=20261060 + index,
        )
        vs_kd[key] = compare(kd200, summaries["native972_candidate"][key], seed=20261064 + index)
    key = str(selected_epoch)
    candidate = summaries["native972_candidate"][key]
    control = summaries["native756_control"][key]
    dc = vs_control[key]["ema_minus_raw"]
    dk = vs_kd[key]["ema_minus_raw"]
    hits = int(round(float(candidate["automatic_point_hit_rate"]) * 108))
    gates = {
        "beats_same_epoch_control_scan": candidate["locator_dsc_mean"] > control["locator_dsc_mean"],
        "beats_same_epoch_control_subject": candidate["locator_subject_dsc_mean"] > control["locator_subject_dsc_mean"],
        "beats_kd200_scan": candidate["locator_dsc_mean"] > kd200["locator_dsc_mean"],
        "beats_kd200_subject": candidate["locator_subject_dsc_mean"] > kd200["locator_subject_dsc_mean"],
        "point_hits_at_least_101_of_108": hits >= 101,
        "scan_ci_lower_at_least_minus_0.003_vs_control": dc["scan_95ci"][0] >= -0.003,
        "subject_ci_lower_at_least_minus_0.003_vs_control": dc["subject_95ci"][0] >= -0.003,
        "scan_ci_lower_at_least_minus_0.003_vs_kd200": dk["scan_95ci"][0] >= -0.003,
        "subject_ci_lower_at_least_minus_0.003_vs_kd200": dk["subject_95ci"][0] >= -0.003,
        "implementation_all_gates_passed": receipt["all_gates_passed"] is True,
        "control_formal_all_gates_passed": control_report["all_gates_passed"] is True,
        "candidate_formal_all_gates_passed": candidate_report["all_gates_passed"] is True,
    }
    report = {
        "schema": SCHEMA, "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summaries": summaries, "checkpoints": checkpoints,
        "archived_kd200": kd200,
        "candidate_minus_same_epoch_control": vs_control,
        "candidate_minus_kd200": vs_kd,
        "selected_candidate_epoch": selected_epoch,
        "selected_candidate_summary": candidate,
        "selected_control_summary": control,
        "selected_candidate_point_hits": hits,
        "selection": "candidate maximum Scan, Subject exact tie, earlier epoch exact tie",
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "candidate_authorized_for_test": all(gates.values()),
        "implementation_receipt_sha256": sha256_file(args.implementation_receipt),
        "validation_access": True, "outer_selector_access": False,
        "confirmation_access": False, "test_access": False,
    }
    write_json_atomic(args.output, report)
    print(json.dumps({
        "selected_candidate_epoch": selected_epoch,
        "selected_candidate": {"scan": candidate["locator_dsc_mean"], "subject": candidate["locator_subject_dsc_mean"], "point_hits": hits},
        "same_epoch_control": {"scan": control["locator_dsc_mean"], "subject": control["locator_subject_dsc_mean"], "point": control["automatic_point_hit_rate"]},
        "kd200": {"scan": kd200["locator_dsc_mean"], "subject": kd200["locator_subject_dsc_mean"], "point": kd200["automatic_point_hit_rate"]},
        "gates": gates, "candidate_authorized_for_test": report["candidate_authorized_for_test"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
