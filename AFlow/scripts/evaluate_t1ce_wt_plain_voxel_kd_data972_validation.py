#!/usr/bin/env python3
"""Fixed validation108 decision for cache-matched 756/972 voxel-KD arms."""
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
    BrainCropDataset,
    read_case_ids,
    sha256_file,
    validate,
    write_json_atomic,
)
from scripts.train_t1ce_wt_plain_voxel_kd_data972 import (  # noqa: E402
    AMENDMENT_SHA256,
    ARM_COUNTS,
    CACHE_INDEX_SHA256 as TRAIN_CACHE_INDEX_SHA256,
    CHECKPOINT_SCHEMA,
    FEASIBILITY_SHA256,
    IDEA_SHA256,
    MAX_FORMAL_SECONDS,
    MAX_PEAK_BYTES,
    PREREG_SHA256,
    SAVE_GLOBAL_EPOCHS,
    SCHEMA as TRAIN_SCHEMA,
    _model,
    state_dict_sha256,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_voxel_kd_data972_validation"
VALIDATION_MANIFEST_SHA256 = "f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb"
VALIDATION_CACHE_INDEX_SHA256 = "9339e4575561447dd5bd99bc29e909ae1e9f566721d351a244d087a40622f322"
KD_VALIDATION_SHA256 = "db7c3bc6b5ff9e1076827d25c9f4de0dea88781bfc5de45da0fb2f79a09bfb57"
IMPLEMENTATION_SCHEMA = "aflow.sammed3d.t1ce_wt_voxel_kd_data972_implementation_receipt"


def _require_access_false(value: dict[str, Any]) -> None:
    for key in (
        "validation_access",
        "outer_selector_access",
        "confirmation_access",
        "test_access",
    ):
        if value.get(key) is not False:
            raise ValueError(f"artifact must declare {key}=false")


def load_training_report(path: Path, arm: str, runner_sha256: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != TRAIN_SCHEMA or value.get("status") != "complete":
        raise ValueError(f"unexpected/incomplete training report: {path}")
    if value.get("arm") != arm or value.get("runner_sha256") != runner_sha256:
        raise ValueError(f"training report arm/runner mismatch: {path}")
    if value.get("all_gates_passed") is not True:
        raise ValueError(f"training report gates failed: {path}")
    if float(value.get("elapsed_seconds", float("inf"))) > MAX_FORMAL_SECONDS:
        raise ValueError("formal elapsed-time gate failed")
    if int(value.get("peak_gpu_memory_bytes", MAX_PEAK_BYTES + 1)) > MAX_PEAK_BYTES:
        raise ValueError("formal peak-memory gate failed")
    if set(value.get("checkpoints", {})) != {"176", "184", "192", "200"}:
        raise ValueError("training report checkpoint grid mismatch")
    if any(float(x) != 0.0 for x in value.get("reload_max_abs_difference_by_epoch", {}).values()):
        raise ValueError("training checkpoint reload was not exact")
    expected_count, expected_subjects = ARM_COUNTS[arm]
    role = value.get("data_role", {})
    if role.get("cases") != expected_count or role.get("subjects") != expected_subjects:
        raise ValueError("training report data role mismatch")
    sampling = value.get("sampling_contract", {})
    expected_sampling = {
        "population_count": expected_count,
        "sampler_num_samples": 756,
        "consumed_draws_per_epoch": 752,
        "dropped_draws_per_epoch": 4,
        "optimizer_steps_per_epoch": 94,
        "replacement": True,
    }
    for key, expected in expected_sampling.items():
        if sampling.get(key) != expected:
            raise ValueError(f"training sampling mismatch for {key}")
    _require_access_false(value)
    return value


def load_checkpoint(
    path: Path,
    *,
    arm: str,
    epoch: int,
    report: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    report_item = report["checkpoints"][str(epoch)]
    if sha256_file(path) != report_item["sha256"]:
        raise ValueError(f"checkpoint SHA mismatch: {path}")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unexpected data972 checkpoint schema")
    if value.get("arm") != arm or value.get("global_epoch") != epoch:
        raise ValueError("checkpoint arm/epoch mismatch")
    if value.get("continuation_epoch") != epoch - 168 or value.get("resolution") != 128:
        raise ValueError("checkpoint epoch/resolution mismatch")
    if value.get("model_state_sha256") != state_dict_sha256(value["model_state_dict"]):
        raise ValueError("checkpoint model-state SHA mismatch")
    if value.get("model_state_sha256") != report_item["model_state_sha256"]:
        raise ValueError("checkpoint/report model-state SHA mismatch")
    docs = value.get("method_documents", {})
    expected_docs = {
        "idea_sha256": IDEA_SHA256,
        "prereg_sha256": PREREG_SHA256,
        "amendment_sha256": AMENDMENT_SHA256,
        "feasibility_sha256": FEASIBILITY_SHA256,
    }
    if docs != expected_docs:
        raise ValueError("checkpoint method-document contract mismatch")
    expected_count, expected_subjects = ARM_COUNTS[arm]
    role = value.get("data_role", {})
    if role.get("cases") != expected_count or role.get("subjects") != expected_subjects:
        raise ValueError("checkpoint data-role mismatch")
    if role.get("cache_index_sha256") != TRAIN_CACHE_INDEX_SHA256:
        raise ValueError("checkpoint train-cache index mismatch")
    sampling = value.get("sampling_contract", {})
    if sampling.get("population_count") != expected_count:
        raise ValueError("checkpoint population count mismatch")
    if sampling.get("sampler_num_samples") != 756 or sampling.get("optimizer_steps_per_epoch") != 94:
        raise ValueError("checkpoint fixed-compute contract mismatch")
    architecture = value.get("architecture", {})
    if architecture.get("parameter_count") != 4_700_897:
        raise ValueError("checkpoint plain-model parameter count mismatch")
    _require_access_false(value)
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
    started_path = args.output.with_suffix(".started.json")
    if args.output.exists() or started_path.exists():
        raise FileExistsError("refusing repeated data972 validation")
    if sha256_file(args.validation_manifest) != VALIDATION_MANIFEST_SHA256:
        raise ValueError("validation manifest SHA mismatch")
    if sha256_file(args.cache / "index.json") != VALIDATION_CACHE_INDEX_SHA256:
        raise ValueError("validation cache index SHA mismatch")
    if sha256_file(args.kd_validation) != KD_VALIDATION_SHA256:
        raise ValueError("archived KD validation SHA mismatch")
    receipt = json.loads(args.implementation_receipt.read_text(encoding="utf-8"))
    if receipt.get("schema") != IMPLEMENTATION_SCHEMA or receipt.get("status") != "passed":
        raise ValueError("implementation receipt is missing or not passed")
    if receipt.get("all_gates_passed") is not True:
        raise ValueError("implementation receipt gates failed")
    _require_access_false(receipt)
    implementation = receipt.get("implementation", {})
    evaluator_sha = sha256_file(Path(__file__).resolve())
    if implementation.get("validation_evaluator_sha256") != evaluator_sha:
        raise ValueError("implementation receipt evaluator SHA mismatch")
    runner_sha = str(implementation.get("trainer_sha256"))
    control_report = load_training_report(
        args.control_training_report, "data756_control", runner_sha
    )
    candidate_report = load_training_report(
        args.candidate_training_report, "data972_candidate", runner_sha
    )
    if control_report["initial_state_sha256"] != candidate_report["initial_state_sha256"]:
        raise ValueError("arms do not share an identical initial model state")
    kd_report = json.loads(args.kd_validation.read_text(encoding="utf-8"))
    if kd_report.get("schema") != "aflow.sammed3d.t1ce_wt_plain_voxel_kd_validation":
        raise ValueError("unexpected archived KD validation schema")
    kd_summaries = kd_report["summaries"]["kd"]
    kd200 = kd_summaries["200"]

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("data972 validation requires CUDA")
    torch.cuda.set_device(device)
    case_ids = read_case_ids(args.validation_manifest)
    if case_ids is None:
        raise ValueError("validation manifest has no case IDs")
    data = BrainCropDataset(args.cache, expected_split="validation", case_ids=case_ids)
    if len(data) != 108 or len({case.rsplit("-", 1)[0] for case in data.case_ids}) != 50:
        raise ValueError("unexpected fixed validation cohort")
    # All evidence and checkpoints are preflighted before recording validation access.
    write_json_atomic(started_path, {
        "schema": SCHEMA + ".started",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "candidate maximum Scan, Subject exact tie, earlier epoch exact tie",
        "epochs": list(SAVE_GLOBAL_EPOCHS),
        "implementation_receipt_sha256": sha256_file(args.implementation_receipt),
        "validation_access": True,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    })
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    summaries: dict[str, dict[str, Any]] = {
        "data756_control": {}, "data972_candidate": {}
    }
    checkpoints: dict[str, dict[str, Any]] = {
        "data756_control": {}, "data972_candidate": {}
    }
    for arm, directory, report in (
        ("data756_control", args.control_dir, control_report),
        ("data972_candidate", args.candidate_dir, candidate_report),
    ):
        for epoch in SAVE_GLOBAL_EPOCHS:
            path = directory / f"plain_128_{arm}_global_epoch{epoch}.pth"
            model, checkpoint = load_checkpoint(
                path, arm=arm, epoch=epoch, report=report, device=device
            )
            summaries[arm][str(epoch)] = validate(model, loader, device)
            checkpoints[arm][str(epoch)] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "model_state_sha256": checkpoint["model_state_sha256"],
            }
            del model
            torch.cuda.empty_cache()

    selected_epoch = max(
        SAVE_GLOBAL_EPOCHS,
        key=lambda epoch: (
            summaries["data972_candidate"][str(epoch)]["locator_dsc_mean"],
            summaries["data972_candidate"][str(epoch)]["locator_subject_dsc_mean"],
            -epoch,
        ),
    )
    comparisons_control: dict[str, Any] = {}
    comparisons_kd200: dict[str, Any] = {}
    for index, epoch in enumerate(SAVE_GLOBAL_EPOCHS):
        key = str(epoch)
        comparisons_control[key] = compare(
            summaries["data756_control"][key],
            summaries["data972_candidate"][key],
            seed=20261040 + index,
        )
        comparisons_kd200[key] = compare(
            kd200,
            summaries["data972_candidate"][key],
            seed=20261044 + index,
        )
    selected_key = str(selected_epoch)
    candidate = summaries["data972_candidate"][selected_key]
    control = summaries["data756_control"][selected_key]
    delta_control = comparisons_control[selected_key]["ema_minus_raw"]
    delta_kd = comparisons_kd200[selected_key]["ema_minus_raw"]
    point_hits = int(round(float(candidate["automatic_point_hit_rate"]) * 108))
    gates = {
        "beats_same_epoch_control_scan": float(candidate["locator_dsc_mean"]) > float(control["locator_dsc_mean"]),
        "beats_same_epoch_control_subject": float(candidate["locator_subject_dsc_mean"]) > float(control["locator_subject_dsc_mean"]),
        "beats_kd200_scan": float(candidate["locator_dsc_mean"]) > float(kd200["locator_dsc_mean"]),
        "beats_kd200_subject": float(candidate["locator_subject_dsc_mean"]) > float(kd200["locator_subject_dsc_mean"]),
        "point_hits_at_least_101_of_108": point_hits >= 101,
        "scan_ci_lower_at_least_minus_0.003_vs_control": float(delta_control["scan_95ci"][0]) >= -0.003,
        "subject_ci_lower_at_least_minus_0.003_vs_control": float(delta_control["subject_95ci"][0]) >= -0.003,
        "scan_ci_lower_at_least_minus_0.003_vs_kd200": float(delta_kd["scan_95ci"][0]) >= -0.003,
        "subject_ci_lower_at_least_minus_0.003_vs_kd200": float(delta_kd["subject_95ci"][0]) >= -0.003,
        "implementation_all_gates_passed": receipt["all_gates_passed"] is True,
        "control_formal_all_gates_passed": control_report["all_gates_passed"] is True,
        "candidate_formal_all_gates_passed": candidate_report["all_gates_passed"] is True,
    }
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summaries": summaries,
        "checkpoints": checkpoints,
        "archived_kd200": kd200,
        "candidate_minus_same_epoch_control": comparisons_control,
        "candidate_minus_kd200": comparisons_kd200,
        "selected_candidate_epoch": selected_epoch,
        "selected_candidate_summary": candidate,
        "selected_control_summary": control,
        "selected_candidate_point_hits": point_hits,
        "selection": "candidate maximum Scan, Subject exact tie, earlier epoch exact tie",
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "candidate_authorized_for_test": all(gates.values()),
        "implementation_receipt_sha256": sha256_file(args.implementation_receipt),
        "validation_access": True,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    write_json_atomic(args.output, report)
    print(json.dumps({
        "selected_candidate_epoch": selected_epoch,
        "selected_candidate": {
            "scan": candidate["locator_dsc_mean"],
            "subject": candidate["locator_subject_dsc_mean"],
            "point_hits": point_hits,
        },
        "same_epoch_control": {
            "scan": control["locator_dsc_mean"],
            "subject": control["locator_subject_dsc_mean"],
            "point": control["automatic_point_hit_rate"],
        },
        "kd200": {
            "scan": kd200["locator_dsc_mean"],
            "subject": kd200["locator_subject_dsc_mean"],
            "point": kd200["automatic_point_hit_rate"],
        },
        "gates": gates,
        "candidate_authorized_for_test": report["candidate_authorized_for_test"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
