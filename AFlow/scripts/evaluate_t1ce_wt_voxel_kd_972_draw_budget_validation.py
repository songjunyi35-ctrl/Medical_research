#!/usr/bin/env python3
"""Fixed validation108 decision for the 972-case voxel-KD draw-budget pair."""
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
import scripts.train_t1ce_wt_voxel_kd_972_draw_budget as train  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_voxel_kd_972_draw_budget_validation"
PREACCESS_SCHEMA = SCHEMA + ".preaccess_receipt"
IMPLEMENTATION_SCHEMA = "aflow.brats.t1ce_wt_voxel_kd_972_draw_budget_implementation_receipt"
VALIDATION_MANIFEST_SHA256 = "f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb"
VALIDATION_CACHE_INDEX_SHA256 = "9339e4575561447dd5bd99bc29e909ae1e9f566721d351a244d087a40622f322"
KD_VALIDATION_SHA256 = "db7c3bc6b5ff9e1076827d25c9f4de0dea88781bfc5de45da0fb2f79a09bfb57"
SAVE_EPOCHS = (176, 184, 192, 200)
FROZEN_TRAIN_METHOD_DOCUMENTS = {
    "amendment_sha256": "bce7f00eed898a7d2825b03536f3cf99335b4390315359cd189ec368a57462a8",
    "feasibility_sha256": "32be09856d71e78f512792c8bcf8f5f9ee137e6ba1e4358862ff2a61ba828c31",
    "idea_sha256": "b61ca77285a259bab0b45a25bcd46bb9e546f18d4a043e1649e2f644d28023a9",
    "native_artifact_sha256": "32be09856d71e78f512792c8bcf8f5f9ee137e6ba1e4358862ff2a61ba828c31",
    "prereg_sha256": "ce1afcb81ee8c56303898c4122673dfaefa24210aa993a5725e363b9cbe173a8",
}
ARM_BUDGET = {
    "budget752_control": (756, 752, 94, 2700),
    "budget968_candidate": (972, 968, 121, 3600),
}


def require_access_false(value: dict[str, Any]) -> None:
    for key in ("validation_access", "outer_selector_access", "confirmation_access", "test_access"):
        if value.get(key) is not False:
            raise ValueError(f"artifact must declare {key}=false")


def load_training_report(path: Path, arm: str, runner_sha: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != train.SCHEMA or value.get("status") != "complete":
        raise ValueError("unexpected or incomplete current training report")
    if value.get("arm") != arm or value.get("runner_sha256") != runner_sha:
        raise ValueError("current training arm/runner mismatch")
    if value.get("all_gates_passed") is not True:
        raise ValueError("formal gates failed")
    if len(value.get("history", [])) != 32:
        raise ValueError("formal epoch count mismatch")
    if set(value.get("checkpoints", {})) != {str(epoch) for epoch in SAVE_EPOCHS}:
        raise ValueError("formal checkpoint grid mismatch")
    if any(float(item) != 0.0 for item in value.get("reload_max_abs_difference_by_epoch", {}).values()):
        raise ValueError("formal checkpoint reload mismatch")
    draws, consumed, steps, max_seconds = ARM_BUDGET[arm]
    if float(value.get("elapsed_seconds", float("inf"))) > max_seconds:
        raise ValueError("formal elapsed gate failed")
    if int(value.get("peak_gpu_memory_bytes", 1 << 62)) > 24 * 1024**3:
        raise ValueError("formal memory gate failed")
    sampling = value.get("sampling_contract", {})
    expected = {
        "population_count": 972,
        "sampler_num_samples": draws,
        "consumed_draws_per_epoch": consumed,
        "optimizer_steps_per_epoch": steps,
        "native_volume_artifact_sha256": train.NATIVE_ARTIFACT_SHA256,
        "native_weight_sha256": train.NATIVE_WEIGHT_SHA256,
    }
    for key, expected_value in expected.items():
        if sampling.get(key) != expected_value:
            raise ValueError(f"formal sampling mismatch: {key}")
    require_access_false(value)
    return value


def load_checkpoint(
    path: Path,
    arm: str,
    epoch: int,
    report: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    item = report["checkpoints"][str(epoch)]
    if path.resolve() != Path(item["path"]).resolve() or sha256_file(path) != item["sha256"]:
        raise ValueError("checkpoint path or file SHA mismatch")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("schema") != train.CHECKPOINT_SCHEMA:
        raise ValueError("unexpected current checkpoint schema")
    if value.get("arm") != arm or value.get("global_epoch") != epoch:
        raise ValueError("checkpoint arm/epoch mismatch")
    if value.get("method_documents") != FROZEN_TRAIN_METHOD_DOCUMENTS:
        raise ValueError("checkpoint method-document mismatch")
    sampling = value.get("sampling_contract", {})
    draws, consumed, steps, _ = ARM_BUDGET[arm]
    if (
        sampling.get("sampler_num_samples") != draws
        or sampling.get("consumed_draws_per_epoch") != consumed
        or sampling.get("optimizer_steps_per_epoch") != steps
    ):
        raise ValueError("checkpoint arm budget mismatch")
    if sampling.get("native_weight_sha256") != train.NATIVE_WEIGHT_SHA256:
        raise ValueError("checkpoint native weight mismatch")
    if value.get("architecture", {}).get("parameter_count") != 4_700_897:
        raise ValueError("checkpoint parameter count mismatch")
    if value.get("model_state_sha256") != train.state_dict_sha256(value["model_state_dict"]):
        raise ValueError("checkpoint model-state SHA mismatch")
    require_access_false(value)
    return _model(value, device).eval(), value


def checkpoint_sha_map(report: dict[str, Any]) -> dict[str, str]:
    return {
        str(epoch): str(report["checkpoints"][str(epoch)]["sha256"])
        for epoch in SAVE_EPOCHS
    }


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
    parser.add_argument("--preaccess-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    started = args.output.with_suffix(".started.json")
    if args.output.exists() or started.exists():
        raise FileExistsError("refusing repeated draw-budget validation")
    if sha256_file(args.validation_manifest) != VALIDATION_MANIFEST_SHA256:
        raise ValueError("validation manifest SHA mismatch")
    if sha256_file(args.cache / "index.json") != VALIDATION_CACHE_INDEX_SHA256:
        raise ValueError("validation cache index SHA mismatch")
    if sha256_file(args.kd_validation) != KD_VALIDATION_SHA256:
        raise ValueError("archived KD validation SHA mismatch")

    receipt = json.loads(args.implementation_receipt.read_text(encoding="utf-8"))
    if receipt.get("schema") != IMPLEMENTATION_SCHEMA or receipt.get("status") != "passed_prevalidation":
        raise ValueError("implementation receipt not passed for validation")
    if receipt.get("all_gates_passed") is not True:
        raise ValueError("implementation receipt gates failed")
    require_access_false(receipt)
    implementation = receipt.get("implementation", {})
    evaluator_sha = sha256_file(Path(__file__).resolve())
    if implementation.get("validation_evaluator_sha256") != evaluator_sha:
        raise ValueError("validation evaluator SHA mismatch")
    runner_sha = implementation.get("trainer_sha256")
    if runner_sha != sha256_file(ROOT / implementation["trainer_path"]):
        raise ValueError("trainer SHA changed after receipt")

    preaccess = json.loads(args.preaccess_receipt.read_text(encoding="utf-8"))
    if preaccess.get("schema") != PREACCESS_SCHEMA or preaccess.get("status") != "frozen_before_validation":
        raise ValueError("unexpected validation preaccess receipt")
    if preaccess.get("implementation_receipt_sha256") != sha256_file(args.implementation_receipt):
        raise ValueError("preaccess implementation receipt SHA mismatch")
    if preaccess.get("evaluator_sha256") != evaluator_sha:
        raise ValueError("preaccess evaluator SHA mismatch")
    require_access_false(preaccess)

    control_report = load_training_report(
        args.control_training_report, "budget752_control", runner_sha
    )
    candidate_report = load_training_report(
        args.candidate_training_report, "budget968_candidate", runner_sha
    )
    if sha256_file(args.control_training_report) != preaccess["control_training_report_sha256"]:
        raise ValueError("control training report changed after preaccess freeze")
    if sha256_file(args.candidate_training_report) != preaccess["candidate_training_report_sha256"]:
        raise ValueError("candidate training report changed after preaccess freeze")
    if control_report["initial_state_sha256"] != candidate_report["initial_state_sha256"]:
        raise ValueError("paired arms initial states differ")
    expected_checkpoint_shas = {
        "budget752_control": checkpoint_sha_map(control_report),
        "budget968_candidate": checkpoint_sha_map(candidate_report),
    }
    if preaccess.get("checkpoint_sha256") != expected_checkpoint_shas:
        raise ValueError("preaccess checkpoint SHA grid mismatch")

    kd_report = json.loads(args.kd_validation.read_text(encoding="utf-8"))
    kd200 = kd_report["summaries"]["kd"]["200"]
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("fixed validation requires CUDA")
    torch.cuda.set_device(device)
    case_ids = read_case_ids(args.validation_manifest)
    if case_ids is None:
        raise ValueError("validation manifest lacks case IDs")
    data = BrainCropDataset(args.cache, expected_split="validation", case_ids=case_ids)
    if len(data) != 108 or len({case.rsplit("-", 1)[0] for case in data.case_ids}) != 50:
        raise ValueError("unexpected fixed validation cohort")

    write_json_atomic(started, {
        "schema": SCHEMA + ".started",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "candidate maximum Scan, Subject exact tie, earlier epoch exact tie",
        "epochs": list(SAVE_EPOCHS),
        "preaccess_receipt_sha256": sha256_file(args.preaccess_receipt),
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
    summaries = {arm: {} for arm in train.ARMS}
    checkpoints = {arm: {} for arm in train.ARMS}
    for arm, directory, report in (
        ("budget752_control", args.control_dir, control_report),
        ("budget968_candidate", args.candidate_dir, candidate_report),
    ):
        for epoch in SAVE_EPOCHS:
            path = directory / f"plain_128_{arm}_global_epoch{epoch}.pth"
            model, checkpoint = load_checkpoint(path, arm, epoch, report, device)
            summaries[arm][str(epoch)] = validate(model, loader, device)
            checkpoints[arm][str(epoch)] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "model_state_sha256": checkpoint["model_state_sha256"],
            }
            del model
            torch.cuda.empty_cache()

    candidate_arm = "budget968_candidate"
    control_arm = "budget752_control"
    selected_epoch = max(
        SAVE_EPOCHS,
        key=lambda epoch: (
            summaries[candidate_arm][str(epoch)]["locator_dsc_mean"],
            summaries[candidate_arm][str(epoch)]["locator_subject_dsc_mean"],
            -epoch,
        ),
    )
    vs_control, vs_kd = {}, {}
    for index, epoch in enumerate(SAVE_EPOCHS):
        key = str(epoch)
        vs_control[key] = compare(
            summaries[control_arm][key], summaries[candidate_arm][key],
            seed=20261300 + index,
        )
        vs_kd[key] = compare(
            kd200, summaries[candidate_arm][key], seed=20261304 + index,
        )
    key = str(selected_epoch)
    candidate = summaries[candidate_arm][key]
    control = summaries[control_arm][key]
    delta_control = vs_control[key]["ema_minus_raw"]
    delta_kd = vs_kd[key]["ema_minus_raw"]
    hits = int(round(float(candidate["automatic_point_hit_rate"]) * 108))
    gates = {
        "beats_same_epoch_control_scan": candidate["locator_dsc_mean"] > control["locator_dsc_mean"],
        "beats_same_epoch_control_subject": candidate["locator_subject_dsc_mean"] > control["locator_subject_dsc_mean"],
        "beats_kd200_scan": candidate["locator_dsc_mean"] > kd200["locator_dsc_mean"],
        "beats_kd200_subject": candidate["locator_subject_dsc_mean"] > kd200["locator_subject_dsc_mean"],
        "point_hits_at_least_101_of_108": hits >= 101,
        "scan_ci_lower_at_least_minus_0.003_vs_control": delta_control["scan_95ci"][0] >= -0.003,
        "subject_ci_lower_at_least_minus_0.003_vs_control": delta_control["subject_95ci"][0] >= -0.003,
        "scan_ci_lower_at_least_minus_0.003_vs_kd200": delta_kd["scan_95ci"][0] >= -0.003,
        "subject_ci_lower_at_least_minus_0.003_vs_kd200": delta_kd["subject_95ci"][0] >= -0.003,
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
        "preaccess_receipt_sha256": sha256_file(args.preaccess_receipt),
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
            "point_hits": hits,
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
