#!/usr/bin/env python3
"""One-shot fixed-validation108 comparison of frozen Full972 width16/width24 models."""
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
from scripts.train_t1ce_wt_plain_voxel_kd_continuation import _model  # noqa: E402
import scripts.train_t1ce_wt_full972_width_final as train  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_full972_width_validation"
PREACCESS_SCHEMA = SCHEMA + ".preaccess_receipt"
IMPLEMENTATION_SCHEMA = "aflow.brats.t1ce_wt_full972_width24_implementation_receipt"
VALIDATION_MANIFEST_SHA256 = "f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb"
VALIDATION_CACHE_INDEX_SHA256 = "9339e4575561447dd5bd99bc29e909ae1e9f566721d351a244d087a40622f322"
ARCHIVED_SEED32_VALIDATION_SHA256 = "30cd65c78ad1ed2a2b5748652f48ec05cde3b54468b44a66eba77d062d3ce129"
ARMS = ("full972_width16", "full972_width24")
CHECKPOINT_NAMES = {arm: f"plain_fulltrajectory_{arm}_epoch166.pth" for arm in ARMS}


def require_access_false(
    value: dict[str, Any], *, require_fixed_validation_field: bool = True
) -> None:
    keys = ["validation_access", "outer_selector_access", "confirmation_access", "test_access"]
    if require_fixed_validation_field:
        keys.append("fixed_validation108_access")
    for key in keys:
        if value.get(key) is not False:
            raise ValueError(f"artifact must declare {key}=false")


def load_training_report(path: Path, arm: str, runner_sha: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema") != train.SCHEMA or value.get("status") != "complete":
        raise ValueError("unexpected final report schema/status")
    if value.get("arm") != arm or value.get("runner_sha256") != runner_sha:
        raise ValueError("final report arm/runner mismatch")
    if value.get("all_gates_passed") is not True or len(value.get("history", [])) != 166:
        raise ValueError("final training gates/epoch count failed")
    if value.get("fixed_epoch") != 166 or value.get("authorized_final_epoch") != 166:
        raise ValueError("final epoch contract mismatch")
    if value.get("schedule_total_epochs") != 200:
        raise ValueError("schedule contract mismatch")
    if value.get("parameter_count") != train.PARAMETERS[arm]:
        raise ValueError("parameter-count contract mismatch")
    sampling = value.get("sampling_contract", {})
    if (
        sampling.get("requested_draws"),
        sampling.get("consumed_draws"),
        sampling.get("optimizer_steps"),
    ) != (972, 968, 121):
        raise ValueError("final compute contract mismatch")
    require_access_false(value)
    return value


def load_checkpoint(
    path: Path,
    arm: str,
    report: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if path.resolve() != Path(report["checkpoint"]["path"]).resolve():
        raise ValueError("checkpoint path mismatch")
    if sha256_file(path) != report["checkpoint"]["sha256"]:
        raise ValueError("checkpoint SHA mismatch")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if (
        value.get("schema") != train.CHECKPOINT_SCHEMA
        or value.get("arm") != arm
        or value.get("epoch") != 166
    ):
        raise ValueError("checkpoint schema/arm/epoch mismatch")
    if value.get("model_state_sha256") != train.state_dict_sha256(value["model_state_dict"]):
        raise ValueError("checkpoint model-state hash mismatch")
    architecture = value.get("architecture", {})
    if int(architecture.get("init_filters", -1)) != train.WIDTH[arm]:
        raise ValueError("checkpoint width mismatch")
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
    parser.add_argument("--width16-dir", type=Path, required=True)
    parser.add_argument("--width24-dir", type=Path, required=True)
    parser.add_argument("--width16-report", type=Path, required=True)
    parser.add_argument("--width24-report", type=Path, required=True)
    parser.add_argument("--archived-seed32-validation", type=Path, required=True)
    parser.add_argument("--implementation-receipt", type=Path, required=True)
    parser.add_argument("--preaccess-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    started = args.output.with_suffix(".started.json")
    if args.output.exists() or started.exists():
        raise FileExistsError("refusing repeated width validation")
    evaluator_sha = sha256_file(Path(__file__).resolve())
    if sha256_file(args.validation_manifest) != VALIDATION_MANIFEST_SHA256:
        raise ValueError("validation manifest SHA mismatch")
    if sha256_file(args.cache / "index.json") != VALIDATION_CACHE_INDEX_SHA256:
        raise ValueError("validation cache SHA mismatch")
    if sha256_file(args.archived_seed32_validation) != ARCHIVED_SEED32_VALIDATION_SHA256:
        raise ValueError("archived seed32 validation SHA mismatch")

    implementation = json.loads(args.implementation_receipt.read_text())
    if implementation.get("schema") != IMPLEMENTATION_SCHEMA:
        raise ValueError("implementation receipt schema mismatch")
    if implementation.get("final_smoke", {}).get("all_gates_passed") is not True:
        raise ValueError("final implementation smoke did not pass")
    # This receipt predates the split access fields; validation_access=false
    # and test_access=false are its complete applicable legacy seal.
    if implementation.get("validation_access") is not False or implementation.get("test_access") is not False:
        raise ValueError("legacy implementation access seal failed")
    runner_sha = sha256_file(ROOT / "scripts/train_t1ce_wt_full972_width_final.py")
    if implementation.get("final_runner_sha256") != runner_sha:
        raise ValueError("final runner changed after implementation receipt")

    preaccess = json.loads(args.preaccess_receipt.read_text())
    if (
        preaccess.get("schema") != PREACCESS_SCHEMA
        or preaccess.get("status") != "frozen_before_validation"
    ):
        raise ValueError("preaccess receipt invalid")
    if preaccess.get("evaluator_sha256") != evaluator_sha:
        raise ValueError("evaluator changed after preaccess freeze")
    if preaccess.get("implementation_receipt_sha256") != sha256_file(args.implementation_receipt):
        raise ValueError("implementation receipt hash mismatch")
    require_access_false(preaccess)

    report_paths = {
        "full972_width16": args.width16_report,
        "full972_width24": args.width24_report,
    }
    reports = {
        arm: load_training_report(path, arm, runner_sha)
        for arm, path in report_paths.items()
    }
    if preaccess.get("training_report_sha256") != {
        arm: sha256_file(path) for arm, path in report_paths.items()
    }:
        raise ValueError("training report receipt mismatch")
    if preaccess.get("checkpoint_sha256") != {
        arm: reports[arm]["checkpoint"]["sha256"] for arm in ARMS
    }:
        raise ValueError("checkpoint receipt mismatch")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("validation requires CUDA")
    torch.cuda.set_device(device)
    case_ids = read_case_ids(args.validation_manifest)
    data = BrainCropDataset(args.cache, expected_split="validation", case_ids=case_ids)
    if len(data) != 108 or len({case.rsplit("-", 1)[0] for case in data.case_ids}) != 50:
        raise ValueError("validation cohort mismatch")
    write_json_atomic(
        started,
        {
            "schema": SCHEMA + ".started",
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "preaccess_receipt_sha256": sha256_file(args.preaccess_receipt),
            "validation_access": True,
            "fixed_validation108_access": True,
            "outer_selector_access": False,
            "confirmation_access": False,
            "test_access": False,
        },
    )
    loader = DataLoader(
        data,
        batch_size=8,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    directories = {
        "full972_width16": args.width16_dir,
        "full972_width24": args.width24_dir,
    }
    summaries: dict[str, Any] = {}
    checkpoints: dict[str, Any] = {}
    for arm in ARMS:
        path = directories[arm] / CHECKPOINT_NAMES[arm]
        model, checkpoint = load_checkpoint(path, arm, reports[arm], device)
        summaries[arm] = validate(model, loader, device)
        checkpoints[arm] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "model_state_sha256": checkpoint["model_state_sha256"],
            "init_filters": train.WIDTH[arm],
            "parameter_count": train.PARAMETERS[arm],
        }
        del model
        torch.cuda.empty_cache()

    archived_report = json.loads(args.archived_seed32_validation.read_text())
    archived = archived_report["summaries"]["full972_candidate"]
    matched = compare(summaries["full972_width16"], summaries["full972_width24"], seed=20261801)
    versus_archived = compare(archived, summaries["full972_width24"], seed=20261802)
    candidate = summaries["full972_width24"]
    control = summaries["full972_width16"]
    dm = matched["ema_minus_raw"]
    da = versus_archived["ema_minus_raw"]
    candidate_hits = int(round(candidate["automatic_point_hit_rate"] * 108))
    control_hits = int(round(control["automatic_point_hit_rate"] * 108))
    archived_hits = int(round(archived["automatic_point_hit_rate"] * 108))
    gates = {
        "beats_matched_width16_scan": candidate["locator_dsc_mean"] > control["locator_dsc_mean"],
        "beats_matched_width16_subject": candidate["locator_subject_dsc_mean"] > control["locator_subject_dsc_mean"],
        "beats_archived_seed32_scan": candidate["locator_dsc_mean"] > archived["locator_dsc_mean"],
        "beats_archived_seed32_subject": candidate["locator_subject_dsc_mean"] > archived["locator_subject_dsc_mean"],
        "point_hits_at_least_101": candidate_hits >= 101,
        "scan_ci_vs_matched": dm["scan_95ci"][0] >= -0.003,
        "subject_ci_vs_matched": dm["subject_95ci"][0] >= -0.003,
        "scan_ci_vs_archived_seed32": da["scan_95ci"][0] >= -0.003,
        "subject_ci_vs_archived_seed32": da["subject_95ci"][0] >= -0.003,
        "engineering_gates": all(report["all_gates_passed"] for report in reports.values()),
    }
    result = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summaries": summaries,
        "archived_seed32": archived,
        "checkpoints": checkpoints,
        "width24_minus_width16": matched,
        "width24_minus_archived_seed32": versus_archived,
        "point_hits": {
            "full972_width16": control_hits,
            "full972_width24": candidate_hits,
            "archived_seed32": archived_hits,
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "candidate_authorized_for_test": all(gates.values()),
        "preaccess_receipt_sha256": sha256_file(args.preaccess_receipt),
        "validation_access": True,
        "fixed_validation108_access": True,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    write_json_atomic(args.output, result)
    print(
        json.dumps(
            {
                "width16": {
                    "scan": control["locator_dsc_mean"],
                    "subject": control["locator_subject_dsc_mean"],
                    "point_hits": control_hits,
                },
                "width24": {
                    "scan": candidate["locator_dsc_mean"],
                    "subject": candidate["locator_subject_dsc_mean"],
                    "point_hits": candidate_hits,
                },
                "archived_seed32": {
                    "scan": archived["locator_dsc_mean"],
                    "subject": archived["locator_subject_dsc_mean"],
                    "point_hits": archived_hits,
                },
                "gates": gates,
                "candidate_authorized_for_test": result["candidate_authorized_for_test"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
