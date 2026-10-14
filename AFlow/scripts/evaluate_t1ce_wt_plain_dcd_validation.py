#!/usr/bin/env python3
"""Evaluate preregistered DCD checkpoints on fixed validation108."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
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


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_dcd_validation"
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_dcd_checkpoint"
IMPLEMENTATION_RECEIPT_SCHEMA = (
    "aflow.sammed3d.t1ce_wt_dcd_implementation_receipt"
)
TRAINING_REPORT_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_dcd_continuation"
KD_VALIDATION_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_voxel_kd_validation"
TRAINER_ENTRYPOINT = "scripts/train_t1ce_wt_plain_dcd_continuation.py"

EPOCHS = (176, 184, 192, 200)
VALIDATION_CASES = 108
VALIDATION_SUBJECTS = 50
VALIDATION_MANIFEST_SHA256 = (
    "f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb"
)
VALIDATION_CACHE_INDEX_SHA256 = (
    "9339e4575561447dd5bd99bc29e909ae1e9f566721d351a244d087a40622f322"
)
KD_VALIDATION_SHA256 = (
    "db7c3bc6b5ff9e1076827d25c9f4de0dea88781bfc5de45da0fb2f79a09bfb57"
)
KD_EPOCH = 200
KD_SCAN = 0.7343843078447713
KD_SUBJECT = 0.7244341433615912
KD_POINT = 0.9351851851851852
SAMPLER_WEIGHTS_SHA256 = (
    "757e463d8e60f96ef6429613f6783fb3aa1014b580932959f9ed354d37072124"
)
ACCESS_FLAGS = (
    "validation_access",
    "outer_selector_access",
    "confirmation_access",
    "test_access",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _require_false_access(value: dict[str, Any], *, source: str) -> None:
    bad = {key: value.get(key) for key in ACCESS_FLAGS if value.get(key) is not False}
    if bad:
        raise ValueError(f"{source} does not declare all access flags false: {bad}")


def _require_exact_zero(value: Any, *, field: str, source: str) -> None:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} lacks numeric {field}") from exc
    if not math.isfinite(number) or number != 0.0:
        raise ValueError(f"{source} {field} must be exactly zero, got {number}")


def _require_finite_numbers(value: Any, *, source: str) -> None:
    """Reject NaN/Inf anywhere in the recorded DCD loss diagnostics."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{source} contains a non-finite diagnostic")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_numbers(item, source=f"{source}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_numbers(item, source=f"{source}[{index}]")
        return
    raise ValueError(f"{source} contains unsupported diagnostic type {type(value)!r}")


def _expected_dcd_weight(global_epoch: int) -> float:
    local_index = global_epoch - 169
    if local_index < 5:
        return 0.0
    progress = min((local_index - 5) / 26.0, 1.0)
    return 0.01 * 0.5 * (1.0 - math.cos(math.pi * progress))


def _validate_summary(summary: dict[str, Any], *, source: str) -> None:
    if int(summary.get("cases", -1)) != VALIDATION_CASES:
        raise ValueError(f"{source} does not contain 108 cases")
    if int(summary.get("subjects", -1)) != VALIDATION_SUBJECTS:
        raise ValueError(f"{source} does not contain 50 subjects")
    per_case = summary.get("per_case")
    if not isinstance(per_case, list) or len(per_case) != VALIDATION_CASES:
        raise ValueError(f"{source} per_case is incomplete")
    case_ids = [str(item.get("case_id")) for item in per_case]
    if len(set(case_ids)) != VALIDATION_CASES:
        raise ValueError(f"{source} contains duplicate case IDs")
    subject_ids = {str(item.get("subject_id")) for item in per_case}
    if len(subject_ids) != VALIDATION_SUBJECTS:
        raise ValueError(f"{source} per_case subject count differs")
    for field in (
        "locator_dsc_mean",
        "locator_subject_dsc_mean",
        "automatic_point_hit_rate",
    ):
        number = float(summary[field])
        if not math.isfinite(number):
            raise ValueError(f"{source} contains non-finite {field}")


def _validate_kd_report(path: Path) -> tuple[dict[str, Any], str]:
    digest = sha256_file(path)
    if digest != KD_VALIDATION_SHA256:
        raise ValueError("KD_VALIDATION SHA-256 mismatch")
    report = _read_json(path)
    if report.get("schema") != KD_VALIDATION_SCHEMA or report.get("schema_version") != 1:
        raise ValueError("unexpected KD validation report schema")
    if report.get("outer_selector_access") is not False:
        raise ValueError("KD validation report declares outer-selector access")
    if report.get("confirmation_access") is not False:
        raise ValueError("KD validation report declares confirmation access")
    if report.get("test_access") is not False:
        raise ValueError("KD validation report declares test access")
    if report.get("selected_kd_epoch") != KD_EPOCH:
        raise ValueError("KD validation report no longer selects epoch200")
    summaries = report.get("summaries")
    if not isinstance(summaries, dict):
        raise ValueError("KD validation report lacks summaries")
    kd = summaries.get("kd")
    if not isinstance(kd, dict) or set(kd) != {str(epoch) for epoch in EPOCHS}:
        raise ValueError("KD validation report lacks the four frozen base-KD epochs")
    for epoch in EPOCHS:
        _validate_summary(kd[str(epoch)], source=f"KD report epoch{epoch}")
    kd200 = kd[str(KD_EPOCH)]
    for field, expected in (
        ("locator_dsc_mean", KD_SCAN),
        ("locator_subject_dsc_mean", KD_SUBJECT),
        ("automatic_point_hit_rate", KD_POINT),
    ):
        if float(kd200[field]) != expected:
            raise ValueError(f"KD epoch200 {field} differs from preregistration")
    return report, digest


def _validate_implementation_receipt(path: Path) -> tuple[dict[str, Any], str]:
    receipt = _read_json(path)
    if receipt.get("schema") != IMPLEMENTATION_RECEIPT_SCHEMA:
        raise ValueError("unexpected DCD implementation receipt schema")
    _require_false_access(receipt, source="DCD implementation receipt")
    for field in (
        "synthetic_all_gates_passed",
        "gpu_smoke_all_gates_passed",
        "all_gates_passed",
    ):
        if receipt.get(field) is not True:
            raise ValueError(f"DCD implementation receipt has {field} != true")
    return receipt, sha256_file(path)


def _validate_training_report(path: Path) -> tuple[dict[str, Any], str]:
    report = _read_json(path)
    if report.get("schema") != TRAINING_REPORT_SCHEMA:
        raise ValueError("unexpected DCD training report schema")
    if report.get("status") != "complete":
        raise ValueError("DCD training report is not complete")
    _require_false_access(report, source="DCD training report")
    _require_exact_zero(
        report.get("reload_max_abs_difference"),
        field="reload_max_abs_difference",
        source="DCD training report",
    )
    _require_exact_zero(
        report.get("stripped_deploy_max_abs_difference"),
        field="stripped_deploy_max_abs_difference",
        source="DCD training report",
    )
    return report, sha256_file(path)


def _load_model(
    path: Path,
    expected_epoch: int,
    device: torch.device,
) -> tuple[AutomaticPromptLocator, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint is not a dictionary: {path}")
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"unexpected DCD checkpoint schema: {path}")
    if checkpoint.get("arm") != "dcd":
        raise ValueError(f"unexpected DCD checkpoint arm: {path}")
    if checkpoint.get("global_epoch") != expected_epoch:
        raise ValueError(f"unexpected DCD checkpoint epoch: {path}")
    if checkpoint.get("continuation_epoch") != expected_epoch - 168:
        raise ValueError(f"unexpected continuation epoch: {path}")
    if checkpoint.get("resolution") != 128:
        raise ValueError(f"DCD checkpoint is not true-128: {path}")
    _require_false_access(checkpoint, source=f"DCD checkpoint epoch{expected_epoch}")
    learning_rate = float(checkpoint.get("learning_rate"))
    if not math.isfinite(learning_rate) or learning_rate < 0.0:
        raise ValueError(f"DCD checkpoint has invalid learning_rate: {path}")
    dcd_weight = float(checkpoint.get("dcd_weight"))
    expected_weight = _expected_dcd_weight(expected_epoch)
    if not math.isfinite(dcd_weight) or not math.isclose(
        dcd_weight, expected_weight, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"DCD checkpoint weight mismatch at epoch{expected_epoch}: "
            f"expected {expected_weight}, got {dcd_weight}"
        )
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError(f"DCD checkpoint lacks deploy model_state_dict: {path}")
    for field in (
        "architecture",
        "base_voxel_kd",
        "detail_teacher",
        "dcd",
        "loss_diagnostics",
        "data_role",
        "sampling_contract",
    ):
        if not isinstance(checkpoint.get(field), dict):
            raise ValueError(f"DCD checkpoint lacks {field}: {path}")
    _require_finite_numbers(
        checkpoint["loss_diagnostics"],
        source=f"DCD epoch{expected_epoch}.loss_diagnostics",
    )
    _require_exact_zero(
        checkpoint.get("stripped_deploy_max_abs_difference"),
        field="stripped_deploy_max_abs_difference",
        source=f"DCD checkpoint epoch{expected_epoch}",
    )
    _require_exact_zero(
        checkpoint.get("checkpoint_reload_max_abs_difference"),
        field="checkpoint_reload_max_abs_difference",
        source=f"DCD checkpoint epoch{expected_epoch}",
    )
    sampling = checkpoint["sampling_contract"]
    if any(
        (
            sampling.get("weights_sha256") != SAMPLER_WEIGHTS_SHA256,
            sampling.get("generator_seed") != 20260846,
            sampling.get("num_samples") != 756,
            sampling.get("replacement") is not True,
        )
    ):
        raise ValueError(f"DCD checkpoint sampling contract mismatch: {path}")
    data_role = checkpoint["data_role"]
    if not isinstance(data_role.get("train"), dict):
        raise ValueError(f"DCD checkpoint lacks frozen train role: {path}")
    if data_role.get("cases") != 756 or data_role.get("subjects") != 346:
        raise ValueError(f"DCD checkpoint train cohort mismatch: {path}")
    architecture = checkpoint["architecture"]
    expected_architecture = {
        "class": "AutomaticPromptLocator",
        "backbone": "MONAI SegResNet",
        "center_aware": False,
        "init_filters": 16,
        "dropout": 0.1,
        "input_channels": ["T1ce"],
        "outputs": ["WT segmentation logit"],
    }
    for key, expected in expected_architecture.items():
        if architecture.get(key) != expected:
            raise ValueError(f"DCD checkpoint architecture mismatch for {key}: {path}")
    model = AutomaticPromptLocator(
        center_aware=False,
        init_filters=int(architecture["init_filters"]),
        dropout=float(architecture["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint


def _validate_cache(cache: Path, manifest: Path) -> BrainCropDataset:
    if sha256_file(manifest) != VALIDATION_MANIFEST_SHA256:
        raise ValueError("validation manifest SHA-256 mismatch")
    index_path = cache / "index.json"
    if sha256_file(index_path) != VALIDATION_CACHE_INDEX_SHA256:
        raise ValueError("validation cache index SHA-256 mismatch")
    case_ids = read_case_ids(manifest)
    if case_ids is None:
        raise ValueError("validation manifest did not produce case IDs")
    data = BrainCropDataset(cache, expected_split="validation", case_ids=case_ids)
    if len(data) != VALIDATION_CASES:
        raise ValueError("unexpected validation case count")
    subjects = {case.rsplit("-", 1)[0] for case in data.case_ids}
    if len(subjects) != VALIDATION_SUBJECTS:
        raise ValueError("unexpected validation subject count")
    if data.index.get("manifest_sha256") != VALIDATION_MANIFEST_SHA256:
        raise ValueError("validation cache manifest binding mismatch")
    for filename, field in (
        ("images.f32", "images_sha256"),
        ("labels.u8", "labels_sha256"),
    ):
        if sha256_file(cache / filename) != data.index.get(field):
            raise ValueError(f"validation cache {filename} SHA-256 mismatch")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--dcd-dir", type=Path, required=True)
    parser.add_argument("--kd-validation", type=Path, required=True)
    parser.add_argument("--implementation-receipt", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    started_path = args.output.with_suffix(".started.json")
    if args.output.exists() or started_path.exists():
        raise FileExistsError("refusing repeated DCD validation")
    kd_report, kd_report_sha = _validate_kd_report(args.kd_validation)
    implementation, implementation_sha = _validate_implementation_receipt(
        args.implementation_receipt
    )
    training_report, training_report_sha = _validate_training_report(
        args.training_report
    )
    data = _validate_cache(args.cache, args.validation_manifest)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("DCD validation requires CUDA")
    torch.cuda.set_device(device)

    checkpoint_paths = {
        epoch: args.dcd_dir / f"plain_128_dcd_global_epoch{epoch}.pth"
        for epoch in EPOCHS
    }
    # Verify every input before recording the first new validation access.
    preloaded: dict[int, tuple[AutomaticPromptLocator, dict[str, Any]]] = {}
    checkpoint_sha: dict[int, str] = {}
    for epoch, path in checkpoint_paths.items():
        checkpoint_sha[epoch] = sha256_file(path)
        preloaded[epoch] = _load_model(path, epoch, device)

    write_json_atomic(
        started_path,
        {
            "schema": SCHEMA + ".started",
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_scope": "fixed validation108; development selection only",
            "selection": (
                "maximum DCD scan Dice among epochs 176/184/192/200; "
                "subject Dice exact-tie-break"
            ),
            "validation_manifest_sha256": VALIDATION_MANIFEST_SHA256,
            "validation_cache_index_sha256": VALIDATION_CACHE_INDEX_SHA256,
            "kd_validation_sha256": kd_report_sha,
            "implementation_receipt_sha256": implementation_sha,
            "training_report_sha256": training_report_sha,
            "checkpoint_sha256": {
                str(epoch): checkpoint_sha[epoch] for epoch in EPOCHS
            },
            "threshold": 0.0,
            "tta": False,
            "sam": False,
            "postprocessing": False,
            "validation_access": True,
            "outer_selector_access": False,
            "confirmation_access": False,
            "test_access": False,
        },
    )

    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    summaries: dict[str, Any] = {}
    checkpoints: dict[str, Any] = {}
    for epoch in EPOCHS:
        key = str(epoch)
        model, checkpoint = preloaded.pop(epoch)
        summaries[key] = validate(model, loader, device)
        _validate_summary(summaries[key], source=f"DCD epoch{epoch}")
        checkpoints[key] = {
            "path": str(checkpoint_paths[epoch].resolve()),
            "sha256": checkpoint_sha[epoch],
            "global_epoch": epoch,
            "continuation_epoch": checkpoint["continuation_epoch"],
            "learning_rate": checkpoint["learning_rate"],
            "dcd_weight": checkpoint["dcd_weight"],
            "architecture": checkpoint["architecture"],
            "base_voxel_kd": checkpoint["base_voxel_kd"],
            "detail_teacher": checkpoint["detail_teacher"],
            "dcd": checkpoint["dcd"],
            "loss_diagnostics": checkpoint["loss_diagnostics"],
            "data_role": checkpoint["data_role"],
            "sampling_contract": checkpoint["sampling_contract"],
            "stripped_deploy_max_abs_difference": checkpoint[
                "stripped_deploy_max_abs_difference"
            ],
        }
        del model
        torch.cuda.empty_cache()

    selected_epoch = max(
        EPOCHS,
        key=lambda epoch: (
            summaries[str(epoch)]["locator_dsc_mean"],
            summaries[str(epoch)]["locator_subject_dsc_mean"],
        ),
    )
    selected_key = str(selected_epoch)
    selected = summaries[selected_key]
    base_kd = kd_report["summaries"]["kd"]
    kd200 = base_kd[str(KD_EPOCH)]
    versus_base_kd = {
        str(epoch): compare(
            base_kd[str(epoch)],
            summaries[str(epoch)],
            seed=20260890 + index,
        )
        for index, epoch in enumerate(EPOCHS)
    }
    versus_kd200 = {
        str(epoch): compare(
            kd200,
            summaries[str(epoch)],
            seed=20260894 + index,
        )
        for index, epoch in enumerate(EPOCHS)
    }
    selected_base_delta = versus_base_kd[selected_key]["ema_minus_raw"]
    selected_kd200_delta = versus_kd200[selected_key]["ema_minus_raw"]
    selected_base = base_kd[selected_key]

    prerequisite_gates = {
        "implementation_synthetic_gates_passed": implementation[
            "synthetic_all_gates_passed"
        ]
        is True,
        "implementation_gpu_smoke_gates_passed": implementation[
            "gpu_smoke_all_gates_passed"
        ]
        is True,
        "implementation_all_gates_passed": implementation["all_gates_passed"] is True,
        "training_complete": training_report["status"] == "complete",
        "formal_checkpoint_reload_exact": float(
            training_report["reload_max_abs_difference"]
        )
        == 0.0,
        "stripped_deploy_logits_exact": (
            float(training_report["stripped_deploy_max_abs_difference"]) == 0.0
            and all(
                float(checkpoints[str(epoch)]["stripped_deploy_max_abs_difference"])
                == 0.0
                for epoch in EPOCHS
            )
        ),
    }
    promotion_gates = {
        "selected_beats_same_epoch_base_voxel_kd_scan": float(
            selected["locator_dsc_mean"]
        )
        > float(selected_base["locator_dsc_mean"]),
        "selected_beats_same_epoch_base_voxel_kd_subject": float(
            selected["locator_subject_dsc_mean"]
        )
        > float(selected_base["locator_subject_dsc_mean"]),
        "selected_beats_voxel_kd_epoch200_scan": float(
            selected["locator_dsc_mean"]
        )
        > float(kd200["locator_dsc_mean"]),
        "selected_beats_voxel_kd_epoch200_subject": float(
            selected["locator_subject_dsc_mean"]
        )
        > float(kd200["locator_subject_dsc_mean"]),
        "selected_point_within_one_case_of_voxel_kd_epoch200": float(
            selected["automatic_point_hit_rate"]
        )
        >= float(kd200["automatic_point_hit_rate"]) - 1.0 / VALIDATION_CASES,
        "paired_scan_ci_lower_at_least_minus_0.003_vs_voxel_kd_epoch200": float(
            selected_kd200_delta["scan_95ci"][0]
        )
        >= -0.003,
        "paired_subject_ci_lower_at_least_minus_0.003_vs_voxel_kd_epoch200": float(
            selected_kd200_delta["subject_95ci"][0]
        )
        >= -0.003,
        "formal_checkpoint_reload_exact": prerequisite_gates[
            "formal_checkpoint_reload_exact"
        ],
        "stripped_deploy_logits_exact": prerequisite_gates[
            "stripped_deploy_logits_exact"
        ],
        "all_implementation_and_gpu_smoke_gates_passed": all(
            prerequisite_gates[key]
            for key in (
                "implementation_synthetic_gates_passed",
                "implementation_gpu_smoke_gates_passed",
                "implementation_all_gates_passed",
            )
        ),
    }
    all_gates_passed = all(prerequisite_gates.values()) and all(
        promotion_gates.values()
    )
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "fixed validation108; development selection only",
        "not_outer_selector": True,
        "not_external_confirmation": True,
        "protocol": {
            "trainer_entrypoint": TRAINER_ENTRYPOINT,
            "resolution": 128,
            "threshold": "raw logit > 0",
            "tta": False,
            "sam": False,
            "postprocessing": False,
            "locator_forwards": 1,
            "epochs": list(EPOCHS),
            "selection": "maximum DCD scan Dice; subject Dice exact-tie-break",
            "checkpoint_selection_uses_test": False,
        },
        "data": {
            "manifest": str(args.validation_manifest.resolve()),
            "manifest_sha256": VALIDATION_MANIFEST_SHA256,
            "cache": str(args.cache.resolve()),
            "cache_index_sha256": VALIDATION_CACHE_INDEX_SHA256,
            "cases": VALIDATION_CASES,
            "subjects": VALIDATION_SUBJECTS,
        },
        "evidence_inputs": {
            "kd_validation": str(args.kd_validation.resolve()),
            "kd_validation_sha256": kd_report_sha,
            "implementation_receipt": str(args.implementation_receipt.resolve()),
            "implementation_receipt_sha256": implementation_sha,
            "training_report": str(args.training_report.resolve()),
            "training_report_sha256": training_report_sha,
        },
        "checkpoints": checkpoints,
        "summaries": summaries,
        "base_voxel_kd_summaries": base_kd,
        "voxel_kd_epoch200_summary": kd200,
        "dcd_minus_same_epoch_base_voxel_kd": versus_base_kd,
        "dcd_minus_voxel_kd_epoch200": versus_kd200,
        "selection_rule": (
            "maximum DCD scan Dice among epochs 176/184/192/200; "
            "subject Dice exact-tie-break"
        ),
        "selected_epoch": selected_epoch,
        "selected_checkpoint": checkpoints[selected_key],
        "selected_summary": selected,
        "selected_minus_same_epoch_base_voxel_kd": selected_base_delta,
        "selected_minus_voxel_kd_epoch200": selected_kd200_delta,
        "prerequisite_gates": prerequisite_gates,
        "promotion_gates": promotion_gates,
        "all_gates_passed": all_gates_passed,
        "dcd_candidate_authorized_for_test": all_gates_passed,
        "validation_access": True,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "selected_epoch": selected_epoch,
                "selected": {
                    "scan": selected["locator_dsc_mean"],
                    "subject": selected["locator_subject_dsc_mean"],
                    "point_hit": selected["automatic_point_hit_rate"],
                },
                "selected_minus_same_epoch_base_voxel_kd": selected_base_delta,
                "selected_minus_voxel_kd_epoch200": selected_kd200_delta,
                "prerequisite_gates": prerequisite_gates,
                "promotion_gates": promotion_gates,
                "dcd_candidate_authorized_for_test": all_gates_passed,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
