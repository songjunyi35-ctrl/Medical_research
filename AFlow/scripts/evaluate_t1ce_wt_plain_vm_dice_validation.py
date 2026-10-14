#!/usr/bin/env python3
"""Evaluate frozen paper-Dice gamma0 versus VM-Dice gamma1 on validation108."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator  # noqa: E402
from scripts.train_t1ce_wt_capg3d import (  # noqa: E402
    BrainCropDataset,
    read_case_ids,
    sha256_file,
    validate,
    write_json_atomic,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_vm_dice_validation"
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_vm_dice_checkpoint"
TRAINING_REPORT_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_vm_dice_continuation"
IMPLEMENTATION_RECEIPT_SCHEMA = (
    "aflow.sammed3d.t1ce_wt_vm_dice_implementation_receipt"
)
PAIRING_RECEIPT_SCHEMA = "aflow.sammed3d.t1ce_wt_vm_dice_pairing_receipt"
KD_VALIDATION_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_voxel_kd_validation"
TRAINER_ENTRYPOINT = "scripts/train_t1ce_wt_plain_vm_dice_continuation.py"

ARMS = ("paper_dice_gamma0", "vm_dice_gamma1")
EXPECTED_GAMMAS = {"paper_dice_gamma0": 0.0, "vm_dice_gamma1": 1.0}
EPOCHS = (208, 216, 224, 232)
TRAINING_GLOBAL_EPOCHS = tuple(range(201, 233))
VALIDATION_CASES = 108
VALIDATION_SUBJECTS = 50
BOOTSTRAP_SAMPLES = 10_000
GPU_MEMORY_STOP_BYTES = 24 * 1024**3

VALIDATION_MANIFEST_SHA256 = (
    "f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb"
)
VALIDATION_CACHE_INDEX_SHA256 = (
    "9339e4575561447dd5bd99bc29e909ae1e9f566721d351a244d087a40622f322"
)
VALIDATION_LABELS_SHA256 = (
    "6ac3a3909f157eaa629c0fca0dbfb30bd78c23206317c62be2376443ce469e0e"
)
KD_VALIDATION_SHA256 = (
    "db7c3bc6b5ff9e1076827d25c9f4de0dea88781bfc5de45da0fb2f79a09bfb57"
)
KD_EPOCH = 200
KD_SCAN = 0.7343843078447713
KD_SUBJECT = 0.7244341433615912
KD_POINT = 0.9351851851851852
START_CHECKPOINT_SHA256 = (
    "56f54898ded11a0208e75f8000d57d4d053e80d6dedd760402730ad345c111b7"
)
PREREG_SHA256 = "a50aa875bb0b7cb509b8020f3eae1056d24a9cb56bb570f2f52d474b89cb0e0e"
AMENDMENT_001_SHA256 = (
    "af01410acdff1d565e5def1150432b027053bf08104f006be1fdccdcc7e7dea6"
)
AMENDMENT_002_SHA256 = (
    "a37fc5482f35008f77d445e8b0a935e830945e184af82db6a57011da4860ddc1"
)
EXPECTED_ARCHITECTURE = {
    "class": "AutomaticPromptLocator",
    "backbone": "MONAI SegResNet",
    "center_aware": False,
    "init_filters": 16,
    "dropout": 0.1,
    "input_channels": ["T1ce"],
    "outputs": ["WT segmentation logit"],
    "deploy_parameter_count": 4_700_897,
}

BOOTSTRAP_SEEDS_VS_GAMMA0 = {
    208: 20260950,
    216: 20260951,
    224: 20260952,
    232: 20260953,
}
BOOTSTRAP_SEEDS_VS_KD200 = {
    208: 20260954,
    216: 20260955,
    224: 20260956,
    232: 20260957,
}
BOOTSTRAP_SEEDS_BOUNDARY = {
    208: 20260958,
    216: 20260959,
    224: 20260960,
    232: 20260961,
}
BOOTSTRAP_SEEDS_BOTTOM27 = {
    208: 20260962,
    216: 20260963,
    224: 20260964,
    232: 20260965,
}

KD200_BOTTOM27 = (
    "BraTS-GLI-02657-100",
    "BraTS-GLI-02249-100",
    "BraTS-GLI-02208-102",
    "BraTS-GLI-02103-103",
    "BraTS-GLI-02208-100",
    "BraTS-GLI-02991-101",
    "BraTS-GLI-02208-101",
    "BraTS-GLI-02327-100",
    "BraTS-GLI-02103-104",
    "BraTS-GLI-02103-102",
    "BraTS-GLI-02249-101",
    "BraTS-GLI-02905-100",
    "BraTS-GLI-02905-101",
    "BraTS-GLI-02991-100",
    "BraTS-GLI-02720-100",
    "BraTS-GLI-02656-100",
    "BraTS-GLI-02792-100",
    "BraTS-GLI-02103-105",
    "BraTS-GLI-00498-100",
    "BraTS-GLI-02694-100",
    "BraTS-GLI-02910-101",
    "BraTS-GLI-02910-100",
    "BraTS-GLI-02792-101",
    "BraTS-GLI-02103-106",
    "BraTS-GLI-02387-100",
    "BraTS-GLI-02256-100",
    "BraTS-GLI-02912-101",
)
KD200_BOTTOM27_SHA256 = (
    "dd3828815ad4608ef3cac5d7586de42bf23b141cb23ef79fae1700ba5bf0aedb"
)
KD200_BOTTOM27_INCLUDED_DICE = 0.6852907538414001
KD200_BOTTOM27_EXCLUDED_DICE = 0.6897181868553162

BOUNDARY_TOP27 = (
    "BraTS-GLI-02657-100",
    "BraTS-GLI-02249-100",
    "BraTS-GLI-02327-100",
    "BraTS-GLI-02905-100",
    "BraTS-GLI-02219-101",
    "BraTS-GLI-02249-101",
    "BraTS-GLI-02905-101",
    "BraTS-GLI-02208-102",
    "BraTS-GLI-02910-100",
    "BraTS-GLI-03059-101",
    "BraTS-GLI-02910-101",
    "BraTS-GLI-00498-100",
    "BraTS-GLI-02219-100",
    "BraTS-GLI-02208-101",
    "BraTS-GLI-00046-100",
    "BraTS-GLI-02647-100",
    "BraTS-GLI-00505-100",
    "BraTS-GLI-02720-100",
    "BraTS-GLI-02875-100",
    "BraTS-GLI-03007-100",
    "BraTS-GLI-02219-103",
    "BraTS-GLI-02919-100",
    "BraTS-GLI-03007-101",
    "BraTS-GLI-02875-101",
    "BraTS-GLI-02694-100",
    "BraTS-GLI-00505-101",
    "BraTS-GLI-02782-100",
)
BOUNDARY_TOP27_SHA256 = (
    "668ae9bfd782009e674c4437fa83be97455ac382371716ec7d8e4fb2ed759775"
)
BOUNDARY_INCLUDED_PROXY = 0.7837987635898529
BOUNDARY_EXCLUDED_PROXY = 0.7737909516380655

ACCESS_FLAGS = (
    "validation_access",
    "outer_selector_access",
    "confirmation_access",
    "test_access",
)
PAIRING_FIELDS = (
    "initial_student_state_sha256",
    "first_batch_case_ids",
    "first_batch_case_ids_sha256",
    "first_batch_raw_image_sha256",
    "first_batch_raw_label_sha256",
    "first_batch_augmented_image_sha256",
    "first_batch_augmented_label_sha256",
    "epoch_case_order_sha256",
    "epoch_cuda_rng_before_sha256",
    "epoch_cuda_rng_after_sha256",
)


def _expected_formula_contract(arm: str) -> dict[str, Any]:
    gamma = EXPECTED_GAMMAS[arm]
    return {
        "arm": arm,
        "gamma": gamma,
        "gamma_foreground": gamma,
        "gamma_background": gamma,
        "epsilon": 1e-6,
        "class_order": ["background", "foreground"],
        "present_gt_classes_only": True,
        "modulation_detached": True,
        "denominator": "sum(m*(target^2+probability^2))",
        "bce_with_logits_weight": 0.1,
        "teacher_forwards": 0,
        "deploy": "plain student final raw foreground logit only",
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _text_lines_sha256(values: tuple[str, ...] | list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _require_false_access(value: dict[str, Any], *, source: str) -> None:
    bad = {flag: value.get(flag) for flag in ACCESS_FLAGS if value.get(flag) is not False}
    if bad:
        raise ValueError(f"{source} access flags are not all false: {bad}")


def _require_exact_zero(value: Any, *, field: str, source: str) -> None:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} lacks numeric {field}") from exc
    if not math.isfinite(number) or number != 0.0:
        raise ValueError(f"{source} {field} must equal zero")


def _require_finite(value: Any, *, source: str) -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{source} contains non-finite value")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite(item, source=f"{source}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite(item, source=f"{source}[{index}]")
        return
    raise ValueError(f"{source} contains unsupported type")


def _validate_pairing(
    value: Any, *, source: str, through_epoch: int = 232
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{source} is not an object")
    for field in PAIRING_FIELDS:
        item = value.get(field)
        if item is None or item == "" or item == [] or item == {}:
            raise ValueError(f"{source} lacks {field}")
    if not isinstance(value["first_batch_case_ids"], list) or len(
        value["first_batch_case_ids"]
    ) != 8:
        raise ValueError(f"{source} first batch is not batch8")
    expected_epochs = {
        str(epoch) for epoch in TRAINING_GLOBAL_EPOCHS if epoch <= through_epoch
    }
    for field in PAIRING_FIELDS[-3:]:
        if not isinstance(value[field], dict) or set(value[field]) != expected_epochs:
            raise ValueError(f"{source}.{field} epoch coverage differs")
    return value


def _validate_summary(
    summary: dict[str, Any], *, source: str, expected_case_ids: list[str]
) -> None:
    if summary.get("cases") != VALIDATION_CASES or summary.get("subjects") != VALIDATION_SUBJECTS:
        raise ValueError(f"{source} cohort counts differ")
    per_case = summary.get("per_case")
    if not isinstance(per_case, list) or len(per_case) != VALIDATION_CASES:
        raise ValueError(f"{source} per_case is incomplete")
    if [str(item.get("case_id")) for item in per_case] != expected_case_ids:
        raise ValueError(f"{source} case order differs from frozen manifest order")
    if len({str(item.get("subject_id")) for item in per_case}) != VALIDATION_SUBJECTS:
        raise ValueError(f"{source} subject count differs")
    for field in (
        "locator_dsc_mean",
        "locator_subject_dsc_mean",
        "automatic_point_hit_rate",
    ):
        if not math.isfinite(float(summary[field])):
            raise ValueError(f"{source} contains non-finite {field}")


def _validate_kd_report(
    path: Path, expected_case_ids: list[str]
) -> tuple[dict[str, Any], str]:
    digest = sha256_file(path)
    if digest != KD_VALIDATION_SHA256:
        raise ValueError("KD_VALIDATION SHA-256 mismatch")
    report = _read_json(path)
    if report.get("schema") != KD_VALIDATION_SCHEMA or report.get("schema_version") != 1:
        raise ValueError("unexpected KD_VALIDATION schema")
    if report.get("selected_kd_epoch") != KD_EPOCH:
        raise ValueError("KD_VALIDATION no longer selects epoch200")
    if report.get("outer_selector_access") is not False or report.get(
        "confirmation_access"
    ) is not False or report.get("test_access") is not False:
        raise ValueError("KD_VALIDATION declares forbidden later-cohort access")
    kd200 = report.get("summaries", {}).get("kd", {}).get(str(KD_EPOCH))
    if not isinstance(kd200, dict):
        raise ValueError("KD_VALIDATION lacks KD200 summary")
    _validate_summary(kd200, source="historical KD200", expected_case_ids=expected_case_ids)
    for field, expected in (
        ("locator_dsc_mean", KD_SCAN),
        ("locator_subject_dsc_mean", KD_SUBJECT),
        ("automatic_point_hit_rate", KD_POINT),
    ):
        if float(kd200[field]) != expected:
            raise ValueError(f"KD200 {field} changed")
    ordered = sorted(
        kd200["per_case"],
        key=lambda item: (float(item["locator_dsc"]), str(item["case_id"])),
    )
    bottom = tuple(str(item["case_id"]) for item in ordered[:27])
    if bottom != KD200_BOTTOM27 or _text_lines_sha256(bottom) != KD200_BOTTOM27_SHA256:
        raise ValueError("frozen KD200 bottom27 stratum changed")
    if float(ordered[26]["locator_dsc"]) != KD200_BOTTOM27_INCLUDED_DICE:
        raise ValueError("bottom27 included cutoff changed")
    if float(ordered[27]["locator_dsc"]) != KD200_BOTTOM27_EXCLUDED_DICE:
        raise ValueError("bottom27 excluded cutoff changed")
    return report, digest


def _validate_implementation_receipt(path: Path) -> tuple[dict[str, Any], str]:
    receipt = _read_json(path)
    if receipt.get("schema") != IMPLEMENTATION_RECEIPT_SCHEMA or receipt.get(
        "schema_version"
    ) != 1:
        raise ValueError("unexpected VM Dice implementation receipt")
    _require_false_access(receipt, source="VM Dice implementation receipt")
    for field in (
        "synthetic_all_gates_passed",
        "gpu_smoke_all_gates_passed",
        "all_gates_passed",
    ):
        if receipt.get(field) is not True:
            raise ValueError(f"implementation receipt {field} != true")
    implementation = receipt.get("implementation")
    if not isinstance(implementation, dict):
        raise ValueError("implementation receipt lacks code bindings")
    if implementation.get("trainer_path") != TRAINER_ENTRYPOINT:
        raise ValueError("implementation receipt trainer path differs")
    if not isinstance(implementation.get("trainer_sha256"), str):
        raise ValueError("implementation receipt lacks trainer SHA")
    if implementation.get("validation_evaluator_path") != str(
        Path(__file__).resolve().relative_to(ROOT)
    ):
        raise ValueError("implementation receipt evaluator path differs")
    if implementation.get("validation_evaluator_sha256") != sha256_file(
        Path(__file__).resolve()
    ):
        raise ValueError("implementation receipt evaluator SHA differs")
    if implementation.get("model_path") != "sammed3d/t1ce_wt_vm_dice.py":
        raise ValueError("implementation receipt model path differs")
    if implementation.get("model_sha256") != sha256_file(
        ROOT / "sammed3d/t1ce_wt_vm_dice.py"
    ):
        raise ValueError("implementation receipt model SHA differs")
    if implementation.get("test_path") != "tests/test_t1ce_wt_vm_dice.py":
        raise ValueError("implementation receipt test path differs")
    if implementation.get("test_sha256") != sha256_file(
        ROOT / "tests/test_t1ce_wt_vm_dice.py"
    ):
        raise ValueError("implementation receipt test SHA differs")
    return receipt, sha256_file(path)


def _validate_training_report(
    path: Path, *, arm: str
) -> tuple[dict[str, Any], str]:
    report = _read_json(path)
    source = f"{arm} TRAIN_DONE"
    if report.get("schema") != TRAINING_REPORT_SCHEMA or report.get("schema_version") != 1:
        raise ValueError(f"unexpected {source} schema")
    if report.get("arm") != arm or float(report.get("gamma")) != EXPECTED_GAMMAS[arm]:
        raise ValueError(f"{source} arm/gamma mismatch")
    if report.get("status") != "complete" or report.get("all_self_gates_passed") is not True:
        raise ValueError(f"{source} did not complete self gates")
    _require_false_access(report, source=source)
    for field in ("reload_max_abs_difference", "stripped_deploy_max_abs_difference"):
        _require_exact_zero(report.get(field), field=field, source=source)
    if int(report.get("peak_gpu_memory_bytes", -1)) > GPU_MEMORY_STOP_BYTES:
        raise ValueError(f"{source} exceeded formal 24-GiB gate")
    if report.get("start_checkpoint_sha256") != START_CHECKPOINT_SHA256:
        raise ValueError(f"{source} start checkpoint changed")
    expected_provenance = {
        "preregistration_sha256": PREREG_SHA256,
        "amendment_001_sha256": AMENDMENT_001_SHA256,
        "amendment_002_sha256": AMENDMENT_002_SHA256,
    }
    for field, expected in expected_provenance.items():
        if report.get(field) != expected:
            raise ValueError(f"{source} {field} differs")
    if report.get("vm_dice") != _expected_formula_contract(arm):
        raise ValueError(f"{source} formula contract differs")
    _validate_pairing(report.get("pairing"), source=f"{source}.pairing")
    history = report.get("history")
    if not isinstance(history, list) or [item.get("global_epoch") for item in history] != list(
        TRAINING_GLOBAL_EPOCHS
    ):
        raise ValueError(f"{source} history differs")
    if any(float(item.get("learning_rate")) != 1e-5 for item in history):
        raise ValueError(f"{source} learning rate differs")
    _require_finite(history, source=f"{source}.history")
    checkpoints = report.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != {
        str(epoch) for epoch in EPOCHS
    }:
        raise ValueError(f"{source} checkpoint grid differs")
    for epoch in EPOCHS:
        item = checkpoints[str(epoch)]
        if not isinstance(item, dict) or not isinstance(item.get("sha256"), str):
            raise ValueError(f"{source} epoch{epoch} binding missing")
        _require_exact_zero(
            item.get("checkpoint_reload_max_abs_difference"),
            field="checkpoint_reload_max_abs_difference",
            source=f"{source} epoch{epoch}",
        )
        _require_exact_zero(
            item.get("stripped_deploy_max_abs_difference"),
            field="stripped_deploy_max_abs_difference",
            source=f"{source} epoch{epoch}",
        )
    return report, sha256_file(path)


def _validate_pairing_receipt(
    path: Path,
    *,
    reports: dict[str, dict[str, Any]],
    report_sha: dict[str, str],
) -> tuple[dict[str, Any], str]:
    receipt = _read_json(path)
    if receipt.get("schema") != PAIRING_RECEIPT_SCHEMA or receipt.get(
        "schema_version"
    ) != 1:
        raise ValueError("unexpected VM Dice pairing receipt")
    _require_false_access(receipt, source="VM Dice pairing receipt")
    if receipt.get("all_gates_passed") is not True or receipt.get(
        "formal_peak_memory_at_most_24_gib_each"
    ) is not True:
        raise ValueError("VM Dice pairing receipt gates failed")
    arms = receipt.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise ValueError("pairing receipt arm set differs")
    for arm in ARMS:
        item = arms[arm]
        if item.get("training_report_sha256") != report_sha[arm]:
            raise ValueError(f"pairing receipt {arm} report SHA differs")
        if item.get("pairing") != reports[arm]["pairing"]:
            raise ValueError(f"pairing receipt {arm} evidence differs")
    mismatched = [
        field
        for field in PAIRING_FIELDS
        if arms[ARMS[0]]["pairing"][field] != arms[ARMS[1]]["pairing"][field]
    ]
    if mismatched:
        raise ValueError(f"paired arms differ: {mismatched}")
    return receipt, sha256_file(path)


def _load_model(
    path: Path,
    *,
    arm: str,
    epoch: int,
    expected_sha: str,
    device: torch.device,
) -> tuple[AutomaticPromptLocator, dict[str, Any]]:
    if sha256_file(path) != expected_sha:
        raise ValueError(f"{arm} epoch{epoch} checkpoint SHA differs")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = f"{arm} checkpoint epoch{epoch}"
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{source} is not a dictionary")
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA or checkpoint.get("schema_version") != 1:
        raise ValueError(f"unexpected {source} schema")
    if checkpoint.get("arm") != arm or checkpoint.get(
        "vm_dice"
    ) != _expected_formula_contract(arm):
        raise ValueError(f"{source} arm/gamma differs")
    if checkpoint.get("global_epoch") != epoch or checkpoint.get(
        "continuation_epoch"
    ) != epoch - 200:
        raise ValueError(f"{source} epoch metadata differs")
    if checkpoint.get("resolution") != 128 or float(checkpoint.get("learning_rate")) != 1e-5:
        raise ValueError(f"{source} resolution/LR differs")
    _require_false_access(checkpoint, source=source)
    if checkpoint.get("start_checkpoint_sha256") != START_CHECKPOINT_SHA256:
        raise ValueError(f"{source} start checkpoint differs")
    for field, expected in (
        ("preregistration_sha256", PREREG_SHA256),
        ("amendment_001_sha256", AMENDMENT_001_SHA256),
        ("amendment_002_sha256", AMENDMENT_002_SHA256),
    ):
        if checkpoint.get(field) != expected:
            raise ValueError(f"{source} {field} differs")
    if checkpoint.get("start_checkpoint_historical_validation_access") is not True or checkpoint.get(
        "start_checkpoint_historical_test_access_after_selection"
    ) is not True:
        raise ValueError(f"{source} historical access disclosure missing")
    if checkpoint.get("architecture") != EXPECTED_ARCHITECTURE:
        raise ValueError(f"{source} architecture differs")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"{source} model state missing")
    _validate_pairing(
        checkpoint.get("pairing"), source=f"{source}.pairing", through_epoch=epoch
    )
    _require_finite(checkpoint.get("loss_diagnostics"), source=f"{source}.loss")
    for field in (
        "checkpoint_reload_max_abs_difference",
        "stripped_deploy_max_abs_difference",
    ):
        _require_exact_zero(checkpoint.get(field), field=field, source=source)
    model = AutomaticPromptLocator(
        center_aware=False,
        init_filters=16,
        dropout=0.1,
    ).to(device)
    model.load_state_dict(state, strict=True)
    return model, checkpoint


def _validate_cache_and_boundary_stratum(
    cache: Path, manifest: Path
) -> tuple[BrainCropDataset, list[str], dict[str, float]]:
    if sha256_file(manifest) != VALIDATION_MANIFEST_SHA256:
        raise ValueError("validation manifest SHA differs")
    index_path = cache / "index.json"
    if sha256_file(index_path) != VALIDATION_CACHE_INDEX_SHA256:
        raise ValueError("validation cache index SHA differs")
    index = _read_json(index_path)
    if index.get("labels_sha256") != VALIDATION_LABELS_SHA256 or sha256_file(
        cache / "labels.u8"
    ) != VALIDATION_LABELS_SHA256:
        raise ValueError("validation labels SHA differs")
    case_ids = read_case_ids(manifest)
    if case_ids is None:
        raise ValueError("validation manifest produced no cases")
    data = BrainCropDataset(cache, expected_split="validation", case_ids=case_ids)
    if len(data) != VALIDATION_CASES or len(
        {case.rsplit("-", 1)[0] for case in data.case_ids}
    ) != VALIDATION_SUBJECTS:
        raise ValueError("validation cohort counts differ")
    available = [str(value) for value in index["case_ids"]]
    slot = {case_id: position for position, case_id in enumerate(available)}
    labels = np.memmap(
        cache / "labels.u8",
        dtype=np.uint8,
        mode="r",
        shape=(int(index["count"]), 128, 128, 128),
    )
    proxies: dict[str, float] = {}
    for case_id in case_ids:
        mask = np.asarray(labels[slot[case_id]]) > 0
        volume = int(mask.sum())
        if volume <= 0:
            raise ValueError(f"empty validation GT: {case_id}")
        adjacent = int(
            (mask[1:, :, :] & mask[:-1, :, :]).sum()
            + (mask[:, 1:, :] & mask[:, :-1, :]).sum()
            + (mask[:, :, 1:] & mask[:, :, :-1]).sum()
        )
        proxies[case_id] = float((6 * volume - 2 * adjacent) / volume)
    ordered = sorted(case_ids, key=lambda case_id: (-proxies[case_id], case_id))
    boundary = tuple(ordered[:27])
    if boundary != BOUNDARY_TOP27 or _text_lines_sha256(boundary) != BOUNDARY_TOP27_SHA256:
        raise ValueError("frozen boundary top27 stratum changed")
    if proxies[ordered[26]] != BOUNDARY_INCLUDED_PROXY:
        raise ValueError("boundary included cutoff changed")
    if proxies[ordered[27]] != BOUNDARY_EXCLUDED_PROXY:
        raise ValueError("boundary excluded cutoff changed")
    return data, case_ids, proxies


def _subject_values(per_case: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    grouped: dict[str, list[float]] = defaultdict(list)
    order: list[str] = []
    for item in per_case:
        subject = str(item["subject_id"])
        if subject not in grouped:
            order.append(subject)
        grouped[subject].append(float(item["locator_dsc"]))
    return order, np.asarray(
        [statistics.fmean(grouped[subject]) for subject in order], dtype=np.float64
    )


def _compare_segmentation(
    comparator: dict[str, Any],
    candidate: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    left = comparator["per_case"]
    right = candidate["per_case"]
    if [item["case_id"] for item in left] != [item["case_id"] for item in right]:
        raise ValueError("paired segmentation case order differs")
    case_delta = np.asarray(
        [float(r["locator_dsc"]) - float(l["locator_dsc"]) for l, r in zip(left, right, strict=True)],
        dtype=np.float64,
    )
    left_order, left_subject = _subject_values(left)
    right_order, right_subject = _subject_values(right)
    if left_order != right_order:
        raise ValueError("paired subject order differs")
    subject_delta = right_subject - left_subject
    rng = np.random.default_rng(seed)
    case_draws = rng.integers(
        0, len(case_delta), size=(BOOTSTRAP_SAMPLES, len(case_delta))
    )
    subject_draws = rng.integers(
        0, len(subject_delta), size=(BOOTSTRAP_SAMPLES, len(subject_delta))
    )
    scan_ci = np.quantile(
        case_delta[case_draws].mean(axis=1),
        [0.025, 0.975],
        method="linear",
    )
    subject_ci = np.quantile(
        subject_delta[subject_draws].mean(axis=1),
        [0.025, 0.975],
        method="linear",
    )
    return {
        "candidate_minus_comparator": {
            "scan": float(case_delta.mean()),
            "subject": float(subject_delta.mean()),
            "point_hit": float(
                candidate["automatic_point_hit_rate"]
                - comparator["automatic_point_hit_rate"]
            ),
            "scan_95ci": [float(value) for value in scan_ci],
            "subject_95ci": [float(value) for value in subject_ci],
            "cases_improved_tied_worse": [
                int((case_delta > 0).sum()),
                int((case_delta == 0).sum()),
                int((case_delta < 0).sum()),
            ],
            "subjects_improved_tied_worse": [
                int((subject_delta > 0).sum()),
                int((subject_delta == 0).sum()),
                int((subject_delta < 0).sum()),
            ],
            "bootstrap_seed": seed,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "quantile_method": "linear",
        }
    }


def _compare_stratum(
    gamma0: dict[str, Any],
    gamma1: dict[str, Any],
    *,
    case_ids: tuple[str, ...],
    seed: int,
) -> dict[str, Any]:
    left = {str(item["case_id"]): float(item["locator_dsc"]) for item in gamma0["per_case"]}
    right = {str(item["case_id"]): float(item["locator_dsc"]) for item in gamma1["per_case"]}
    if any(case_id not in left or case_id not in right for case_id in case_ids):
        raise ValueError("frozen stratum case absent from summary")
    delta = np.asarray([right[case_id] - left[case_id] for case_id in case_ids])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(BOOTSTRAP_SAMPLES, len(delta)))
    interval = np.quantile(
        delta[draws].mean(axis=1), [0.025, 0.975], method="linear"
    )
    return {
        "case_ids": list(case_ids),
        "text_lines_sha256": _text_lines_sha256(case_ids),
        "n": len(case_ids),
        "gamma1_minus_gamma0_mean": float(delta.mean()),
        "gamma1_minus_gamma0_95ci": [float(value) for value in interval],
        "improved_tied_worse": [
            int((delta > 0).sum()),
            int((delta == 0).sum()),
            int((delta < 0).sum()),
        ],
        "per_case_delta": [
            {"case_id": case_id, "delta": float(value)}
            for case_id, value in zip(case_ids, delta, strict=True)
        ],
        "bootstrap_seed": seed,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "quantile_method": "linear",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--gamma0-dir", type=Path, required=True)
    parser.add_argument("--gamma1-dir", type=Path, required=True)
    parser.add_argument("--gamma0-training-report", type=Path, required=True)
    parser.add_argument("--gamma1-training-report", type=Path, required=True)
    parser.add_argument("--pairing-receipt", type=Path, required=True)
    parser.add_argument("--implementation-receipt", type=Path, required=True)
    parser.add_argument("--kd-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    started_path = args.output.with_suffix(".started.json")
    if args.output.exists() or started_path.exists():
        raise FileExistsError("refusing repeated VM Dice validation")
    if args.batch_size != 8:
        raise ValueError("fixed VM Dice validation batch size must remain8")

    data, case_ids, boundary_proxy = _validate_cache_and_boundary_stratum(
        args.cache, args.validation_manifest
    )
    kd_report, kd_sha = _validate_kd_report(args.kd_validation, case_ids)
    implementation, implementation_sha = _validate_implementation_receipt(
        args.implementation_receipt
    )
    reports: dict[str, dict[str, Any]] = {}
    report_sha: dict[str, str] = {}
    report_paths = {
        "paper_dice_gamma0": args.gamma0_training_report,
        "vm_dice_gamma1": args.gamma1_training_report,
    }
    for arm in ARMS:
        reports[arm], report_sha[arm] = _validate_training_report(
            report_paths[arm], arm=arm
        )
    if implementation["implementation"]["trainer_sha256"] != reports[
        "vm_dice_gamma1"
    ].get("runner_sha256"):
        raise ValueError("implementation trainer SHA differs from TRAIN_DONE")
    pairing, pairing_sha = _validate_pairing_receipt(
        args.pairing_receipt, reports=reports, report_sha=report_sha
    )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("VM Dice validation requires CUDA")
    torch.cuda.set_device(device)
    arm_dirs = {
        "paper_dice_gamma0": args.gamma0_dir,
        "vm_dice_gamma1": args.gamma1_dir,
    }
    checkpoint_paths = {
        arm: {
            epoch: arm_dirs[arm] / f"plain_128_{arm}_global_epoch{epoch}.pth"
            for epoch in EPOCHS
        }
        for arm in ARMS
    }
    checkpoint_sha = {
        arm: {
            epoch: str(reports[arm]["checkpoints"][str(epoch)]["sha256"])
            for epoch in EPOCHS
        }
        for arm in ARMS
    }
    preloaded: dict[str, dict[int, tuple[AutomaticPromptLocator, dict[str, Any]]]] = {
        arm: {} for arm in ARMS
    }
    frozen_pairing = pairing["arms"]["paper_dice_gamma0"]["pairing"]
    for arm in ARMS:
        for epoch in EPOCHS:
            preloaded[arm][epoch] = _load_model(
                checkpoint_paths[arm][epoch],
                arm=arm,
                epoch=epoch,
                expected_sha=checkpoint_sha[arm][epoch],
                device=device,
            )
            checkpoint_pairing = preloaded[arm][epoch][1]["pairing"]
            for field in PAIRING_FIELDS[:-3]:
                if checkpoint_pairing[field] != frozen_pairing[field]:
                    raise ValueError(f"{arm} epoch{epoch} scalar pairing differs")
            for field in PAIRING_FIELDS[-3:]:
                expected = {
                    str(global_epoch): frozen_pairing[field][str(global_epoch)]
                    for global_epoch in TRAINING_GLOBAL_EPOCHS
                    if global_epoch <= epoch
                }
                if checkpoint_pairing[field] != expected:
                    raise ValueError(f"{arm} epoch{epoch} {field} differs")

    write_json_atomic(
        started_path,
        {
            "schema": SCHEMA + ".started",
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_scope": "fixed validation108; development selection only",
            "selection_key": "(gamma1_scan,gamma1_subject,-global_epoch)",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seeds": {
                "vs_gamma0": {str(e): BOOTSTRAP_SEEDS_VS_GAMMA0[e] for e in EPOCHS},
                "vs_kd200": {str(e): BOOTSTRAP_SEEDS_VS_KD200[e] for e in EPOCHS},
                "boundary27": {str(e): BOOTSTRAP_SEEDS_BOUNDARY[e] for e in EPOCHS},
                "bottom27": {str(e): BOOTSTRAP_SEEDS_BOTTOM27[e] for e in EPOCHS},
            },
            "frozen_strata": {
                "kd200_bottom27_sha256": KD200_BOTTOM27_SHA256,
                "boundary_top27_sha256": BOUNDARY_TOP27_SHA256,
                "overlap_cases": 13,
            },
            "validation_manifest_sha256": VALIDATION_MANIFEST_SHA256,
            "validation_cache_index_sha256": VALIDATION_CACHE_INDEX_SHA256,
            "validation_labels_sha256": VALIDATION_LABELS_SHA256,
            "kd_validation_sha256": kd_sha,
            "implementation_receipt_sha256": implementation_sha,
            "pairing_receipt_sha256": pairing_sha,
            "training_report_sha256": report_sha,
            "checkpoint_sha256": {
                arm: {str(e): checkpoint_sha[arm][e] for e in EPOCHS}
                for arm in ARMS
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
    summaries: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    checkpoints: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for epoch in EPOCHS:
            key = str(epoch)
            model, checkpoint = preloaded[arm].pop(epoch)
            summaries[arm][key] = validate(model, loader, device)
            _validate_summary(
                summaries[arm][key], source=f"{arm} epoch{epoch}", expected_case_ids=case_ids
            )
            checkpoints[arm][key] = {
                "path": str(checkpoint_paths[arm][epoch].resolve()),
                "sha256": checkpoint_sha[arm][epoch],
                "arm": arm,
                "gamma": EXPECTED_GAMMAS[arm],
                "global_epoch": epoch,
                "continuation_epoch": epoch - 200,
                "vm_dice": checkpoint["vm_dice"],
                "loss_diagnostics": checkpoint["loss_diagnostics"],
                "pairing": checkpoint["pairing"],
                "checkpoint_reload_max_abs_difference": checkpoint[
                    "checkpoint_reload_max_abs_difference"
                ],
                "stripped_deploy_max_abs_difference": checkpoint[
                    "stripped_deploy_max_abs_difference"
                ],
            }
            del model
            torch.cuda.empty_cache()

    selected_epoch = max(
        EPOCHS,
        key=lambda epoch: (
            float(summaries["vm_dice_gamma1"][str(epoch)]["locator_dsc_mean"]),
            float(
                summaries["vm_dice_gamma1"][str(epoch)][
                    "locator_subject_dsc_mean"
                ]
            ),
            -epoch,
        ),
    )
    selected_key = str(selected_epoch)
    selected = summaries["vm_dice_gamma1"][selected_key]
    selected_control = summaries["paper_dice_gamma0"][selected_key]
    kd200 = kd_report["summaries"]["kd"][str(KD_EPOCH)]

    versus_gamma0 = {
        str(epoch): _compare_segmentation(
            summaries["paper_dice_gamma0"][str(epoch)],
            summaries["vm_dice_gamma1"][str(epoch)],
            seed=BOOTSTRAP_SEEDS_VS_GAMMA0[epoch],
        )
        for epoch in EPOCHS
    }
    versus_kd200 = {
        str(epoch): _compare_segmentation(
            kd200,
            summaries["vm_dice_gamma1"][str(epoch)],
            seed=BOOTSTRAP_SEEDS_VS_KD200[epoch],
        )
        for epoch in EPOCHS
    }
    boundary_results = {
        str(epoch): _compare_stratum(
            summaries["paper_dice_gamma0"][str(epoch)],
            summaries["vm_dice_gamma1"][str(epoch)],
            case_ids=BOUNDARY_TOP27,
            seed=BOOTSTRAP_SEEDS_BOUNDARY[epoch],
        )
        for epoch in EPOCHS
    }
    bottom_results = {
        str(epoch): _compare_stratum(
            summaries["paper_dice_gamma0"][str(epoch)],
            summaries["vm_dice_gamma1"][str(epoch)],
            case_ids=KD200_BOTTOM27,
            seed=BOOTSTRAP_SEEDS_BOTTOM27[epoch],
        )
        for epoch in EPOCHS
    }
    delta_control = versus_gamma0[selected_key]["candidate_minus_comparator"]
    delta_kd = versus_kd200[selected_key]["candidate_minus_comparator"]
    selected_boundary = boundary_results[selected_key]
    selected_bottom = bottom_results[selected_key]

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
        "both_formal_training_reports_complete": all(
            reports[arm]["status"] == "complete" for arm in ARMS
        ),
        "both_formal_peak_memory_at_most_24_gib": all(
            int(reports[arm]["peak_gpu_memory_bytes"]) <= GPU_MEMORY_STOP_BYTES
            for arm in ARMS
        ),
        "pairing_receipt_all_gates_passed": pairing["all_gates_passed"] is True,
        "all_checkpoint_reloads_and_deploy_exports_exact": all(
            float(checkpoints[arm][str(epoch)]["checkpoint_reload_max_abs_difference"])
            == 0.0
            and float(
                checkpoints[arm][str(epoch)]["stripped_deploy_max_abs_difference"]
            )
            == 0.0
            for arm in ARMS
            for epoch in EPOCHS
        ),
        "frozen_strata_hashes_and_cutoffs_verified": True,
        "all_access_and_start_disclosures_verified": True,
    }
    promotion_gates = {
        "selected_beats_same_epoch_gamma0_scan": float(selected["locator_dsc_mean"])
        > float(selected_control["locator_dsc_mean"]),
        "selected_beats_same_epoch_gamma0_subject": float(
            selected["locator_subject_dsc_mean"]
        )
        > float(selected_control["locator_subject_dsc_mean"]),
        "selected_beats_historical_kd200_scan": float(selected["locator_dsc_mean"])
        > float(kd200["locator_dsc_mean"]),
        "selected_beats_historical_kd200_subject": float(
            selected["locator_subject_dsc_mean"]
        )
        > float(kd200["locator_subject_dsc_mean"]),
        "selected_point_at_least_101_of_108": float(
            selected["automatic_point_hit_rate"]
        )
        >= 101.0 / VALIDATION_CASES,
        "paired_scan_ci_lower_at_least_minus_0.003_vs_gamma0": float(
            delta_control["scan_95ci"][0]
        )
        >= -0.003,
        "paired_subject_ci_lower_at_least_minus_0.003_vs_gamma0": float(
            delta_control["subject_95ci"][0]
        )
        >= -0.003,
        "paired_scan_ci_lower_at_least_minus_0.003_vs_kd200": float(
            delta_kd["scan_95ci"][0]
        )
        >= -0.003,
        "paired_subject_ci_lower_at_least_minus_0.003_vs_kd200": float(
            delta_kd["subject_95ci"][0]
        )
        >= -0.003,
        "boundary_top27_mean_delta_strictly_positive": float(
            selected_boundary["gamma1_minus_gamma0_mean"]
        )
        > 0.0,
        "kd200_bottom27_mean_delta_strictly_positive": float(
            selected_bottom["gamma1_minus_gamma0_mean"]
        )
        > 0.0,
        "all_prerequisite_gates_passed": all(prerequisite_gates.values()),
    }
    all_gates_passed = all(prerequisite_gates.values()) and all(
        promotion_gates.values()
    )

    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "raw fixed-validation report",
        "evidence_scope": "fixed validation108; development selection only",
        "protocol": {
            "trainer_entrypoint": TRAINER_ENTRYPOINT,
            "method": "paired paper-Dice gamma0 versus VM-Dice gamma1",
            "resolution": 128,
            "threshold": "raw logit > 0",
            "tta": False,
            "sam": False,
            "postprocessing": False,
            "locator_forwards": 1,
            "epochs": list(EPOCHS),
            "selection_key": "(gamma1_scan,gamma1_subject,-global_epoch)",
            "strata_or_point_select_checkpoint": False,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_quantile_method": "linear",
            "checkpoint_selection_uses_test": False,
        },
        "data": {
            "manifest": str(args.validation_manifest.resolve()),
            "manifest_sha256": VALIDATION_MANIFEST_SHA256,
            "cache": str(args.cache.resolve()),
            "cache_index_sha256": VALIDATION_CACHE_INDEX_SHA256,
            "labels_sha256": VALIDATION_LABELS_SHA256,
            "cases": VALIDATION_CASES,
            "subjects": VALIDATION_SUBJECTS,
        },
        "evidence_inputs": {
            "kd_validation": str(args.kd_validation.resolve()),
            "kd_validation_sha256": kd_sha,
            "implementation_receipt": str(args.implementation_receipt.resolve()),
            "implementation_receipt_sha256": implementation_sha,
            "pairing_receipt": str(args.pairing_receipt.resolve()),
            "pairing_receipt_sha256": pairing_sha,
            "training_report_sha256": report_sha,
        },
        "frozen_strata": {
            "evidence_warning": (
                "both strata use reused fixed validation and overlap in13 cases; "
                "they are mechanism gates, not independent confirmation"
            ),
            "kd200_bottom27": {
                "case_ids": list(KD200_BOTTOM27),
                "text_lines_sha256": KD200_BOTTOM27_SHA256,
                "included_cutoff_dice": KD200_BOTTOM27_INCLUDED_DICE,
                "excluded_next_dice": KD200_BOTTOM27_EXCLUDED_DICE,
            },
            "boundary_top27": {
                "case_ids": list(BOUNDARY_TOP27),
                "text_lines_sha256": BOUNDARY_TOP27_SHA256,
                "included_cutoff_proxy": BOUNDARY_INCLUDED_PROXY,
                "excluded_next_proxy": BOUNDARY_EXCLUDED_PROXY,
                "case_proxy": {
                    case_id: boundary_proxy[case_id] for case_id in BOUNDARY_TOP27
                },
                "grid_warning": (
                    "128^3 resampled six-neighbor exposed-face proxy; not physical-mm surface"
                ),
            },
            "overlap_cases": len(set(KD200_BOTTOM27) & set(BOUNDARY_TOP27)),
        },
        "checkpoints": checkpoints,
        "summaries": summaries,
        "historical_kd200_summary": kd200,
        "gamma1_minus_same_epoch_gamma0": versus_gamma0,
        "gamma1_minus_historical_kd200": versus_kd200,
        "boundary_top27_gamma1_minus_gamma0": boundary_results,
        "kd200_bottom27_gamma1_minus_gamma0": bottom_results,
        "selected_epoch": selected_epoch,
        "selected_checkpoint": checkpoints["vm_dice_gamma1"][selected_key],
        "selected_summary": selected,
        "selected_same_epoch_gamma0_summary": selected_control,
        "selected_minus_same_epoch_gamma0": delta_control,
        "selected_minus_historical_kd200": delta_kd,
        "selected_boundary_top27": selected_boundary,
        "selected_kd200_bottom27": selected_bottom,
        "prerequisite_gates": prerequisite_gates,
        "promotion_gates": promotion_gates,
        "all_gates_passed": all_gates_passed,
        "vm_dice_candidate_authorized_for_test": all_gates_passed,
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
                "selected_boundary_top27_mean_delta": selected_boundary[
                    "gamma1_minus_gamma0_mean"
                ],
                "selected_kd200_bottom27_mean_delta": selected_bottom[
                    "gamma1_minus_gamma0_mean"
                ],
                "promotion_gates": promotion_gates,
                "vm_dice_candidate_authorized_for_test": all_gates_passed,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
