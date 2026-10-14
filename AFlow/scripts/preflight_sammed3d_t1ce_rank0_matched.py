#!/usr/bin/env python3
"""Fail-closed preflight for the matched T1ce/WT whole-volume rank-0 arm.

This command only reads the frozen base checkpoint as bytes, the 972/108
development manifests, the completed LoRA arm's training-only metadata, and
the implementation files bound by the preregistration. It never loads image
or label volumes and it does not launch training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREG = (
    ROOT
    / "workspace/BraTS/sammed3d_t1ce_wt/"
    "FV_RANK0_MATCHED_AUG_PREREG_20260724.json"
)
SCHEMA = "aflow.sammed3d.t1ce_wt.fv_rank0_matched_preflight"

EXPECTED_HASHES = {
    "base_checkpoint": "899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c",
    "train_manifest": "ea21e2306321b71d821146ed01198f52b612d390c0736f8703f81a259c06c23e",
    "validation_manifest": "fc750ab15266b8a53c6d9a7ff62bd043b10cc21733e9b68354a2a989fd1f488c",
    "trainer": "369986ff41166ffbbe8081b5767e1f276ac14a7ddbe03acce3b800d6ba4b0988",
    "generic_trainer": "0c5026e98b2e0f848c16b436610e9b7eca391995c062ac128ce1bee26d1bfa4f",
    "preprocessing_module": "4d13e2b3894e2b9a91fac7f76dd74b1c744429eef665f09a5f160a604b0989ae",
    "peft_module": "4844b1a086c74737f786399c4c67a2b3c56ae34f35b152e423f63591f226bc63",
    "engine_module": "f101db584f4f4ef1add180b03af99f775ee035dd63d1cbee04528986f3a9852a",
    "lora_train_config": "b509c4ab2baae2dc68967be097f69e649965e96c978fe51e988d6c0474973e93",
    "lora_training_protocol": "df71f27366fd4f0503b00acc84b2858761fc71d51f2ce7e2bd6488acb0b90a93",
}

EXPECTED_SHARED_ARGS = {
    "accumulation_steps": 2,
    "allow_nonempty_output": False,
    "amp_dtype": "bf16",
    "batch_size": 1,
    "click_method": "random",
    "device": "cuda",
    "encoder_qv_lora_alpha": 8.0,
    "epochs": 50,
    "lr": 0.0001,
    "max_train_cases": 0,
    "max_val_cases": 0,
    "no_amp": False,
    "num_clicks": 3,
    "num_workers": 2,
    "overfit": False,
    "seed": 20260725,
    "selection_metric": "val_dice",
    "strategy": "decoder_only",
    "subset_mode": "balanced",
    "target_sampling": "natural",
    "target_token": False,
    "targets": "",
    "train_jsonl": (
        "data/datasets/brats2024_sammed3d_t1ce_wt_"
        "inner90_10_seed20260723_v1/train_t1ce_wt.jsonl"
    ),
    "train_rows_per_target": 0,
    "train_samples_per_epoch": 0,
    "val_jsonl": (
        "data/datasets/brats2024_sammed3d_t1ce_wt_"
        "inner90_10_seed20260723_v1/validation_t1ce_wt.jsonl"
    ),
    "val_rows_per_target": 0,
    "val_seed": 20260723,
    "weight_decay": 0.1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} mismatch: {actual!r} != {expected!r}")


def resolve_bound_file(root: Path, raw_path: str, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError(f"{label} path must be a non-empty string")
    path = Path(raw_path)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    return resolved


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return payload


def validate_bound_hash(
    root: Path,
    record: dict[str, Any],
    label: str,
    expected_sha256: str,
) -> Path:
    if not isinstance(record, dict):
        raise RuntimeError(f"{label} binding must be a JSON object")
    path = resolve_bound_file(root, record.get("path"), label)
    require_equal(record.get("sha256"), expected_sha256, f"{label} prereg SHA256")
    require_equal(sha256_file(path), expected_sha256, f"{label} byte SHA256")
    return path


def read_manifest(path: Path, expected_split: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid JSON in {path} line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"{path} line {line_number} is not an object")
            require_equal(row.get("split"), expected_split, "manifest split")
            require_equal(row.get("modality"), "t1ce", "manifest modality")
            require_equal(row.get("target"), "WT", "manifest target")
            require_equal(
                row.get("foreground_labels"),
                [1, 2, 3],
                "manifest foreground labels",
            )
            require_equal(row.get("excluded_labels"), [4], "manifest excluded labels")
            for key in ("case_id", "subject_id", "image_path", "label_path"):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise RuntimeError(
                        f"{path} line {line_number} has invalid {key}"
                    )
            rows.append(row)
    if not rows:
        raise RuntimeError(f"empty manifest: {path}")
    cases = {row["case_id"] for row in rows}
    subjects = {row["subject_id"] for row in rows}
    if len(cases) != len(rows):
        raise RuntimeError(f"duplicate case_id in {path}")
    return {
        "rows": len(rows),
        "cases": cases,
        "subjects": subjects,
    }


def validate_lora_reference(
    config: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    args = config.get("args")
    if not isinstance(args, dict):
        raise RuntimeError("LoRA train_config.args must be an object")
    for key, expected in EXPECTED_SHARED_ARGS.items():
        require_equal(args.get(key), expected, f"LoRA shared arg {key}")
    require_equal(args.get("encoder_qv_lora_rank"), 4, "LoRA reference rank")
    require_equal(
        args.get("output_dir"),
        (
            "workspace/BraTS/sammed3d_t1ce_wt/"
            "fv_lora_r4_aug_train972_val108_e50_s20260725"
        ),
        "LoRA reference output_dir",
    )
    require_equal(config.get("train_rows"), 972, "LoRA train rows")
    require_equal(config.get("val_rows"), 108, "LoRA validation rows")
    require_equal(config.get("effective_train_seed"), 20260725, "LoRA train seed")
    require_equal(config.get("train_loader_seed"), 20260726, "LoRA loader seed")
    require_equal(config.get("train_sampler_seed"), 20260725, "LoRA sampler seed")
    require_equal(config.get("effective_val_seed"), 20260723, "LoRA validation seed")
    require_equal(config.get("selection_metric"), "val_dice", "LoRA selection")
    require_equal(
        config.get("trainable_parameters"),
        7_731_760,
        "LoRA trainable parameters",
    )
    require_equal(
        config.get("total_parameters"),
        100_656_688,
        "LoRA total parameters",
    )

    optimization = protocol.get("optimization")
    preprocessing = protocol.get("preprocessing")
    source = protocol.get("source")
    if not all(isinstance(item, dict) for item in (optimization, preprocessing, source)):
        raise RuntimeError("LoRA protocol is missing optimization/preprocessing/source")
    image_encoder = optimization.get("image_encoder")
    require_equal(
        image_encoder,
        {
            "alpha": 8.0,
            "blocks": list(range(12)),
            "projections": ["q", "v"],
            "rank": 4,
            "update": "qv_lora",
        },
        "LoRA image-encoder protocol",
    )
    require_equal(
        preprocessing.get("geometry", {}).get("name"),
        "whole_resize",
        "LoRA geometry",
    )
    require_equal(
        preprocessing.get("augmentation", {}).get("name"),
        "affine_elastic_noise",
        "LoRA augmentation",
    )
    require_equal(
        preprocessing.get("augmentation", {}).get("validation_augmentation"),
        False,
        "LoRA validation augmentation",
    )
    require_equal(
        source.get("base_checkpoint_sha256"),
        EXPECTED_HASHES["base_checkpoint"],
        "LoRA base checkpoint binding",
    )
    require_equal(
        source.get("train_manifest_sha256"),
        EXPECTED_HASHES["train_manifest"],
        "LoRA train manifest binding",
    )
    require_equal(
        source.get("validation_manifest_sha256"),
        EXPECTED_HASHES["validation_manifest"],
        "LoRA validation manifest binding",
    )


def run_preflight(prereg_path: Path, root: Path = ROOT) -> dict[str, Any]:
    prereg_path = prereg_path.resolve()
    prereg = load_json(prereg_path, "rank-0 preregistration")
    require_equal(
        prereg.get("schema"),
        "aflow.sammed3d.t1ce_wt.fv_rank0_matched_aug_prereg",
        "prereg schema",
    )
    require_equal(prereg.get("schema_version"), 1, "prereg schema version")
    require_equal(
        prereg.get("status"),
        "frozen_before_full_training",
        "prereg status",
    )

    evidence = prereg.get("evidence_boundary")
    if not isinstance(evidence, dict):
        raise RuntimeError("prereg evidence_boundary must be an object")
    require_equal(evidence.get("development_only"), True, "development-only boundary")
    require_equal(evidence.get("train_cases"), 972, "prereg train cases")
    require_equal(evidence.get("validation_cases"), 108, "prereg validation cases")
    require_equal(
        evidence.get("external_holdout_access"),
        False,
        "external holdout access boundary",
    )
    for key in ("threshold_search", "tta", "morphology", "postprocess_search"):
        require_equal(evidence.get(key), False, f"prereg {key}")

    data = prereg.get("data")
    initialization = prereg.get("initialization")
    source = prereg.get("source")
    reference = prereg.get("matched_lora_reference")
    for value, label in (
        (data, "data"),
        (initialization, "initialization"),
        (source, "source"),
        (reference, "matched_lora_reference"),
    ):
        if not isinstance(value, dict):
            raise RuntimeError(f"prereg {label} must be an object")

    base_path = validate_bound_hash(
        root,
        initialization.get("base_checkpoint"),
        "base checkpoint",
        EXPECTED_HASHES["base_checkpoint"],
    )
    train_path = validate_bound_hash(
        root,
        data.get("train_manifest"),
        "train manifest",
        EXPECTED_HASHES["train_manifest"],
    )
    validation_path = validate_bound_hash(
        root,
        data.get("validation_manifest"),
        "validation manifest",
        EXPECTED_HASHES["validation_manifest"],
    )
    for key in (
        "trainer",
        "generic_trainer",
        "preprocessing_module",
        "peft_module",
        "engine_module",
    ):
        validate_bound_hash(root, source.get(key), key, EXPECTED_HASHES[key])

    train = read_manifest(train_path, "train")
    validation = read_manifest(validation_path, "validation")
    require_equal(train["rows"], 972, "train manifest rows")
    require_equal(len(train["subjects"]), 444, "train manifest subjects")
    require_equal(validation["rows"], 108, "validation manifest rows")
    require_equal(len(validation["subjects"]), 50, "validation manifest subjects")
    require_equal(
        len(train["cases"].intersection(validation["cases"])),
        0,
        "train/validation case overlap",
    )
    require_equal(
        len(train["subjects"].intersection(validation["subjects"])),
        0,
        "train/validation subject overlap",
    )

    lora_config_path = validate_bound_hash(
        root,
        reference.get("train_config"),
        "lora_train_config",
        EXPECTED_HASHES["lora_train_config"],
    )
    lora_protocol_path = validate_bound_hash(
        root,
        reference.get("training_protocol"),
        "lora_training_protocol",
        EXPECTED_HASHES["lora_training_protocol"],
    )
    validate_lora_reference(
        load_json(lora_config_path, "LoRA train config"),
        load_json(lora_protocol_path, "LoRA training protocol"),
    )

    rank0 = prereg.get("rank0_arm")
    optimization = prereg.get("optimization")
    preprocessing = prereg.get("preprocessing")
    if not all(isinstance(item, dict) for item in (rank0, optimization, preprocessing)):
        raise RuntimeError("prereg rank0/optimization/preprocessing must be objects")
    require_equal(rank0.get("encoder_qv_lora_rank"), 0, "rank-0 adapter rank")
    require_equal(rank0.get("image_encoder_update"), "frozen", "rank-0 encoder update")
    require_equal(rank0.get("prompt_encoder_trainable"), True, "rank-0 prompt encoder")
    require_equal(rank0.get("mask_decoder_trainable"), True, "rank-0 mask decoder")
    require_equal(rank0.get("full_image_encoder_finetuning"), False, "full encoder ban")
    require_equal(rank0.get("expected_trainable_parameters"), 7_584_304, "rank-0 trainable count")
    require_equal(rank0.get("expected_total_parameters"), 100_509_232, "rank-0 total count")
    require_equal(preprocessing.get("spatial_preprocessing"), "whole_resize", "rank-0 geometry")
    require_equal(preprocessing.get("train_augmentation"), "affine_elastic_noise", "rank-0 augmentation")
    require_equal(preprocessing.get("validation_augmentation"), False, "rank-0 validation augmentation")
    require_equal(optimization.get("epochs"), 50, "rank-0 epochs")
    require_equal(optimization.get("learning_rate"), 0.0001, "rank-0 learning rate")
    require_equal(optimization.get("weight_decay"), 0.1, "rank-0 weight decay")
    require_equal(optimization.get("batch_size"), 1, "rank-0 batch size")
    require_equal(optimization.get("gradient_accumulation_steps"), 2, "rank-0 accumulation")
    require_equal(optimization.get("num_workers"), 2, "rank-0 workers")
    require_equal(optimization.get("amp"), "bf16", "rank-0 AMP")
    require_equal(optimization.get("train_seed"), 20260725, "rank-0 train seed")
    require_equal(optimization.get("validation_click_seed"), 20260723, "rank-0 validation seed")
    require_equal(optimization.get("checkpoint_selection"), "maximum validation val_dice", "rank-0 selection")

    output_dir = Path(str(prereg.get("output_dir")))
    require_equal(
        output_dir.as_posix(),
        (
            "workspace/BraTS/sammed3d_t1ce_wt/"
            "fv_rank0_matched_aug_train972_val108_e50_s20260725"
        ),
        "rank-0 output directory",
    )
    return {
        "schema": SCHEMA,
        "safe_to_launch": True,
        "training_started": False,
        "preregistration": str(prereg_path),
        "base_checkpoint": {
            "path": str(base_path),
            "sha256": EXPECTED_HASHES["base_checkpoint"],
        },
        "development_data": {
            "train_cases": train["rows"],
            "train_subjects": len(train["subjects"]),
            "validation_cases": validation["rows"],
            "validation_subjects": len(validation["subjects"]),
            "case_overlap": 0,
            "subject_overlap": 0,
        },
        "matched_difference": {
            "lora_rank": 4,
            "rank0_rank": 0,
            "all_other_preregistered_training_fields": "identical",
        },
        "output_dir": output_dir.as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    args = parser.parse_args()
    print(json.dumps(run_preflight(args.prereg), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
