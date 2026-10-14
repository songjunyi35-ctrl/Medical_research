#!/usr/bin/env python3
"""Evaluate fixed raw/EMA epoch168/200 checkpoints on historical validation108."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
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
from scripts.train_t1ce_wt_plain_ema_full_cohort import CHECKPOINT_SCHEMA  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_ema_validation"
VALIDATION_MANIFEST_SHA256 = "f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb"
BOOTSTRAP_SEED = 20260834
BOOTSTRAP_SAMPLES = 10_000


def _subjects(per_case: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in per_case:
        grouped[str(item["subject_id"])].append(float(item["locator_dsc"]))
    return {subject: statistics.fmean(values) for subject, values in grouped.items()}


def compare(
    raw: dict[str, Any],
    ema: dict[str, Any],
    *,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    raw_cases = {str(item["case_id"]): item for item in raw["per_case"]}
    ema_cases = {str(item["case_id"]): item for item in ema["per_case"]}
    if raw_cases.keys() != ema_cases.keys():
        raise ValueError("raw/EMA case sets differ")
    raw_subjects = _subjects(list(raw_cases.values()))
    ema_subjects = _subjects(list(ema_cases.values()))
    if raw_subjects.keys() != ema_subjects.keys():
        raise ValueError("raw/EMA subject sets differ")
    case_delta = np.asarray([
        float(ema_cases[key]["locator_dsc"]) - float(raw_cases[key]["locator_dsc"])
        for key in sorted(raw_cases)
    ])
    subject_delta = np.asarray([
        ema_subjects[key] - raw_subjects[key] for key in sorted(raw_subjects)
    ])
    rng = np.random.default_rng(seed)
    case_draws = rng.integers(0, len(case_delta), size=(samples, len(case_delta)))
    subject_draws = rng.integers(
        0, len(subject_delta), size=(samples, len(subject_delta))
    )
    point_delta = float(
        ema["automatic_point_hit_rate"] - raw["automatic_point_hit_rate"]
    )
    scan_delta = float(case_delta.mean())
    subject_delta_mean = float(subject_delta.mean())
    gates = {
        "scan_gain_positive": scan_delta > 0.0,
        "subject_gain_positive": subject_delta_mean > 0.0,
        "point_hit_drop_at_most_0.01": point_delta >= -0.01,
        "paired_subject_ci_lower_at_least_minus_0.003": float(
            np.quantile(subject_delta[subject_draws].mean(axis=1), 0.025)
        ) >= -0.003
    }
    return {
        "raw": {
            "scan": float(raw["locator_dsc_mean"]),
            "subject": float(raw["locator_subject_dsc_mean"]),
            "point_hit": float(raw["automatic_point_hit_rate"])
        },
        "ema": {
            "scan": float(ema["locator_dsc_mean"]),
            "subject": float(ema["locator_subject_dsc_mean"]),
            "point_hit": float(ema["automatic_point_hit_rate"])
        },
        "ema_minus_raw": {
            "scan": scan_delta,
            "subject": subject_delta_mean,
            "point_hit": point_delta,
            "scan_95ci": [
                float(value) for value in np.quantile(
                    case_delta[case_draws].mean(axis=1), [0.025, 0.975]
                )
            ],
            "subject_95ci": [
                float(value) for value in np.quantile(
                    subject_delta[subject_draws].mean(axis=1), [0.025, 0.975]
                )
            ],
            "cases_improved": int(np.sum(case_delta > 0)),
            "cases_tied": int(np.sum(case_delta == 0)),
            "cases_worse": int(np.sum(case_delta < 0)),
            "subjects_improved": int(np.sum(subject_delta > 0)),
            "subjects_tied": int(np.sum(subject_delta == 0)),
            "subjects_worse": int(np.sum(subject_delta < 0)),
            "bootstrap_seed": seed,
            "bootstrap_samples": samples
        },
        "gates": gates,
        "all_gates_passed": all(gates.values())
    }


def load_model(path: Path, expected_kind: str, expected_epoch: int, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"unexpected EMA checkpoint schema: {path}")
    if checkpoint.get("kind") != expected_kind or checkpoint.get("epoch") != expected_epoch:
        raise ValueError(f"unexpected kind/epoch: {path}")
    if checkpoint.get("resolution") != 128:
        raise ValueError("EMA validation accepts only true-128 checkpoints")
    if any(
        checkpoint.get(key) is not False
        for key in ("validation_access", "outer_selector_access", "confirmation_access", "test_access")
    ):
        raise ValueError("checkpoint declares prior evaluation access")
    architecture = checkpoint["architecture"]
    model = AutomaticPromptLocator(
        center_aware=False,
        init_filters=int(architecture["init_filters"]),
        dropout=float(architecture["dropout"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint


def run(args: argparse.Namespace) -> dict[str, Any]:
    if sha256_file(args.validation_manifest) != VALIDATION_MANIFEST_SHA256:
        raise ValueError("historical validation manifest SHA-256 mismatch")
    if args.output.exists() or args.output.with_suffix(".started.json").exists():
        raise FileExistsError("refusing repeated EMA validation")
    case_ids = read_case_ids(args.validation_manifest)
    assert case_ids is not None
    data = BrainCropDataset(
        args.cache,
        expected_split="validation",
        case_ids=case_ids
    )
    if len(data) != 108 or len({item.rsplit("-", 1)[0] for item in data.case_ids}) != 45:
        raise ValueError("unexpected historical validation case/subject count")
    if data.index.get("manifest_sha256") != VALIDATION_MANIFEST_SHA256:
        raise ValueError("validation cache manifest binding mismatch")
    for filename, field in (("images.f32", "images_sha256"), ("labels.u8", "labels_sha256")):
        if sha256_file(args.cache / filename) != data.index[field]:
            raise ValueError(f"validation cache {filename} SHA-256 mismatch")
    write_json_atomic(args.output.with_suffix(".started.json"), {
        "schema": SCHEMA + ".started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_manifest_sha256": VALIDATION_MANIFEST_SHA256,
        "checkpoint_selection_uses_validation": False,
        "threshold": 0.0,
        "tta": False,
        "postprocessing": False,
        "test_access": False
    })
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("EMA validation requires CUDA")
    torch.cuda.set_device(device)
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0
    )
    specs = {
        "raw_168": (args.raw168, "raw", 168),
        "ema_168": (args.ema168, "ema", 168),
        "raw_200": (args.raw200, "raw", 200),
        "ema_200": (args.ema200, "ema", 200)
    }
    summaries: dict[str, Any] = {}
    checkpoint_metadata: dict[str, Any] = {}
    for name, (path, kind, epoch) in specs.items():
        model, checkpoint = load_model(path, kind, epoch, device)
        summaries[name] = validate(model, loader, device)
        checkpoint_metadata[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "kind": kind,
            "epoch": epoch,
            "ema": checkpoint["ema"]
        }
        del model
        torch.cuda.empty_cache()
    primary = compare(summaries["raw_200"], summaries["ema_200"])
    secondary = compare(summaries["raw_168"], summaries["ema_168"])
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "historical fixed validation108; development only",
        "not_external_confirmation": True,
        "protocol": {
            "resolution": 128,
            "threshold": "raw logit > 0",
            "tta": False,
            "postprocessing": False,
            "ema_decay": 0.999,
            "ema_start_after_epoch": 100,
            "primary": "EMA epoch200 minus raw epoch200",
            "secondary": "EMA epoch168 minus raw epoch168"
        },
        "data": {
            "manifest": str(args.validation_manifest.resolve()),
            "manifest_sha256": VALIDATION_MANIFEST_SHA256,
            "cache": str(args.cache.resolve()),
            "cache_index_sha256": sha256_file(args.cache / "index.json"),
            "cases": 108,
            "subjects": 45
        },
        "checkpoints": checkpoint_metadata,
        "summaries": summaries,
        "primary_epoch200": primary,
        "secondary_epoch168": secondary,
        "ema_candidate_authorized_for_test": primary["all_gates_passed"],
        "selected_candidate": "ema_200" if primary["all_gates_passed"] else None,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False
    }
    write_json_atomic(args.output, report)
    print(json.dumps({
        "primary": primary,
        "secondary": secondary,
        "ema_candidate_authorized_for_test": report["ema_candidate_authorized_for_test"]
    }, indent=2, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--raw168", type=Path, required=True)
    parser.add_argument("--ema168", type=Path, required=True)
    parser.add_argument("--raw200", type=Path, required=True)
    parser.add_argument("--ema200", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
