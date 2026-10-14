#!/usr/bin/env python3
"""Evaluate fixed historical/EMA168 logit blends on validation108."""
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
    dice_per_case,
    read_case_ids,
    sha256_file,
    summarize_cases,
    write_json_atomic,
)
from scripts.train_t1ce_wt_plain_ema_full_cohort import (  # noqa: E402
    CHECKPOINT_SCHEMA as EMA_CHECKPOINT_SCHEMA,
)
from scripts.train_t1ce_wt_plain_fixed_epoch import (  # noqa: E402
    CHECKPOINT_SCHEMA as HISTORICAL_CHECKPOINT_SCHEMA,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_logit_ensemble_validation"
VALIDATION_MANIFEST_SHA256 = "f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb"
HISTORICAL_SHA256 = "f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1"
EMA168_SHA256 = "4ea97dacb9a538993e1c16593bc56b03ce6dc3df2f457c1f0426439c6252197e"
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
CANDIDATE_WEIGHTS = (0.25, 0.5, 0.75)


def blend_logits(historical: torch.Tensor, ema: torch.Tensor, ema_weight: float):
    if historical.shape != ema.shape:
        raise ValueError("ensemble logits have different shapes")
    if not 0.0 <= ema_weight <= 1.0:
        raise ValueError("EMA ensemble weight must be in [0,1]")
    return historical * (1.0 - ema_weight) + ema * ema_weight


def point_from_logits(logits: torch.Tensor) -> torch.Tensor:
    flat = torch.argmax(logits[:, 0].flatten(1), dim=1)
    height, width = logits.shape[-2:]
    z = torch.div(flat, height * width, rounding_mode="floor")
    remainder = flat % (height * width)
    y = torch.div(remainder, width, rounding_mode="floor")
    x = remainder % width
    return torch.stack((z, y, x), dim=1)


def _model(checkpoint: dict[str, Any], device: torch.device):
    architecture = checkpoint["architecture"]
    model = AutomaticPromptLocator(
        center_aware=False,
        init_filters=int(architecture["init_filters"]),
        dropout=float(architecture["dropout"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


@torch.inference_mode()
def run(args: argparse.Namespace):
    if sha256_file(args.validation_manifest) != VALIDATION_MANIFEST_SHA256:
        raise ValueError("validation manifest SHA-256 mismatch")
    if sha256_file(args.historical_checkpoint) != HISTORICAL_SHA256:
        raise ValueError("historical checkpoint SHA-256 mismatch")
    if sha256_file(args.ema_checkpoint) != EMA168_SHA256:
        raise ValueError("EMA168 checkpoint SHA-256 mismatch")
    if args.output.exists() or args.output.with_suffix(".started.json").exists():
        raise FileExistsError("refusing repeated ensemble validation")
    historical_checkpoint = torch.load(
        args.historical_checkpoint, map_location="cpu", weights_only=False
    )
    ema_checkpoint = torch.load(
        args.ema_checkpoint, map_location="cpu", weights_only=False
    )
    if historical_checkpoint.get("schema") != HISTORICAL_CHECKPOINT_SCHEMA:
        raise ValueError("unexpected historical checkpoint schema")
    if historical_checkpoint.get("epoch") != 168:
        raise ValueError("historical checkpoint is not epoch168")
    if ema_checkpoint.get("schema") != EMA_CHECKPOINT_SCHEMA:
        raise ValueError("unexpected EMA checkpoint schema")
    if ema_checkpoint.get("kind") != "ema" or ema_checkpoint.get("epoch") != 168:
        raise ValueError("EMA checkpoint is not fixed EMA168")
    for checkpoint in (historical_checkpoint, ema_checkpoint):
        if any(
            checkpoint.get(key) is not False
            for key in ("validation_access", "outer_selector_access", "confirmation_access", "test_access")
        ):
            raise ValueError("checkpoint declares prior evaluation access")
    case_ids = read_case_ids(args.validation_manifest)
    assert case_ids is not None
    data = BrainCropDataset(
        args.cache, expected_split="validation", case_ids=case_ids
    )
    if len(data) != 108 or len({case.rsplit("-", 1)[0] for case in data.case_ids}) != 50:
        raise ValueError("unexpected validation case/subject count")
    if data.index.get("manifest_sha256") != VALIDATION_MANIFEST_SHA256:
        raise ValueError("validation cache binding mismatch")
    write_json_atomic(args.output.with_suffix(".started.json"), {
        "schema": SCHEMA + ".started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_checkpoint_sha256": HISTORICAL_SHA256,
        "ema_checkpoint_sha256": EMA168_SHA256,
        "weights": list(WEIGHTS),
        "candidate_weights": list(CANDIDATE_WEIGHTS),
        "threshold": 0.0,
        "test_access": False
    })
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ensemble validation requires CUDA")
    torch.cuda.set_device(device)
    historical_model = _model(historical_checkpoint, device)
    ema_model = _model(ema_checkpoint, device)
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0
    )
    cases: dict[float, list[dict[str, Any]]] = {weight: [] for weight in WEIGHTS}
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            historical_logits = historical_model(image).segmentation_logits.float()
            ema_logits = ema_model(image).segmentation_logits.float()
        for weight in WEIGHTS:
            logits = blend_logits(historical_logits, ema_logits, weight)
            dsc = dice_per_case(logits, label)
            points = point_from_logits(logits)
            for index, case_id in enumerate(batch["case_id"]):
                z, y, x = (int(value) for value in points[index].tolist())
                cases[weight].append({
                    "case_id": str(case_id),
                    "subject_id": str(batch["subject_id"][index]),
                    "locator_dsc": float(dsc[index]),
                    "automatic_point": [z, y, x],
                    "automatic_point_hits_gt": bool(label[index, 0, z, y, x] > 0.5)
                })
    summaries = {str(weight): summarize_cases(cases[weight]) for weight in WEIGHTS}
    baseline = summaries["0.0"]
    comparisons = {
        str(weight): compare(
            baseline,
            summaries[str(weight)],
            seed=20260839 + index
        )
        for index, weight in enumerate(CANDIDATE_WEIGHTS)
    }
    selected_weight = max(
        CANDIDATE_WEIGHTS,
        key=lambda weight: (
            summaries[str(weight)]["locator_dsc_mean"],
            summaries[str(weight)]["locator_subject_dsc_mean"]
        )
    )
    selected = comparisons[str(selected_weight)]
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "historical fixed validation108/50 subjects",
        "method": "linear model-grid logit blend: (1-w)*historical + w*EMA168",
        "weights": list(WEIGHTS),
        "candidate_weights": list(CANDIDATE_WEIGHTS),
        "selection": "maximum validation scan Dice, subject Dice tie-break",
        "selected_weight": selected_weight,
        "summaries": summaries,
        "comparisons_to_historical": comparisons,
        "selected_comparison": selected,
        "ensemble_candidate_authorized_for_test": selected["all_gates_passed"],
        "checkpoints": {
            "historical": {
                "path": str(args.historical_checkpoint.resolve()),
                "sha256": HISTORICAL_SHA256
            },
            "ema168": {
                "path": str(args.ema_checkpoint.resolve()),
                "sha256": EMA168_SHA256
            }
        },
        "data": {
            "manifest": str(args.validation_manifest.resolve()),
            "manifest_sha256": VALIDATION_MANIFEST_SHA256,
            "cache": str(args.cache.resolve()),
            "cache_index_sha256": sha256_file(args.cache / "index.json"),
            "cases": 108,
            "subjects": 50
        },
        "threshold": 0.0,
        "tta": False,
        "postprocessing": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False
    }
    write_json_atomic(args.output, report)
    print(json.dumps({
        "selected_weight": selected_weight,
        "summaries": {
            weight: {
                "scan": summary["locator_dsc_mean"],
                "subject": summary["locator_subject_dsc_mean"],
                "point_hit": summary["automatic_point_hit_rate"]
            }
            for weight, summary in summaries.items()
        },
        "selected_comparison": selected,
        "ensemble_candidate_authorized_for_test": report[
            "ensemble_candidate_authorized_for_test"
        ]
    }, indent=2, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--historical-checkpoint", type=Path, required=True)
    parser.add_argument("--ema-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
