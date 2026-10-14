#!/usr/bin/env python
"""Independent strict-reload evaluation sourced from the N4 preprocessing cache.

Why a separate evaluator
------------------------
`evaluate_sammed3d_t1ce_wt_checkpoint.py` recomputes preprocessing on the fly and
has no N4 option. The DPD-SAM-aligned arm trains on N4-corrected inputs, so
evaluating it through the un-corrected path would introduce a train/eval
preprocessing mismatch and systematically depress the readout.

Reading the model input straight from the same cache the arm trained on makes
train/eval preprocessing parity exact by construction, not by re-derivation.

Everything else is imported from the existing evaluator so the click protocol,
metrics and subject-balanced aggregation are the same code, not a
reimplementation: `case_seed`, `predict_click_curve`, `overlap_metrics`,
`summarize`, `load_model`, `read_rows`. That file's bytes are not modified.

The sealed 270-case test is reachable only with BOTH `--expected-split test` and
`--i-have-frozen-the-configuration`, and only after the frozen configuration is
already on disk. Default behaviour remains validation-only.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d import engine  # noqa: E402
from sammed3d.t1ce_wt_cached import MODEL_SHAPE  # noqa: E402
import scripts.evaluate_sammed3d_t1ce_wt_checkpoint as _frozen_eval  # noqa: E402
from scripts.evaluate_sammed3d_t1ce_wt_checkpoint import (  # noqa: E402
    case_seed,
    load_model,
    overlap_metrics,
    predict_click_curve,
    read_rows,
    sha256_file,
    summarize,
    write_json_atomic,
)

SCHEMA = "aflow.sammed3d.t1ce_wt_cached_validation_evaluation"
SCHEMA_VERSION = 1


def _geometry_description(index: dict) -> str:
    """Describe geometry from the cache, never from a hardcoded assumption."""
    if index["preprocessing"]["spatial"] == "gt_crop":
        return ("SAM-Med3D native geometry: ToCanonical + label-centred "
                "CropOrPad 128^3 at native resolution. USES GROUND TRUTH TO PLACE "
                "THE FIELD OF VIEW, i.e. oracle localization; not comparable with "
                "automatic segmentation methods.")
    return "canonical whole-volume resize to 128^3; no GT localization"


def open_cache(cache_dir: Path, expected_split: str) -> tuple[dict, np.memmap, np.memmap]:
    index = json.loads((cache_dir / "index.json").read_text())
    # The cache must declare the same split the caller asked for, so a test cache
    # can never be silently consumed as if it were validation, or vice versa.
    if bool(index["test_access"]) != (expected_split == "test"):
        raise RuntimeError(
            f"cache declares test_access={index['test_access']} but caller "
            f"requested split={expected_split}"
        )
    if index.get("split") not in (None, expected_split):
        raise RuntimeError(
            f"cache split {index.get('split')!r} != requested {expected_split!r}"
        )
    count = int(index["count"])
    images = np.memmap(cache_dir / "images.f32", dtype=np.float32, mode="r",
                       shape=(count, *MODEL_SHAPE))
    labels = np.memmap(cache_dir / "labels.u8", dtype=np.uint8, mode="r",
                       shape=(count, *MODEL_SHAPE))
    return index, images, labels


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    # predict_click_curve reads NUM_CLICKS as a module constant. Override it in
    # place rather than editing the frozen evaluator's bytes, and record the value
    # used in the report so the click budget is never implicit.
    _frozen_eval.NUM_CLICKS = int(args.num_clicks)

    if args.expected_split == "test" and not args.i_have_frozen_the_configuration:
        raise SystemExit(
            "Refusing to open the sealed test split. Test evaluation requires an "
            "explicit --i-have-frozen-the-configuration flag, and the frozen "
            "configuration must already be on disk."
        )
    rows, total_rows = read_rows(
        args.manifest.resolve(),
        expected_sha256=args.expected_manifest_sha256,
        expected_split=args.expected_split,
        max_cases=args.max_cases,
    )
    index, images, labels = open_cache(args.cache_dir.resolve(), args.expected_split)
    if index["manifest_sha256"] != args.expected_manifest_sha256:
        raise RuntimeError(
            f"cache manifest SHA {index['manifest_sha256']} != manifest "
            f"{args.expected_manifest_sha256}"
        )
    position = {cid: i for i, cid in enumerate(index["case_ids"])}
    missing = [r["case_id"] for r in rows if r["case_id"] not in position]
    if missing:
        raise RuntimeError(f"{len(missing)} manifest cases absent from cache")

    device = torch.device(args.device)
    model, checkpoint_sha256, adapter_config = load_model(
        args.checkpoint.resolve(),
        device=device,
        expected_sha256=args.expected_checkpoint_sha256,
    )

    cases: list[dict[str, Any]] = []
    start = time.time()
    for n, row in enumerate(rows, start=1):
        slot = position[row["case_id"]]
        image = torch.from_numpy(np.array(images[slot], dtype=np.float32))[None]
        target = torch.from_numpy(np.array(labels[slot])).bool()
        if tuple(image.shape) != (1, *MODEL_SHAPE) or tuple(target.shape) != MODEL_SHAPE:
            raise RuntimeError(f"{row['case_id']} cached tensors have wrong shape")
        if not bool(target.any()):
            raise RuntimeError(f"{row['case_id']} cached target is empty")
        predictions, clicks = predict_click_curve(
            model, image, target, device=device,
            generator=np.random.default_rng(case_seed(args.seed, row["case_id"])),
        )
        metrics = {
            str(k): overlap_metrics(pred, target) for k, pred in predictions.items()
        }
        cases.append({
            "case_id": row["case_id"],
            "subject_id": row["subject_id"],
            "geometry": {
                "spatial_preprocessing": index["preprocessing"]["spatial"],
                "uses_gt_for_localization": bool(
                    index["preprocessing"].get("uses_gt_for_localization",
                                               index["preprocessing"]["spatial"] == "gt_crop")),
                "model_shape": list(MODEL_SHAPE),
                "model_target_voxels": int(target.sum()),
                "source": "N4 preprocessing cache",
            },
            "clicks": clicks,
            "metrics": metrics,
        })
        print(f"[{n}/{len(rows)}] {row['case_id']} "
              f"c1={metrics['1']['dsc']:.4f} c2={metrics['2']['dsc']:.4f} "
              f"c3={metrics['3']['dsc']:.4f}", flush=True)

    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "input": "T1ce single channel",
            "target": "WT = raw labels {1,2,3}; RC label 4 excluded",
            "geometry": _geometry_description(index),
            "spatial_preprocessing": index["preprocessing"]["spatial"],
            "uses_gt_for_localization": bool(
                index["preprocessing"].get("uses_gt_for_localization",
                                           index["preprocessing"]["spatial"] == "gt_crop")),
            "normalization": index["preprocessing"]["normalization"],
            "bias_field_correction": index["preprocessing"]["n4_bias_field_correction"],
            "clicks": ("click1 uniform GT foreground; click2/3 uniform XOR error; "
                       "FN positive, FP negative"),
            "prompt_history": "all points accumulated",
            "mask_prompt": "previous low-resolution mask recurrently supplied",
            "threshold": "raw logit > 0",
            "num_clicks": int(args.num_clicks),
        },
        "data": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": args.expected_manifest_sha256,
            "expected_split": args.expected_split,
            "selected_cases": len(rows),
            "full_manifest_cases": total_rows,
            "cache_dir": str(args.cache_dir.resolve()),
            "cache_images_sha256": index["images_sha256"],
            "cache_labels_sha256": index["labels_sha256"],
        },
        "model": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "model_type": engine.MODEL_TYPE,
            "adapter_config": adapter_config,
        },
        "seed": args.seed,
        "summary": summarize(cases),
        "elapsed_seconds": time.time() - start,
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "test_access": args.expected_split == "test",
        "comparison_caveat": (
            "Reference LoRA/rank-0 values were produced without N4, i.e. on "
            "different model inputs. Treat any delta as descriptive context, not "
            "a controlled contrast."
        ),
        "cases": cases,
    }
    write_json_atomic(args.output.resolve(), report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--expected-manifest-sha256", required=True)
    p.add_argument("--expected-split", choices=("validation", "test"), default="validation")
    p.add_argument("--i-have-frozen-the-configuration", action="store_true",
                   help="required to open the sealed test split")
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--expected-checkpoint-sha256", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260723)
    p.add_argument("--num-clicks", type=int, default=3,
                   help="3 = this project's frozen protocol; 5 = SAM-Med3D official default")
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    report = evaluate(args)
    s = report["summary"]
    print(json.dumps({
        "scan_macro": {k: s["scan_macro"][k]["dsc_mean"] for k in s["scan_macro"]}
        if "scan_macro" in s else s,
    }, indent=2))


if __name__ == "__main__":
    main()
