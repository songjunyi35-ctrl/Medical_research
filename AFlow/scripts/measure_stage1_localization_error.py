#!/usr/bin/env python
"""Measure how well the braincrop arm LOCALIZES the lesion, on validation only.

Why this exists
---------------
A two-stage coarse-to-fine cascade crops stage 2's field of view around stage 1's
prediction. Training stage 2 on crops taken around stage 1's predictions ON THE
TRAINING SET leaks: stage 1 was fitted to those cases, so its training-set
localization is far better than what stage 2 will meet at inference, and stage 2
would learn to trust a precision it never gets.

nnU-Net solves this with cross-validated stage-1 predictions (5x the training
cost). This project instead crops stage 2 around the GROUND TRUTH with jitter
drawn from stage 1's REAL localization error, measured here on the 108 validation
cases -- which stage 1 never trained on. No stage-1 training-set prediction is
ever used, so the leak is structurally impossible rather than merely unlikely.

What is measured, per case, in native 1mm space:
  * centroid offset  : predicted lesion centroid minus GT centroid, in mm
  * extent ratio     : predicted bbox extent / GT bbox extent, per axis
  * containment      : does a box placed on the prediction actually contain all
                       of the GT, as a function of the margin applied

Validation split only. No test data is read.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_sammed3d_t1ce_wt_checkpoint import (  # noqa: E402
    case_seed, load_model, predict_click_curve, read_rows,
)

MANIFEST = ROOT / "data/datasets/brats2024_sammed3d_t1ce_wt_inner90_10_seed20260723_v1/validation_t1ce_wt.jsonl"
MSHA = "fc750ab15266b8a53c6d9a7ff62bd043b10cc21733e9b68354a2a989fd1f488c"
CACHE = Path("/tmp/aflow_braincrop_cache/t1ce_wt_n4_braincrop_128/validation")
CKPT = ROOT / ("workspace/BraTS/sammed3d_t1ce_wt/"
               "sammed3d_braincrop_fullunfreeze_ddp2_train972_val108_e200_s20260806/"
               "sam_model_val_dice_best.pth")
CKSHA = "87dad82c67151128c7e99d64ae87f37c68547c769d2e610a56d432b554e0732a"
SEED = 20260723
OUT = ROOT / "workspace/BraTS/sammed3d_t1ce_wt/STAGE1_LOCALIZATION_ERROR_20260807.json"


def bbox_and_centroid(mask: torch.Tensor):
    idx = torch.nonzero(mask, as_tuple=False)
    if idx.numel() == 0:
        return None
    lo = idx.min(dim=0).values.tolist()
    hi = (idx.max(dim=0).values + 1).tolist()
    centroid = idx.float().mean(dim=0).tolist()
    return {"lo": lo, "hi": hi, "extent": [hi[i] - lo[i] for i in range(3)], "centroid": centroid}


def main() -> None:
    rows, _ = read_rows(MANIFEST, expected_sha256=MSHA,
                        expected_split="validation", max_cases=None)
    index = json.loads((CACHE / "index.json").read_text())
    if index["test_access"] is not False:
        raise SystemExit("refusing: cache declares test access")
    pos = {c: i for i, c in enumerate(index["case_ids"])}
    geom = index["crop_geometry"]
    n = index["count"]
    images = np.memmap(CACHE / "images.f32", dtype=np.float32, mode="r", shape=(n, 128, 128, 128))
    labels = np.memmap(CACHE / "labels.u8", dtype=np.uint8, mode="r", shape=(n, 128, 128, 128))

    device = torch.device("cuda")
    model, sha, _ = load_model(CKPT, device=device, expected_sha256=CKSHA)

    cases = []
    for k, row in enumerate(rows, 1):
        cid = row["case_id"]
        slot = pos[cid]
        image = torch.from_numpy(np.array(images[slot], dtype=np.float32))[None]
        target128 = torch.from_numpy(np.array(labels[slot])).bool()
        preds, _ = predict_click_curve(
            model, image, target128, device=device,
            generator=np.random.default_rng(case_seed(SEED, cid)))
        p128 = preds[3].bool().cpu()

        g = geom[cid]
        origin, extent, native_shape = g["origin"], g["extent"], g["native_shape"]
        # 128^3 crop-grid prediction -> the crop's own native extent -> full native array
        up = F.interpolate(p128[None, None].float(), size=tuple(extent),
                           mode="nearest-exact")[0, 0].bool()
        pred_native = torch.zeros(tuple(native_shape), dtype=torch.bool)
        pred_native[origin[0]:origin[0] + extent[0],
                    origin[1]:origin[1] + extent[1],
                    origin[2]:origin[2] + extent[2]] = up

        seg = nib.as_closest_canonical(nib.load(row["label_path"], mmap=True))
        gt_native = torch.from_numpy(
            np.isin(np.asanyarray(seg.dataobj).astype(np.uint8), [1, 2, 3])).bool()
        zooms = [float(z) for z in seg.header.get_zooms()[:3]]

        pb, gb = bbox_and_centroid(pred_native), bbox_and_centroid(gt_native)
        if pb is None or gb is None:
            cases.append({"case_id": cid, "empty_prediction": pb is None})
            continue
        offset_mm = [(pb["centroid"][i] - gb["centroid"][i]) * zooms[i] for i in range(3)]
        cases.append({
            "case_id": cid,
            "zooms": zooms,
            "centroid_offset_mm": offset_mm,
            "centroid_offset_norm_mm": float(np.linalg.norm(offset_mm)),
            "gt_extent_mm": [gb["extent"][i] * zooms[i] for i in range(3)],
            "pred_extent_mm": [pb["extent"][i] * zooms[i] for i in range(3)],
            "extent_ratio": [pb["extent"][i] / gb["extent"][i] for i in range(3)],
            "gt_lo": gb["lo"], "gt_hi": gb["hi"],
            "pred_lo": pb["lo"], "pred_hi": pb["hi"],
        })
        if k % 27 == 0:
            print(f"{k}/{len(rows)}", flush=True)

    good = [c for c in cases if "centroid_offset_mm" in c]

    # Containment: place a box on the PREDICTION, expanded by `margin`, and ask
    # whether it swallows the whole GT. This is the number that decides whether a
    # two-stage crop can lose lesion voxels it can never recover.
    containment = {}
    for margin in (1.0, 1.2, 1.4, 1.6, 2.0):
        held = 0
        for c in good:
            pc = [(c["pred_lo"][i] + c["pred_hi"][i]) / 2 for i in range(3)]
            half = [max(1.0, (c["pred_hi"][i] - c["pred_lo"][i]) / 2 * margin) for i in range(3)]
            ok = all(pc[i] - half[i] <= c["gt_lo"][i] and pc[i] + half[i] >= c["gt_hi"][i]
                     for i in range(3))
            held += int(ok)
        containment[f"margin_{margin}"] = round(held / len(good), 4)

    off = np.array([c["centroid_offset_norm_mm"] for c in good])
    ratios = np.array([c["extent_ratio"] for c in good])
    report = {
        "schema": "aflow.sammed3d.stage1_localization_error",
        "schema_version": 1,
        "created_utc": "2026-08-07",
        "purpose": ("calibrate the jitter distribution for two-stage stage-2 training crops "
                    "without ever using a stage-1 prediction on the training split"),
        "stage1_checkpoint_sha256": sha,
        "split": "validation 108 (stage 1 never trained on these)",
        "cases_with_nonempty_prediction": len(good),
        "cases_with_empty_prediction": len(cases) - len(good),
        "centroid_offset_mm": {
            "mean": float(off.mean()), "median": float(np.median(off)),
            "p75": float(np.percentile(off, 75)), "p90": float(np.percentile(off, 90)),
            "p95": float(np.percentile(off, 95)), "max": float(off.max()),
        },
        "extent_ratio_pred_over_gt": {
            "median_per_axis": [float(x) for x in np.median(ratios, axis=0)],
            "p10_per_axis": [float(x) for x in np.percentile(ratios, 10, axis=0)],
            "p90_per_axis": [float(x) for x in np.percentile(ratios, 90, axis=0)],
        },
        "gt_containment_rate_by_box_margin": containment,
        "test_access": False,
        "cases": cases,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\nstage-1 localization error, validation n={len(good)}"
          f" ({len(cases)-len(good)} empty predictions)")
    print(f"  centroid offset (mm): median {np.median(off):.1f}  p75 {np.percentile(off,75):.1f}"
          f"  p90 {np.percentile(off,90):.1f}  p95 {np.percentile(off,95):.1f}  max {off.max():.1f}")
    print(f"  extent ratio pred/GT, median per axis: "
          f"{[round(float(x),3) for x in np.median(ratios,axis=0)]}")
    print("\n  GT fully contained in a box placed on the prediction:")
    for k, v in containment.items():
        print(f"    margin {k.split('_')[1]:<4} -> {v*100:5.1f}%")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
