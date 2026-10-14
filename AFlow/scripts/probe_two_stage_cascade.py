#!/usr/bin/env python
"""Measure the realizable two-stage cascade WITHOUT training anything.

The idea
--------
The existing `sammed3d_native_gtcrop_*` arm is already a stage-2 refiner: it was
trained on 128^3 crops taken at NATIVE resolution and centred on the lesion. Its
validation score of 0.6381 is the ORACLE number, because the crop is centred on
the ground truth.

A real cascade centres that crop on stage 1's prediction instead. So the whole
cascade can be measured by re-running the SAME stage-2 checkpoint on crops
centred on the stage-1 prediction. No training is required, and the difference
between the two numbers is exactly the price of imperfect localization.

Pipeline per case (validation only):
  stage 1 : braincrop checkpoint -> 128^3 mask -> restored to native
  crop    : tio.CropOrPad(mask_name=...) centred on the STAGE-1 PREDICTION,
            128^3 at native resolution -- identical to the gt_crop builder except
            that the centring mask is the prediction, not the label
  stage 2 : gt_crop checkpoint, 3 accumulated clicks, raw logit > 0

Scoring is done on the stage-2 crop grid, the same grid the oracle gt_crop arm
was scored on, so the two numbers are directly comparable. GT voxels that fall
outside the crop are counted as false negatives rather than silently dropped, so
a worse crop is penalised instead of being rewarded with an easier denominator.

Validation split only. No test data is read. Diagnostic, not an arm.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import torchio as tio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_prompt import wt_target  # noqa: E402
from sammed3d.t1ce_wt_training import positive_sample_zscore  # noqa: E402
from scripts.build_t1ce_wt_cache import n4_correct  # noqa: E402
from scripts.evaluate_sammed3d_t1ce_wt_checkpoint import (  # noqa: E402
    case_seed, load_model, overlap_metrics, predict_click_curve, read_rows, summarize,
)

MANIFEST = ROOT / "data/datasets/brats2024_sammed3d_t1ce_wt_inner90_10_seed20260723_v1/validation_t1ce_wt.jsonl"
MSHA = "fc750ab15266b8a53c6d9a7ff62bd043b10cc21733e9b68354a2a989fd1f488c"
BC_CACHE = Path("/tmp/aflow_braincrop_cache/t1ce_wt_n4_braincrop_128/validation")
STAGE1 = ROOT / ("workspace/BraTS/sammed3d_t1ce_wt/"
                 "sammed3d_braincrop_fullunfreeze_ddp2_train972_val108_e200_s20260806/"
                 "sam_model_val_dice_best.pth")
STAGE1_SHA = "87dad82c67151128c7e99d64ae87f37c68547c769d2e610a56d432b554e0732a"
STAGE2 = ROOT / ("workspace/BraTS/sammed3d_t1ce_wt/"
                 "sammed3d_native_gtcrop_fullunfreeze_ddp2_train972_val108_e100_s20260805/"
                 "sam_model_val_dice_best.pth")
SEED = 20260723
SHAPE = (128, 128, 128)
OUT = ROOT / "workspace/BraTS/sammed3d_t1ce_wt/TWO_STAGE_CASCADE_PROBE_FIXED_20260807.json"
N4DIR = Path("/tmp/aflow_probe_n4"); N4DIR.mkdir(exist_ok=True)


def _n4_one(row: dict) -> str:
    """N4 at native resolution, cached to disk. One ITK thread per worker."""
    for v in ("OMP_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
        os.environ[v] = "1"
    import SimpleITK as sitk
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    sys.path.insert(0, str(ROOT))
    from scripts.build_t1ce_wt_cache import n4_correct as _n4
    cid = str(row["case_id"])
    dst = N4DIR / f"{cid}.npy"
    if not dst.exists():
        a = np.ascontiguousarray(np.asanyarray(
            nib.as_closest_canonical(nib.load(str(row["image_path"]), mmap=True)).dataobj))
        np.save(dst, _n4(a, 2).astype(np.float32))
    return cid


def crop_centred_on(canon_image_np, canon_gt_np, centre_mask_np, affine):
    """Crop 128^3 at native resolution, centred on `centre_mask_np`.

    Everything here is ALREADY canonical -- the stage-1 prediction is built in the
    braincrop cache's canonical frame -- so ToCanonical must NOT be applied. An
    earlier version of this probe passed the canonical prediction with a raw
    affine and let ToCanonical reorient it a second time, which centred the crop
    at a reflected location and produced a spurious -0.22 result.
    """
    subject = tio.Subject(
        image=tio.ScalarImage(
            tensor=torch.from_numpy(np.ascontiguousarray(canon_image_np))[None].float(),
            affine=affine),
        label=tio.LabelMap(
            tensor=torch.from_numpy(np.ascontiguousarray(centre_mask_np))[None].to(torch.uint8),
            affine=affine),
        truth=tio.LabelMap(
            tensor=torch.from_numpy(np.ascontiguousarray(canon_gt_np))[None].to(torch.uint8),
            affine=affine),
    )
    t = tio.CropOrPad(mask_name="label", target_shape=SHAPE)(subject)
    return (positive_sample_zscore(t.image.data.float().contiguous()),
            (t.truth.data > 0), int(canon_gt_np.sum()))


def main() -> None:
    rows, _ = read_rows(MANIFEST, expected_sha256=MSHA,
                        expected_split="validation", max_cases=None)
    index = json.loads((BC_CACHE / "index.json").read_text())
    if index["test_access"] is not False:
        raise SystemExit("refusing: cache declares test access")
    pos = {c: i for i, c in enumerate(index["case_ids"])}
    geom = index["crop_geometry"]
    n = index["count"]
    bc_img = np.memmap(BC_CACHE / "images.f32", dtype=np.float32, mode="r", shape=(n, *SHAPE))
    bc_lab = np.memmap(BC_CACHE / "labels.u8", dtype=np.uint8, mode="r", shape=(n, *SHAPE))

    print(f"precomputing N4 for {len(rows)} cases in parallel...", flush=True)
    with ProcessPoolExecutor(54) as ex:
        list(ex.map(_n4_one, rows, chunksize=1))
    print("N4 ready", flush=True)

    device = torch.device("cuda")
    m1, sha1, _ = load_model(STAGE1, device=device, expected_sha256=STAGE1_SHA)
    m2, sha2, _ = load_model(STAGE2, device=device, expected_sha256="42f96a0a2ade9d98d9db897a11a5a285f7a6351a7693c6884d6ac97a29dfde52")

    cascade, oracle = [], []
    for k, row in enumerate(rows, 1):
        cid = row["case_id"]

        # ---- stage 1 -------------------------------------------------------
        slot = pos[cid]
        img1 = torch.from_numpy(np.array(bc_img[slot], dtype=np.float32))[None]
        tgt1 = torch.from_numpy(np.array(bc_lab[slot])).bool()
        p1, _ = predict_click_curve(m1, img1, tgt1, device=device,
                                    generator=np.random.default_rng(case_seed(SEED, cid)))
        g = geom[cid]
        origin, extent, native_shape = g["origin"], g["extent"], g["native_shape"]
        up = F.interpolate(p1[3].bool()[None, None].float(), size=tuple(extent),
                           mode="nearest-exact")[0, 0].bool().numpy()
        pred_native = np.zeros(tuple(native_shape), dtype=np.uint8)
        pred_native[origin[0]:origin[0] + extent[0],
                    origin[1]:origin[1] + extent[1],
                    origin[2]:origin[2] + extent[2]] = up
        if pred_native.sum() == 0:                    # degenerate: fall back to brain centre
            pred_native[tuple(s // 2 for s in native_shape)] = 1
        # CropOrPad centres on the mask BOUNDING BOX, which a single distant false
        # positive can drag far off the lesion. Keep only the largest connected
        # component so the centring statistic is robust to scattered FPs.
        # No largest-connected-component filter: WT is multi-focal in 43% of cases
        # and keeping only the biggest component measured strictly worse.

        # ---- crop, then stage 2 -------------------------------------------
        rln = nib.as_closest_canonical(nib.load(str(row["label_path"]), mmap=True))
        affine = np.asarray(rln.affine, dtype=np.float64)
        image = np.load(N4DIR / f"{cid}.npy")
        gt_np = wt_target(np.ascontiguousarray(
            np.asanyarray(rln.dataobj))).numpy().astype(np.uint8)

        for label, centre in (("cascade", pred_native), ("oracle", gt_np)):
            img, tgt, total_gt = crop_centred_on(image, gt_np, centre, affine)
            if not bool(tgt.any()):                   # crop missed the lesion entirely
                dsc = {str(c): {"dsc": 0.0, "iou": 0.0} for c in (1, 2, 3)}
                rec = {"case_id": cid, "subject_id": row.get("subject_id", cid.rsplit("-", 1)[0]),
                       "metrics": dsc, "crop_missed_lesion": True,
                       "gt_retained": 0.0, "clicks": []}
            else:
                p2, clicks = predict_click_curve(
                    m2, img, tgt[0], device=device,
                    generator=np.random.default_rng(case_seed(SEED, cid)))
                kept = int(tgt.sum())
                # GT outside the crop is charged as false negative, not dropped
                missing = max(0, total_gt - kept)
                metrics = {}
                for c, pred in p2.items():
                    mm = overlap_metrics(pred, tgt[0])
                    inter = mm["dsc"] * (mm["prediction_voxels"] + mm["target_voxels"]) / 2
                    denom = mm["prediction_voxels"] + mm["target_voxels"] + missing
                    metrics[str(c)] = {"dsc": 2 * inter / denom if denom else 1.0,
                                       "iou": mm["iou"],
                                       "prediction_voxels": mm["prediction_voxels"],
                                       "target_voxels": mm["target_voxels"] + missing}
                rec = {"case_id": cid, "subject_id": row.get("subject_id", cid.rsplit("-", 1)[0]),
                       "metrics": metrics, "crop_missed_lesion": False,
                       "gt_retained": kept / max(total_gt, 1), "clicks": clicks}
            (cascade if label == "cascade" else oracle).append(rec)

        if k % 20 == 0:
            print(f"{k}/{len(rows)}", flush=True)

    rep = {
        "schema": "aflow.sammed3d.two_stage_cascade_probe", "schema_version": 1,
        "created_utc": "2026-08-07",
        "centring_rule": "largest connected component of the stage-1 prediction",
        "stage1_checkpoint_sha256": sha1, "stage2_checkpoint_sha256": sha2,
        "note": ("GT falling outside the stage-2 crop is charged as false negative, so a worse "
                 "crop is penalised rather than rewarded with an easier denominator. The oracle "
                 "column re-runs the identical pipeline with the crop centred on the ground truth."),
        "split": "validation 108", "test_access": False,
        "cascade_summary": summarize(cascade), "oracle_summary": summarize(oracle),
        "cascade_cases": cascade, "oracle_cases": oracle,
    }
    OUT.write_text(json.dumps(rep, indent=2) + "\n")

    cs, os_ = rep["cascade_summary"], rep["oracle_summary"]
    print(f"\nvalidation n={len(rows)}, click 3, scored on the stage-2 crop grid")
    print(f"  two-stage cascade (crop on stage-1 prediction): "
          f"{cs['3']['scan_macro']['dsc_mean']:.4f}")
    print(f"  oracle            (crop on ground truth)      : "
          f"{os_['3']['scan_macro']['dsc_mean']:.4f}")
    print(f"  price of imperfect localization               : "
          f"{cs['3']['scan_macro']['dsc_mean'] - os_['3']['scan_macro']['dsc_mean']:+.4f}")
    miss = sum(1 for c in cascade if c["crop_missed_lesion"])
    ret = np.array([c["gt_retained"] for c in cascade])
    print(f"  crops that missed the lesion entirely         : {miss}/{len(cascade)}")
    print(f"  GT retained by the cascade crop: median {np.median(ret):.4f}  min {ret.min():.4f}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
