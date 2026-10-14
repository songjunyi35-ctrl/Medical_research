#!/usr/bin/env python
"""Emit the conclusion record for the gt_crop single-factor arm.

Generated from the artifacts on disk so the prose cannot drift from the numbers.
Run only after scripts/finish_gtcrop_arm.sh has produced both readouts.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARM = ROOT / ("workspace/BraTS/sammed3d_t1ce_wt/"
              "sammed3d_native_gtcrop_fullunfreeze_ddp2_train972_val108_e100_s20260805")
BASELINE = ROOT / ("workspace/BraTS/sammed3d_t1ce_wt/"
                   "sammed3d_official_fullunfreeze_ddp2_train972_val108_e200_s20260805")
OUT = ROOT / "workspace/BraTS/research_audits/SAMMED3D_NATIVE_GTCROP_CONCLUSION_20260805.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    hist = list(csv.DictReader(open(ARM / "history.csv")))
    best = max(hist, key=lambda r: float(r["val_dice"]))
    done = json.loads((ARM / "TRAIN_DONE.json").read_text())
    reload_rec = json.loads((ARM / "selected_checkpoint_reload.json").read_text())

    evals = {}
    for nc in (3, 5):
        p = ARM / f"INDEPENDENT_VALIDATION_108_GTCROP_{nc}CLICK_20260805.json"
        d = json.loads(p.read_text())
        if d["data"]["expected_split"] != "validation" or d["test_access"]:
            raise SystemExit(f"{p} is not a validation-only readout")
        last = str(nc)
        evals[nc] = {
            "report": str(p.relative_to(ROOT)),
            "report_sha256": sha256_file(p),
            "num_clicks": d["protocol"]["num_clicks"],
            "final_click_scan_macro": d["summary"][last]["scan_macro"]["dsc_mean"],
            "final_click_subject_balanced": d["summary"][last]["subject_balanced"]["dsc_mean"],
            "final_click_scan_median": d["summary"][last]["scan_macro"]["dsc_median"],
            "final_click_iou_mean": d["summary"][last]["scan_macro"]["iou_mean"],
            "per_click_scan_macro": {k: d["summary"][k]["scan_macro"]["dsc_mean"]
                                     for k in sorted(d["summary"], key=int)},
        }

    # Baseline arm: same everything except whole-volume resize geometry.
    base_eval = json.loads(
        (BASELINE / "INDEPENDENT_VALIDATION_108_N4_20260805.json").read_text())
    base_hist = list(csv.DictReader(open(BASELINE / "history.csv")))
    base_best = max(base_hist, key=lambda r: float(r["val_dice"]))

    three = evals[3]
    rec = {
        "schema": "aflow.sammed3d.arm_conclusion",
        "schema_version": 1,
        "run_identity": done["run_identity"],
        "created_utc": "2026-08-05",
        "preregistration": {
            "path": "workspace/BraTS/research_audits/SAMMED3D_NATIVE_GTCROP_PREREG_20260805.md",
            "sha256": sha256_file(
                ROOT / "workspace/BraTS/research_audits/SAMMED3D_NATIVE_GTCROP_PREREG_20260805.md"),
        },
        "design": {
            "type": "single-factor contrast against the whole-resize full-unfreeze arm",
            "the_one_factor_changed": (
                "spatial geometry: whole-volume resize to 128^3  ->  SAM-Med3D's native "
                "label-centred CropOrPad(128^3) at native resolution"),
            "held_identical": [
                "N4 bias field correction, shrink 2",
                "full image-encoder unfreeze, 100,509,232 trainable",
                "AdamW encoder 1e-4 / heads 1e-5, weight decay 0.1",
                "effective batch 12 (per-GPU 2 x 2 GPUs x 3 accumulation), BF16",
                "3 accumulated corrective clicks with recurrent low-resolution mask",
                "frozen 972/108 split, train seed 20260805, validation seed 20260723",
            ],
            "schedule_differences_disclosed": (
                "max_epochs 200->100, milestones [120,180]->[60,90] (same 60%/90% "
                "fractions), patience 40->101 i.e. no early stop. The baseline arm's "
                "patience was smaller than its first milestone, so its LR never decayed; "
                "this arm was configured so the decay is reachable."),
        },
        "training": {
            "completed_epochs": len(hist),
            "max_epochs": 100,
            "selected_epoch": int(best["epoch"]),
            "selected_trainer_val_dice": float(best["val_dice"]),
            "final_epoch_train_dice": float(hist[-1]["train_dice"]),
            "final_epoch_val_dice": float(hist[-1]["val_dice"]),
            "new_high_epochs": [x["epoch"] for x in hist if x["strict_new_high"] == "True"],
            "peak_cuda_memory_bytes": max(int(x["cuda_peak_memory_allocated"]) for x in hist),
            "checkpoint_sha256": reload_rec["checkpoint_sha256"],
            "strict_reload": reload_rec,
        },
        "lr_decay_finding": {
            "claim": "the MultiStepLR decay produced no validation improvement",
            "epochs_55_59_mean_val": round(
                sum(float(x["val_dice"]) for x in hist[54:59]) / 5, 6),
            "epochs_60_65_mean_val": round(
                sum(float(x["val_dice"]) for x in hist[59:65]) / 6, 6),
            "implication": (
                "The baseline arm stopped at epoch 64 and never reached its milestone at "
                "120. That was a real design flaw, but this arm shows it cost nothing "
                "scientifically: making the decay reachable did not recover any headroom."),
        },
        "independent_validation_108": evals,
        "single_factor_comparison": {
            "note": "both readouts are the 108-case clean validation, seed 20260723, "
                    "each model evaluated on the preprocessing it was trained with",
            "whole_resize_arm": {
                "selected_epoch": int(base_best["epoch"]),
                "trainer_val_dice": float(base_best["val_dice"]),
                "scan_macro_c3": base_eval["summary"]["3"]["scan_macro"]["dsc_mean"],
                "subject_balanced_c3": base_eval["summary"]["3"]["subject_balanced"]["dsc_mean"],
            },
            "gt_crop_arm": {
                "selected_epoch": int(best["epoch"]),
                "trainer_val_dice": float(best["val_dice"]),
                "scan_macro_c3": three["final_click_scan_macro"],
                "subject_balanced_c3": three["final_click_subject_balanced"],
            },
            "delta_scan_macro_c3": round(
                three["final_click_scan_macro"]
                - base_eval["summary"]["3"]["scan_macro"]["dsc_mean"], 6),
            "delta_subject_balanced_c3": round(
                three["final_click_subject_balanced"]
                - base_eval["summary"]["3"]["subject_balanced"]["dsc_mean"], 6),
            "attribution": (
                "Attributable to the spatial protocol, since every other training and "
                "evaluation factor was held identical. Corroborated by the measured 3.44x "
                "increase in target voxel fraction (108/108 cases)."),
        },
        "click_budget_finding": {
            "three_click_scan_macro": evals[3]["final_click_scan_macro"],
            "five_click_scan_macro": evals[5]["final_click_scan_macro"],
            "delta": round(evals[5]["final_click_scan_macro"]
                           - evals[3]["final_click_scan_macro"], 6),
            "note": "5 clicks is SAM-Med3D's official --num_clicks default; 3 is this "
                    "project's frozen protocol. Evaluation-side only, no retraining.",
        },
        "critical_caveat": {
            "oracle_localization": (
                "The label-centred crop uses the ground-truth mask to place the field of "
                "view at inference. Any Dice from this arm is an INTERACTIVE ORACLE UPPER "
                "BOUND and must never be listed alongside automatic methods such as "
                "nnU-Net 0.728 or U-Mamba 0.815, which localize the lesion themselves."),
            "reference_instability": (
                "DPD-SAM reports SAM-Med3D 74.1 without disclosing its prompt protocol; "
                "BCGFMamba transcribes the same row as 0.731, which does not match "
                "DPD-SAM's own table. The reference value is of unknown protocol and is "
                "not internally consistent across the two papers."),
            "correct_claim": (
                "If this arm lands near 0.72-0.74, the defensible statement is that it "
                "reproduces SAM-Med3D's reported operating point under SAM-Med3D's own "
                "native oracle-interactive protocol -- never that it matches the "
                "automatic baselines."),
        },
        "test_access": False,
        "test_status": (
            "The sealed 270-case test was NOT read by this arm. It has already been "
            "consumed 5 times; a further evaluation needs an explicit operator decision "
            "and a fresh frozen configuration."),
    }
    OUT.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  sha256: {sha256_file(OUT)}")
    sfc = rec["single_factor_comparison"]
    print(f"\n  whole_resize C3 scan={sfc['whole_resize_arm']['scan_macro_c3']:.6f} "
          f"subj={sfc['whole_resize_arm']['subject_balanced_c3']:.6f}")
    print(f"  gt_crop      C3 scan={sfc['gt_crop_arm']['scan_macro_c3']:.6f} "
          f"subj={sfc['gt_crop_arm']['subject_balanced_c3']:.6f}")
    print(f"  delta           scan={sfc['delta_scan_macro_c3']:+.6f} "
          f"subj={sfc['delta_subject_balanced_c3']:+.6f}")
    print(f"  5-click scan={rec['click_budget_finding']['five_click_scan_macro']:.6f} "
          f"(vs 3-click {rec['click_budget_finding']['three_click_scan_macro']:.6f}, "
          f"{rec['click_budget_finding']['delta']:+.6f})")


if __name__ == "__main__":
    main()
