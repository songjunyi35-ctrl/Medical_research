#!/usr/bin/env python
"""Generate the braincrop arm's preregistration: machine JSON + human MD, one source.

The human document is rendered ENTIRELY from the `machine` dict below. On the
previous arm the hand-written MD drifted from its own JSON (it claimed a
trainable count and an unfreeze policy the run never used), so no prose here may
restate a number that is not read from `machine`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "workspace/BraTS/sammed3d_t1ce_wt"
RUN = "sammed3d_braincrop_fullunfreeze_ddp2_train972_val108_e200_s20260806"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_index(rel: str) -> dict:
    base = Path(rel) if Path(rel).is_absolute() else ROOT / rel
    return json.loads((base / "index.json").read_text())


def main() -> None:
    tr = cache_index("/tmp/aflow_braincrop_cache/t1ce_wt_n4_braincrop_128/train")
    va = cache_index("/tmp/aflow_braincrop_cache/t1ce_wt_n4_braincrop_128/validation")
    ref = json.loads(
        (OUT / "FINAL_TEST_RESULT_FULLUNFREEZE_20260805.json").read_text()
    )
    diag = json.loads((OUT / "BOTTLENECK_DIAGNOSIS_20260806.json").read_text())
    probe = diag["crop_to_nonzero_probe_20260806"]

    machine = {
        "schema": "aflow.sammed3d.braincrop_preregistration",
        "schema_version": 1,
        "created_utc": "2026-08-06",
        "run_identity": RUN,
        "hypothesis": (
            "Inserting crop-to-nonzero before the 128^3 resize raises validation "
            "DSC relative to the whole-head-resize arm, because it roughly doubles "
            "the lesion's voxel count on the model grid and stops ~52% of the input "
            "(and ~267 of the image encoder's 512 tokens) from covering background."
        ),
        "prior_evidence": {
            "source": "workspace/BraTS/sammed3d_t1ce_wt/BOTTLENECK_DIAGNOSIS_20260806.json",
            "probe": "same whole-head-trained weights, geometry swapped at inference only",
            "epoch64_delta_native_1mm": probe["results_native_1mm"]["epoch64_latest"]["delta"],
            "epoch64_ci95": probe["results_native_1mm"]["epoch64_latest"]["ci95"],
            "epoch64_sign_test_p": probe["results_native_1mm"]["epoch64_latest"]["sign_test_p"],
            "epoch24_delta_native_1mm": probe["results_native_1mm"]["epoch24_best"]["delta"],
            "epoch24_not_significant": True,
            "caveat": (
                "the probe fed brain-cropped input to whole-head-trained weights, "
                "so its delta is a LOWER bound on the trained effect, not an estimate"
            ),
        },
        "single_factor": {
            "changed": "spatial preprocessing only: whole_resize -> braincrop_resize",
            "held_identical_to_the_reference_arm": [
                "base checkpoint and its SHA-256",
                "full image-encoder unfreeze, 100,509,232 trainable parameters",
                "AdamW, encoder LR 1e-4 / head LR 1e-5, weight decay 0.1",
                "per-GPU batch 2 x 2 GPUs x 3 accumulation steps = effective batch 12",
                "MultiStepLR milestones [120, 180], gamma 0.1",
                "200 epoch cap, min 40 epochs before stop, patience 40",
                "bf16, explicit gradient all-reduce across 2 ranks",
                "train seed 20260805, validation seed 20260723",
                "affine + elastic + noise augmentation, train split only",
                "DiceCELoss(sigmoid=True, squared_pred=True)",
                "3 accumulated corrective clicks with the recurrent low-res mask, raw logit > 0",
                "N4 bias field correction, shrink 2, 4x50 iterations, nonzero mask",
                "nonzero-population z-score",
                "the frozen 972 train / 108 validation subject-clean split",
            ],
            "why_this_is_attributable": (
                "Unlike the reference arm -- which moved unfreeze policy, LR direction, "
                "N4, epochs, batch and DDP together and was therefore explicitly marked "
                "non-attributable -- this arm changes exactly one factor, so its delta "
                "against the reference arm IS a controlled contrast."
            ),
        },
        "geometry": {
            "name": tr["preprocessing"]["spatial"],
            "description": tr["preprocessing"]["spatial_description"],
            "uses_gt_for_localization": tr["preprocessing"]["uses_gt_for_localization"],
            "normalization": tr["preprocessing"]["normalization"],
            "builder": "scripts/build_t1ce_wt_braincrop_cache.py",
            "builder_sha256": sha256_file(ROOT / "scripts/build_t1ce_wt_braincrop_cache.py"),
        },
        "data": {
            "train": {
                "cache_dir": "/tmp/aflow_braincrop_cache/t1ce_wt_n4_braincrop_128/train",
                "count": tr["count"], "manifest_sha256": tr["manifest_sha256"],
                "images_sha256": tr["images_sha256"], "labels_sha256": tr["labels_sha256"],
                "test_access": tr["test_access"],
            },
            "validation": {
                "cache_dir": "/tmp/aflow_braincrop_cache/t1ce_wt_n4_braincrop_128/validation",
                "count": va["count"], "manifest_sha256": va["manifest_sha256"],
                "images_sha256": va["images_sha256"], "labels_sha256": va["labels_sha256"],
                "test_access": va["test_access"],
            },
        },
        "selection_rule": {
            "metric": "val_dice",
            "rule": "strict argmax over completed epochs; ties keep the earlier epoch",
            "test_access_during_selection": False,
        },
        "test_protocol": {
            "runs_allowed": 1,
            "precondition": "a frozen configuration record must be on disk BEFORE any test NIfTI is opened",
            "evaluator": "scripts/evaluate_sammed3d_t1ce_wt_braincrop.py",
            "reference_arm_test_result_scan_c3": ref["result"]["scan_macro"]["C3"],
            "reference_arm_test_result_subject_c3": ref["result"]["subject_balanced"]["C3"],
            "note": (
                "This will be the 6th consumption of the 270-case test by this project. "
                "The number is an internal held-out readout, not clean external validation."
            ),
        },
        "prohibitions_still_in_force": [
            "no threshold / TTA / morphology / connected-component / postprocessing search",
            "no mid-run change to LR, unfreeze depth, initialization, augmentation, epoch cap or patience",
            "no use of any test result to select an epoch, a configuration or a next arm",
            "no rewriting of a binding hash to make a check pass; fail closed instead",
        ],
        "known_defects_carried_over_deliberately": [
            {
                "defect": "patience 40 is smaller than the first LR milestone 120, so MultiStepLR can never fire before early stopping",
                "why_kept": "changing it would break the single-factor contrast with the reference arm, which ran under the same defect and stopped at epoch 64",
            },
            {
                "defect": "the trainer constructs T1ceWtPreprocessConfig(spatial_preprocessing='whole_resize') while the cache is braincrop_resize",
                "impact": "none on behaviour: that object is consumed only by build_training_augmentation, which is geometry-independent. The geometry recorded in train_config.json is copied from the cache index and is therefore correct.",
                "why_kept": "the alternative is editing a trainer whose SHA is recorded in the completed arm's train_config.json",
            },
        ],
        "prediction_registered_before_the_run": {
            "primary": "validation scan-macro DSC at click 3 exceeds the reference arm's 0.5633",
            "falsifiable": "if validation DSC does not exceed 0.5633, the crop is recorded as not helping when trained with, and no test run is taken",
            "agent_prior": (
                "The agent previously predicted the inference-only probe would LOWER "
                "the score and was refuted by measurement. That miss is recorded so this "
                "prediction is read with appropriate scepticism."
            ),
        },
    }

    prereg_json = OUT / f"BRAINCROP_PREREG_{machine['created_utc'].replace('-', '')}.json"
    prereg_json.write_text(json.dumps(machine, indent=2) + "\n")

    m = machine
    lines: list[str] = []
    a = lines.append
    a(f"# Preregistration - {m['run_identity']}")
    a("")
    a(f"Created (UTC): {m['created_utc']}")
    a("")
    a("## 1. Hypothesis")
    a("")
    a(m["hypothesis"])
    a("")
    a("## 2. Prior evidence")
    a("")
    p = m["prior_evidence"]
    a(f"- Source: `{p['source']}`")
    a(f"- Probe design: {p['probe']}")
    a(f"- epoch 64 delta (native 1mm): **{p['epoch64_delta_native_1mm']:+.4f}**, "
      f"95% CI [{p['epoch64_ci95'][0]:+.4f}, {p['epoch64_ci95'][1]:+.4f}], "
      f"sign-test p = {p['epoch64_sign_test_p']:.2g}")
    a(f"- epoch 24 delta: {p['epoch24_delta_native_1mm']:+.4f} (NOT significant)")
    a(f"- Caveat: {p['caveat']}")
    a("")
    a("## 3. The single factor")
    a("")
    a(f"**Changed:** {m['single_factor']['changed']}")
    a("")
    a("**Held identical to the reference arm:**")
    a("")
    for item in m["single_factor"]["held_identical_to_the_reference_arm"]:
        a(f"- {item}")
    a("")
    a(m["single_factor"]["why_this_is_attributable"])
    a("")
    a("## 4. Geometry")
    a("")
    g = m["geometry"]
    a(f"- Name: `{g['name']}`")
    a(f"- {g['description']}")
    a(f"- Uses GT for localization: **{g['uses_gt_for_localization']}**")
    a(f"- Normalization: {g['normalization']}")
    a(f"- Builder: `{g['builder']}` (SHA-256 `{g['builder_sha256']}`)")
    a("")
    a("## 5. Data binding")
    a("")
    a("| split | count | manifest SHA-256 | images SHA-256 | test_access |")
    a("|---|---|---|---|---|")
    for name in ("train", "validation"):
        d = m["data"][name]
        a(f"| {name} | {d['count']} | `{d['manifest_sha256'][:16]}...` | "
          f"`{d['images_sha256'][:16]}...` | {d['test_access']} |")
    a("")
    a("## 6. Selection rule")
    a("")
    s = m["selection_rule"]
    a(f"- Metric: `{s['metric']}`")
    a(f"- Rule: {s['rule']}")
    a(f"- Test access during selection: **{s['test_access_during_selection']}**")
    a("")
    a("## 7. Test protocol")
    a("")
    t = m["test_protocol"]
    a(f"- Runs allowed: **{t['runs_allowed']}**")
    a(f"- Precondition: {t['precondition']}")
    a(f"- Evaluator: `{t['evaluator']}`")
    a(f"- Reference arm on the same 270 cases: scan C3 = {t['reference_arm_test_result_scan_c3']}, "
      f"subject C3 = {t['reference_arm_test_result_subject_c3']}")
    a(f"- {t['note']}")
    a("")
    a("## 8. Prohibitions still in force")
    a("")
    for item in m["prohibitions_still_in_force"]:
        a(f"- {item}")
    a("")
    a("## 9. Known defects carried over deliberately")
    a("")
    for d in m["known_defects_carried_over_deliberately"]:
        a(f"- **{d['defect']}**")
        if "impact" in d:
            a(f"  - Impact: {d['impact']}")
        a(f"  - Why kept: {d['why_kept']}")
    a("")
    a("## 10. Prediction registered before the run")
    a("")
    r = m["prediction_registered_before_the_run"]
    a(f"- Primary: {r['primary']}")
    a(f"- Falsifiable: {r['falsifiable']}")
    a(f"- {r['agent_prior']}")
    a("")

    prereg_md = OUT / f"BRAINCROP_PREREG_{m['created_utc'].replace('-', '')}.md"
    prereg_md.write_text("\n".join(lines))

    # consistency check: every number quoted in the MD must appear in the JSON
    blob = json.dumps(machine)
    for probe_value in (str(m["data"]["train"]["count"]),
                        str(m["data"]["validation"]["count"]),
                        m["geometry"]["builder_sha256"],
                        m["data"]["train"]["manifest_sha256"][:16]):
        if probe_value not in blob:
            raise SystemExit(f"consistency check failed for {probe_value!r}")
    md = prereg_md.read_text()
    if "whole_resize" in md.split("## 3.")[1].split("## 4.")[0].replace(
            m["single_factor"]["changed"], ""):
        raise SystemExit("MD section 3 mentions whole_resize outside the change line")

    print(f"wrote {prereg_json.relative_to(ROOT)}  sha256={sha256_file(prereg_json)}")
    print(f"wrote {prereg_md.relative_to(ROOT)}  sha256={sha256_file(prereg_md)}")


if __name__ == "__main__":
    main()
