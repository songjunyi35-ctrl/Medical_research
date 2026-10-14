#!/usr/bin/env python
"""Preregistration for the LR-decay arm: machine JSON + human MD from one source.

Single factor against the braincrop arm: the MultiStepLR milestones move from
[120, 180] -- which early stopping at epoch 118 meant NEVER fired -- to
[30, 60, 90], so annealing actually happens where the validation plateau starts.

All prose is rendered from the `machine` dict. No number may be restated in the
MD that is not read from `machine`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "workspace/BraTS/sammed3d_t1ce_wt"
CACHE = Path("/tmp/aflow_braincrop_cache/t1ce_wt_n4_braincrop_128")
PREV = OUT / "sammed3d_braincrop_fullunfreeze_ddp2_train972_val108_e200_s20260806"
RUN = "sammed3d_braincrop_lrdecay_h100_train972_val108_e200_s20260807"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    tr = json.loads((CACHE / "train" / "index.json").read_text())
    va = json.loads((CACHE / "validation" / "index.json").read_text())
    prev_test = json.loads((OUT / "FINAL_TEST_RESULT_BRAINCROP_20260806.json").read_text())
    prev_val = json.loads(
        (PREV / "INDEPENDENT_VALIDATION_108_BRAINCROP_3CLICK_20260806.json").read_text()
    )
    prereg_prev = json.loads((OUT / "BRAINCROP_PREREG_20260806.json").read_text())

    import csv
    hist = list(csv.DictReader(open(PREV / "history.csv")))
    lrs = sorted({r["encoder_lr"] for r in hist})
    gap = lambda r: float(r["train_dice"]) - float(r["val_dice"])

    machine = {
        "schema": "aflow.sammed3d.lrdecay_preregistration",
        "schema_version": 1,
        "created_utc": "2026-08-07",
        "run_identity": RUN,
        "hypothesis": (
            "Moving the MultiStepLR milestones from [120, 180] to [30, 60, 90] raises "
            "validation DSC, because in the braincrop arm early stopping fired at epoch "
            "118 and the encoder LR therefore stayed at its initial value for the entire "
            "run: the model never annealed."
        ),
        "defect_being_fixed": {
            "observed_in": "sammed3d_braincrop_fullunfreeze_ddp2_train972_val108_e200_s20260806",
            "epochs_run": len(hist),
            "distinct_encoder_lr_values_observed": lrs,
            "lr_ever_decayed": len(lrs) > 1,
            "explanation": (
                "patience 40 with milestones [120, 180] means the run always stops before "
                "the first milestone. Both this arm and the whole-head reference arm trained "
                "at a constant LR from start to finish."
            ),
            "why_it_was_kept_until_now": (
                "The defect was deliberately carried over in the braincrop arm to preserve a "
                "single-factor contrast with the reference arm. Both arms are now complete, "
                "so the defect has no remaining comparability value."
            ),
            "overfit_trajectory_train_minus_val": {
                f"epoch_{e}": round(gap(r), 4)
                for e in (10, 30, 60, 78, 118)
                for r in hist if int(r["epoch"]) == e
            },
            "validation_peaked_at_epoch": int(max(hist, key=lambda r: float(r["val_dice"]))["epoch"]),
        },
        "single_factor": {
            "changed": "lr_milestones: [120, 180] -> [30, 60, 90]; gamma unchanged at 0.1",
            "held_identical_to_the_braincrop_arm": [
                "braincrop_resize geometry and the byte-identical preprocessing cache",
                "base checkpoint and its SHA-256",
                "full image-encoder unfreeze, 100,509,232 trainable parameters",
                "AdamW, encoder LR 1e-4 / head LR 1e-5, weight decay 0.1",
                "EFFECTIVE batch 12",
                "200 epoch cap, min 40 epochs before stop, patience 40",
                "bf16",
                "train seed 20260805, validation seed 20260723",
                "affine + elastic + noise augmentation, train split only",
                "DiceCELoss(sigmoid=True, squared_pred=True)",
                "3 accumulated corrective clicks with the recurrent low-res mask, raw logit > 0",
                "the frozen 972 train / 108 validation subject-clean split",
            ],
        },
        "hardware_change_disclosed": {
            "previous": "2 x 23GB GPU, per-GPU batch 2, gradient accumulation 3, explicit all-reduce -> effective batch 12",
            "now": "1 x NVIDIA H100 80GB, per-GPU batch 12, gradient accumulation 1, no distributed path -> effective batch 12",
            "gradient_mathematics": (
                "identical. The old path computed (1/3) * sum over 3 accumulation steps of the "
                "mean over 2 samples, then averaged over 2 ranks = mean over 12 samples. The new "
                "path computes the mean over 12 samples directly. 972/2 = 486 exactly, so "
                "DistributedSampler never padded and no sample was double-counted."
            ),
            "what_does_differ": (
                "RNG streams. The old run used a per-rank RNG offset for augmentation and click "
                "sampling across 2 processes; the new run draws a single stream. Sample ordering "
                "within a step therefore differs. This is a real but second-order difference and "
                "is disclosed rather than claimed absent."
            ),
            "measured_memory_headroom": {
                "batch_2": "11.3 GiB", "batch_4": "22.7 GiB", "batch_8": "44.1 GiB",
                "batch_12": "65.4 GiB", "batch_16": "OOM",
            },
            "why_effective_batch_was_NOT_increased": (
                "972 training cases at effective batch 12 give 81 optimizer steps per epoch; at 24 "
                "that halves to 40 and at 48 to 20. The diagnosed bottleneck is generalization, not "
                "optimization, and reducing gradient noise on a small dataset removes a regularizer. "
                "Raising the effective batch would also add a second factor and destroy the "
                "controlled contrast."
            ),
        },
        "geometry": {
            "name": tr["preprocessing"]["spatial"],
            "uses_gt_for_localization": tr["preprocessing"]["uses_gt_for_localization"],
            "cache_rebuilt_after_machine_reprovision": True,
            "rebuild_byte_identical_to_published_run": {
                "validation_images_sha256_matches_prereg": (
                    va["images_sha256"] == prereg_prev["data"]["validation"]["images_sha256"]
                ),
                "train_images_sha256_matches_prereg": (
                    tr["images_sha256"] == prereg_prev["data"]["train"]["images_sha256"]
                ),
            },
            "builder": "scripts/build_t1ce_wt_braincrop_cache.py",
            "builder_sha256": sha256_file(ROOT / "scripts/build_t1ce_wt_braincrop_cache.py"),
        },
        "data": {
            "train": {"cache_dir": str(CACHE / "train"), "count": tr["count"],
                      "manifest_sha256": tr["manifest_sha256"],
                      "images_sha256": tr["images_sha256"], "test_access": tr["test_access"]},
            "validation": {"cache_dir": str(CACHE / "validation"), "count": va["count"],
                           "manifest_sha256": va["manifest_sha256"],
                           "images_sha256": va["images_sha256"], "test_access": va["test_access"]},
        },
        "baseline_to_beat": {
            "braincrop_validation_scan_c3": prev_val["summary"]["3"]["scan_macro"]["dsc_mean"],
            "braincrop_validation_subject_c3": prev_val["summary"]["3"]["subject_balanced"]["dsc_mean"],
            "braincrop_test_scan_c3": prev_test["result"]["scan_macro"]["C3"],
        },
        "selection_rule": {
            "metric": "val_dice",
            "rule": "strict argmax over completed epochs; ties keep the earlier epoch",
            "test_access_during_selection": False,
        },
        "prediction_registered_before_the_run": {
            "primary": "validation scan-macro DSC at click 3 exceeds the braincrop arm's value",
            "point_estimate": "+0.01 to +0.03, stated before the run",
            "falsifiable": (
                "if validation does not exceed the braincrop arm, the LR schedule is recorded as "
                "not helping and NO test run is taken"
            ),
            "agent_track_record": (
                "In this project the agent predicted the inference-only brain-crop probe would "
                "LOWER the score (refuted: it raised it) and predicted crop-to-nonzero would help "
                "small lesions most (partly refuted: the 2k-8k bin gained most on validation). "
                "Read this prediction with corresponding scepticism."
            ),
        },
        "test_protocol": {
            "runs_allowed": 1,
            "precondition": "a frozen configuration record on disk BEFORE any test NIfTI is opened",
            "only_if": "validation beats the braincrop arm",
            "note": "would be the 7th consumption of the 270-case test by this project",
        },
        "prohibitions_still_in_force": [
            "no threshold / TTA / morphology / connected-component / postprocessing search",
            "no mid-run change to LR, unfreeze depth, initialization, augmentation, epoch cap or patience",
            "no use of any test result to select an epoch, a configuration or a next arm",
            "no rewriting of a binding hash to make a check pass; fail closed instead",
        ],
        "open_bottleneck_not_addressed_by_this_arm": (
            "The generalization gap. The capacity probe fit 48 unaugmented cases to 0.885 while "
            "972-case validation sits at 0.61. Annealing the LR converges better inside the same "
            "hypothesis class; it does not add data or reduce effective capacity."
        ),
    }

    p_json = OUT / "LRDECAY_PREREG_20260807.json"
    p_json.write_text(json.dumps(machine, indent=2) + "\n")

    m = machine
    L: list[str] = []
    a = L.append
    a(f"# Preregistration - {m['run_identity']}")
    a("")
    a(f"Created (UTC): {m['created_utc']}")
    a("")
    a("## 1. Hypothesis")
    a("")
    a(m["hypothesis"])
    a("")
    a("## 2. The defect being fixed")
    a("")
    d = m["defect_being_fixed"]
    a(f"- Observed in: `{d['observed_in']}`")
    a(f"- Epochs run: {d['epochs_run']}")
    a(f"- Distinct encoder LR values observed across the whole run: `{d['distinct_encoder_lr_values_observed']}`")
    a(f"- LR ever decayed: **{d['lr_ever_decayed']}**")
    a(f"- {d['explanation']}")
    a(f"- Why it was kept until now: {d['why_it_was_kept_until_now']}")
    a(f"- Validation peaked at epoch **{d['validation_peaked_at_epoch']}**")
    a("")
    a("| epoch | train - val |")
    a("|---|---|")
    for k, v in d["overfit_trajectory_train_minus_val"].items():
        a(f"| {k.replace('epoch_', '')} | {v:+.4f} |")
    a("")
    a("## 3. The single factor")
    a("")
    a(f"**Changed:** {m['single_factor']['changed']}")
    a("")
    a("**Held identical to the braincrop arm:**")
    a("")
    for item in m["single_factor"]["held_identical_to_the_braincrop_arm"]:
        a(f"- {item}")
    a("")
    a("## 4. Hardware change (disclosed, not hidden)")
    a("")
    h = m["hardware_change_disclosed"]
    a(f"- Previous: {h['previous']}")
    a(f"- Now: {h['now']}")
    a(f"- Gradient mathematics: {h['gradient_mathematics']}")
    a(f"- What does differ: {h['what_does_differ']}")
    a("")
    a("Measured memory headroom on the H100:")
    a("")
    a("| per-GPU batch | peak allocated |")
    a("|---|---|")
    for k, v in h["measured_memory_headroom"].items():
        a(f"| {k.replace('batch_', '')} | {v} |")
    a("")
    a(f"**Why the effective batch was NOT increased:** {h['why_effective_batch_was_NOT_increased']}")
    a("")
    a("## 5. Cache reproducibility")
    a("")
    g = m["geometry"]
    r = g["rebuild_byte_identical_to_published_run"]
    a(f"The machine was reprovisioned and `/tmp` was wiped, so the cache was rebuilt.")
    a(f"- validation images SHA-256 matches the published preregistration: **{r['validation_images_sha256_matches_prereg']}**")
    a(f"- train images SHA-256 matches the published preregistration: **{r['train_images_sha256_matches_prereg']}**")
    a(f"- Builder `{g['builder']}` SHA-256 `{g['builder_sha256']}`")
    a("")
    a("## 6. Baseline to beat")
    a("")
    b = m["baseline_to_beat"]
    a(f"- braincrop validation scan-macro C3: **{b['braincrop_validation_scan_c3']:.4f}**")
    a(f"- braincrop validation subject-balanced C3: {b['braincrop_validation_subject_c3']:.4f}")
    a(f"- braincrop test scan-macro C3 (for context only, not a selection target): {b['braincrop_test_scan_c3']}")
    a("")
    a("## 7. Selection rule")
    a("")
    s = m["selection_rule"]
    a(f"- Metric `{s['metric']}`; {s['rule']}")
    a(f"- Test access during selection: **{s['test_access_during_selection']}**")
    a("")
    a("## 8. Prediction registered before the run")
    a("")
    p = m["prediction_registered_before_the_run"]
    a(f"- Primary: {p['primary']}")
    a(f"- Point estimate: {p['point_estimate']}")
    a(f"- Falsifiable: {p['falsifiable']}")
    a(f"- Agent track record: {p['agent_track_record']}")
    a("")
    a("## 9. Test protocol")
    a("")
    t = m["test_protocol"]
    a(f"- Runs allowed: **{t['runs_allowed']}**, and only if: {t['only_if']}")
    a(f"- Precondition: {t['precondition']}")
    a(f"- {t['note']}")
    a("")
    a("## 10. Prohibitions still in force")
    a("")
    for item in m["prohibitions_still_in_force"]:
        a(f"- {item}")
    a("")
    a("## 11. Bottleneck this arm does NOT address")
    a("")
    a(m["open_bottleneck_not_addressed_by_this_arm"])
    a("")

    p_md = OUT / "LRDECAY_PREREG_20260807.md"
    p_md.write_text("\n".join(L))

    if not (m["geometry"]["rebuild_byte_identical_to_published_run"]["train_images_sha256_matches_prereg"]
            and m["geometry"]["rebuild_byte_identical_to_published_run"]["validation_images_sha256_matches_prereg"]):
        raise SystemExit("FAIL CLOSED: rebuilt cache is not byte-identical to the published run")
    if m["defect_being_fixed"]["lr_ever_decayed"]:
        raise SystemExit("FAIL CLOSED: the premise (LR never decayed) does not hold")

    print(f"wrote {p_json.relative_to(ROOT)}  sha256={sha256_file(p_json)}")
    print(f"wrote {p_md.relative_to(ROOT)}  sha256={sha256_file(p_md)}")


if __name__ == "__main__":
    main()
