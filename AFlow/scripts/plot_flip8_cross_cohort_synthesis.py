#!/usr/bin/env python3
"""Synthesize saved Full972 identity-vs-Flip8 evidence without new inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "internal": (
        ROOT / "archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/RAW_TEST270.json",
        "29efc557aa12b6f82fd079edd84cc3c8c1e5ad4536880c016b664932234f45ad",
    ),
    "internal_ci_correction": (
        ROOT / "archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/BOOTSTRAP_CORRECTION.json",
        "f94fedaa7bc70357422bd0ea10f9d127527955de62822035d3b382c19eaf0e0f",
    ),
    "africa": (
        ROOT / "archive/research_records/2026-09-06_brats_africa95_frozen_baselines/RAW_EXTERNAL95.json",
        "c9fb07be4412c160129037104936e4c34b6390fa733b998b8a8f35524ccd30b3",
    ),
    "cptac": (
        ROOT / "archive/research_records/2026-09-06_cptac33_africa_informed_confirmation/RAW_CPTAC33.json",
        "4b9d8c90e1b5ec45b0e664e560301db20affd6a96387f416a62b561f8d57b4fd",
    ),
    "waw": (
        ROOT / "archive/research_records/2026-09-06_waw_gliomai152_frozen_panel/RAW_WAW152.json",
        "ef35301a78f6e47ac0574e5727aa7c5015b2f30cd73e1c8c45487fa8591dde91",
    ),
    "waw_postop": (
        ROOT / "archive/research_records/2026-09-06_waw_postop_only31_frozen_panel/RAW_WAW_POSTOP31.json",
        "72269a5a872389db9d6dd316d7d46b614bad8d81796b0bb72625814907d98a98",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def assert_close(actual: float, expected: float) -> None:
    if not np.isclose(actual, expected, rtol=0, atol=1e-12):
        raise ValueError(f"saved result inconsistency: {actual} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    args = parser.parse_args()
    if any(path.exists() for path in (args.json, args.png, args.pdf)):
        raise FileExistsError("output already exists")
    if not (args.json.parent == args.png.parent == args.pdf.parent):
        raise ValueError("outputs must share one directory")
    args.json.parent.mkdir(parents=True, exist_ok=True)

    for path, expected in SOURCES.values():
        if sha256(path) != expected:
            raise ValueError(f"source SHA mismatch: {path}")

    internal = load_json(SOURCES["internal"][0])
    correction = load_json(SOURCES["internal_ci_correction"][0])
    africa = load_json(SOURCES["africa"][0])
    cptac = load_json(SOURCES["cptac"][0])
    waw = load_json(SOURCES["waw"][0])
    waw_postop = load_json(SOURCES["waw_postop"][0])

    rows = [
        {
            "cohort": "Reused internal test270",
            "scope": "270 scans/119 subjects; reused post-test development, not blind/external",
            "cases": 270,
            "subjects": 119,
            "identity_scan": internal["identity"]["locator_dsc_mean"],
            "identity_subject": internal["identity"]["locator_subject_dsc_mean"],
            "identity_scan_ci": correction["identity"]["scan_dsc_mean_95ci"],
            "identity_subject_ci": correction["identity"]["subject_dsc_mean_95ci"],
            "identity_points": internal["point_hits"]["identity"],
            "flip8_scan": internal["flip8"]["locator_dsc_mean"],
            "flip8_subject": internal["flip8"]["locator_subject_dsc_mean"],
            "flip8_scan_ci": correction["flip8"]["scan_dsc_mean_95ci"],
            "flip8_subject_ci": correction["flip8"]["subject_dsc_mean_95ci"],
            "flip8_points": internal["point_hits"]["flip8"],
            "scan_delta": internal["flip8_minus_identity"]["flip8_minus_identity"]["scan_dice"],
            "scan_delta_ci": internal["flip8_minus_identity"]["scan_dice_95ci"],
            "subject_delta": internal["flip8_minus_identity"]["flip8_minus_identity"]["subject_dice"],
            "subject_delta_ci": internal["flip8_minus_identity"]["subject_dice_95ci"],
            "point_delta": internal["point_hits"]["delta"],
            "improved_worse_tied": [
                internal["flip8_minus_identity"]["scan_improved"],
                internal["flip8_minus_identity"]["scan_worse"],
                internal["flip8_minus_identity"]["scan_tied"],
            ],
        },
        {
            "cohort": "BraTS-Africa95",
            "scope": "public post-challenge; four centers; institution-disjoint/no patient bridge",
            "cases": 95,
            "subjects": 95,
            "identity_scan": africa["full972_kd168"]["identity"]["locator_dsc_mean"],
            "identity_subject": africa["full972_kd168"]["identity"]["locator_subject_dsc_mean"],
            "identity_scan_ci": africa["full972_kd168"]["identity_bootstrap"]["scan_dsc_mean_95ci"],
            "identity_subject_ci": africa["full972_kd168"]["identity_bootstrap"]["subject_dsc_mean_95ci"],
            "identity_points": africa["full972_kd168"]["identity"]["automatic_point_hits"],
            "flip8_scan": africa["full972_kd168"]["flip8"]["locator_dsc_mean"],
            "flip8_subject": africa["full972_kd168"]["flip8"]["locator_subject_dsc_mean"],
            "flip8_scan_ci": africa["full972_kd168"]["flip8_bootstrap"]["scan_dsc_mean_95ci"],
            "flip8_subject_ci": africa["full972_kd168"]["flip8_bootstrap"]["subject_dsc_mean_95ci"],
            "flip8_points": africa["full972_kd168"]["flip8"]["automatic_point_hits"],
            "scan_delta": africa["comparisons"]["full_flip8_minus_identity"]["flip8_minus_identity"]["scan_dice"],
            "scan_delta_ci": africa["comparisons"]["full_flip8_minus_identity"]["paired_scan_dice_95ci"],
            "subject_delta": africa["comparisons"]["full_flip8_minus_identity"]["flip8_minus_identity"]["subject_dice"],
            "subject_delta_ci": africa["comparisons"]["full_flip8_minus_identity"]["paired_subject_dice_95ci"],
            "point_delta": africa["comparisons"]["full_flip8_minus_identity"]["flip8_minus_identity"]["point_hits"],
            "improved_worse_tied": [
                africa["comparisons"]["full_flip8_minus_identity"]["cases_improved"],
                africa["comparisons"]["full_flip8_minus_identity"]["cases_worse"],
                africa["comparisons"]["full_flip8_minus_identity"]["cases_tied"],
            ],
        },
        {
            "cohort": "CPTAC33 public Training",
            "scope": "public labels external to this project; not hidden blind",
            "cases": 33,
            "subjects": 33,
            "identity_scan": cptac["full972_kd168"]["identity"]["locator_dsc_mean"],
            "identity_subject": cptac["full972_kd168"]["identity"]["locator_subject_dsc_mean"],
            "identity_scan_ci": cptac["full972_kd168"]["identity_bootstrap"]["scan_dsc_mean_95ci"],
            "identity_subject_ci": cptac["full972_kd168"]["identity_bootstrap"]["subject_dsc_mean_95ci"],
            "identity_points": cptac["full972_kd168"]["identity"]["automatic_point_hits"],
            "flip8_scan": cptac["full972_kd168"]["flip8"]["locator_dsc_mean"],
            "flip8_subject": cptac["full972_kd168"]["flip8"]["locator_subject_dsc_mean"],
            "flip8_scan_ci": cptac["full972_kd168"]["flip8_bootstrap"]["scan_dsc_mean_95ci"],
            "flip8_subject_ci": cptac["full972_kd168"]["flip8_bootstrap"]["subject_dsc_mean_95ci"],
            "flip8_points": cptac["full972_kd168"]["flip8"]["automatic_point_hits"],
            "scan_delta": cptac["comparisons"]["full_flip8_minus_identity"]["flip8_minus_identity"]["scan_dice"],
            "scan_delta_ci": cptac["comparisons"]["full_flip8_minus_identity"]["paired_scan_dice_95ci"],
            "subject_delta": cptac["comparisons"]["full_flip8_minus_identity"]["flip8_minus_identity"]["subject_dice"],
            "subject_delta_ci": cptac["comparisons"]["full_flip8_minus_identity"]["paired_subject_dice_95ci"],
            "point_delta": cptac["comparisons"]["full_flip8_minus_identity"]["flip8_minus_identity"]["point_hits"],
            "improved_worse_tied": [
                cptac["comparisons"]["full_flip8_minus_identity"]["cases_improved"],
                cptac["comparisons"]["full_flip8_minus_identity"]["cases_worse"],
                cptac["comparisons"]["full_flip8_minus_identity"]["cases_tied"],
            ],
        },
        {
            "cohort": "WAW-GliomAI152",
            "scope": "public preoperative final T1c-mask intersection; institution-disjoint/no patient bridge",
            "cases": 152,
            "subjects": 152,
            "identity_scan": waw["full972_kd168"]["identity"]["locator_dsc_mean"],
            "identity_subject": waw["full972_kd168"]["identity"]["locator_subject_dsc_mean"],
            "identity_scan_ci": waw["full972_kd168"]["identity_bootstrap"]["scan_dsc_mean_95ci"],
            "identity_subject_ci": waw["full972_kd168"]["identity_bootstrap"]["subject_dsc_mean_95ci"],
            "identity_points": waw["full972_kd168"]["identity"]["automatic_point_hits"],
            "flip8_scan": waw["full972_kd168"]["flip8"]["locator_dsc_mean"],
            "flip8_subject": waw["full972_kd168"]["flip8"]["locator_subject_dsc_mean"],
            "flip8_scan_ci": waw["full972_kd168"]["flip8_bootstrap"]["scan_dsc_mean_95ci"],
            "flip8_subject_ci": waw["full972_kd168"]["flip8_bootstrap"]["subject_dsc_mean_95ci"],
            "flip8_points": waw["full972_kd168"]["flip8"]["automatic_point_hits"],
            "scan_delta": waw["comparisons"]["full_flip8_minus_identity"]["flip8_minus_identity"]["scan_dice"],
            "scan_delta_ci": waw["comparisons"]["full_flip8_minus_identity"]["paired_scan_dice_95ci"],
            "subject_delta": waw["comparisons"]["full_flip8_minus_identity"]["flip8_minus_identity"]["subject_dice"],
            "subject_delta_ci": waw["comparisons"]["full_flip8_minus_identity"]["paired_subject_dice_95ci"],
            "point_delta": waw["comparisons"]["full_flip8_minus_identity"]["flip8_minus_identity"]["point_hits"],
            "improved_worse_tied": [
                waw["comparisons"]["full_flip8_minus_identity"]["cases_improved"],
                waw["comparisons"]["full_flip8_minus_identity"]["cases_worse"],
                waw["comparisons"]["full_flip8_minus_identity"]["cases_tied"],
            ],
        },
        {
            "cohort": "WAW early-postop31",
            "scope": "public day0-22 earliest scan; patients absent from preop152; not hidden blind",
            "cases": 31,
            "subjects": 31,
            "identity_scan": waw_postop["full972_kd168"]["identity"]["locator_dsc_mean"],
            "identity_subject": waw_postop["full972_kd168"]["identity"]["locator_subject_dsc_mean"],
            "identity_scan_ci": waw_postop["full972_kd168"]["identity_bootstrap"]["scan_dsc_mean_95ci"],
            "identity_subject_ci": waw_postop["full972_kd168"]["identity_bootstrap"]["subject_dsc_mean_95ci"],
            "identity_points": waw_postop["full972_kd168"]["identity"]["automatic_point_hits"],
            "flip8_scan": waw_postop["full972_kd168"]["flip8"]["locator_dsc_mean"],
            "flip8_subject": waw_postop["full972_kd168"]["flip8"]["locator_subject_dsc_mean"],
            "flip8_scan_ci": waw_postop["full972_kd168"]["flip8_bootstrap"]["scan_dsc_mean_95ci"],
            "flip8_subject_ci": waw_postop["full972_kd168"]["flip8_bootstrap"]["subject_dsc_mean_95ci"],
            "flip8_points": waw_postop["full972_kd168"]["flip8"]["automatic_point_hits"],
            "scan_delta": waw_postop["comparisons"]["full_flip8_minus_identity"]["flip8_minus_identity"]["scan_dice"],
            "scan_delta_ci": waw_postop["comparisons"]["full_flip8_minus_identity"]["paired_scan_dice_95ci"],
            "subject_delta": waw_postop["comparisons"]["full_flip8_minus_identity"]["flip8_minus_identity"]["subject_dice"],
            "subject_delta_ci": waw_postop["comparisons"]["full_flip8_minus_identity"]["paired_subject_dice_95ci"],
            "point_delta": waw_postop["comparisons"]["full_flip8_minus_identity"]["flip8_minus_identity"]["point_hits"],
            "improved_worse_tied": [waw_postop["comparisons"]["full_flip8_minus_identity"]["cases_improved"], waw_postop["comparisons"]["full_flip8_minus_identity"]["cases_worse"], waw_postop["comparisons"]["full_flip8_minus_identity"]["cases_tied"]],
        },
    ]

    for row in rows:
        assert_close(row["flip8_scan"] - row["identity_scan"], row["scan_delta"])
        assert_close(row["flip8_subject"] - row["identity_subject"], row["subject_delta"])
        if row["flip8_points"] - row["identity_points"] != row["point_delta"]:
            raise ValueError("point delta inconsistency")
        if sum(row["improved_worse_tied"]) != row["cases"]:
            raise ValueError("case count inconsistency")

    result = {
        "schema": "aflow.flip8.cross_cohort_saved_report_synthesis",
        "schema_version": 3,
        "status": "post_access_descriptive_separate_cohorts_no_pooling_or_selection",
        "checkpoint_sha256": "f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b",
        "cohorts": rows,
        "direction_summary": {
            "positive_scan_point_estimates": sum(row["scan_delta"] > 0 for row in rows),
            "cohorts": len(rows),
            "scan_ci_entirely_above_zero": [row["cohort"] for row in rows if row["scan_delta_ci"][0] > 0],
            "scan_ci_crosses_zero": [row["cohort"] for row in rows if row["scan_delta_ci"][0] <= 0 <= row["scan_delta_ci"][1]],
            "interpretation": "All five saved cohorts have positive Scan Dice point estimates; paired intervals support improvement on reused internal270, CPTAC33, and WAW preop152, while Africa95 and WAW early-postop31 remain uncertain. This is not a pooled effect or proof of universal benefit.",
        },
        "deployment": {
            "identity_locator_forwards_per_case": 1,
            "flip8_locator_forwards_per_case": 8,
            "checkpoint_or_parameter_increase": False,
            "latency_compute_multiplier_approx": 8,
        },
        "ucsd75_not_included_reason": "fixed Flip8 was not run on UCSD75",
        "new_model_inference": False,
        "selection_or_tuning": False,
        "source_sha256": {name: expected for name, (_, expected) in SOURCES.items()},
        "analyzer_sha256": sha256(Path(__file__)),
    }
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    labels = ["Internal270", "Africa95", "CPTAC33", "WAW-pre152", "WAW-post31"]
    x = np.arange(len(rows))
    width = 0.32
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), constrained_layout=True)
    for offset, key, ci_key, label, color in [
        (-width / 2, "identity_scan", "identity_scan_ci", "Identity (1×)", "#4c72b0"),
        (width / 2, "flip8_scan", "flip8_scan_ci", "Fixed Flip8 (8×)", "#dd8452"),
    ]:
        values = np.array([row[key] for row in rows])
        cis = np.array([row[ci_key] for row in rows])
        axes[0].bar(x + offset, values, width, label=label, color=color)
        axes[0].errorbar(
            x + offset,
            values,
            yerr=np.vstack((values - cis[:, 0], cis[:, 1] - values)),
            fmt="none",
            ecolor="black",
            capsize=3,
            lw=1,
        )
    axes[0].set(
        ylabel="Scan Dice",
        title="Absolute scores (separate cohorts)",
        xticks=x,
        xticklabels=labels,
        ylim=(0.47, 0.89),
    )
    axes[0].legend(frameon=False)

    deltas = np.array([row["scan_delta"] for row in rows])
    delta_cis = np.array([row["scan_delta_ci"] for row in rows])
    axes[1].axhline(0, color="black", lw=1)
    axes[1].errorbar(
        x,
        deltas,
        yerr=np.vstack((deltas - delta_cis[:, 0], delta_cis[:, 1] - deltas)),
        fmt="o",
        color="#c44e52",
        capsize=5,
    )
    axes[1].set(
        ylabel="Flip8 − identity Scan Dice",
        title="Positive points; certainty differs",
        xticks=x,
        xticklabels=labels,
    )

    identity_points = np.array([row["identity_points"] / row["cases"] for row in rows])
    flip8_points = np.array([row["flip8_points"] / row["cases"] for row in rows])
    axes[2].bar(x - width / 2, identity_points, width, label="Identity (1×)", color="#4c72b0")
    axes[2].bar(x + width / 2, flip8_points, width, label="Fixed Flip8 (8×)", color="#dd8452")
    axes[2].set(
        ylabel="Point hit rate",
        title="Point effects are also non-negative",
        xticks=x,
        xticklabels=labels,
        ylim=(0.4, 1.01),
    )
    axes[2].legend(frameon=False)
    fig.suptitle("Frozen Flip8 synthesis (saved reports; no pooling, inference, or tuning)")
    fig.savefig(args.png, dpi=180)
    fig.savefig(args.pdf)
    plt.close(fig)
    print(
        json.dumps(
            {
                "cohorts": len(rows),
                "json_sha256": sha256(args.json),
                "png_sha256": sha256(args.png),
                "pdf_sha256": sha256(args.pdf),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
