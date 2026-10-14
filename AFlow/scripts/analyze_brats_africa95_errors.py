#!/usr/bin/env python3
"""Post-hoc descriptive size/error audit for frozen Africa95 outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "archive/research_records/2026-09-06_brats_africa95_frozen_baselines"
RAW = RECORD / "RAW_EXTERNAL95.json"
RAW_SHA = "c9fb07be4412c160129037104936e4c34b6390fa733b998b8a8f35524ccd30b3"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""): digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--json", type=Path, required=True); parser.add_argument("--png", type=Path, required=True); parser.add_argument("--pdf", type=Path, required=True); args = parser.parse_args()
    if any(path.exists() for path in (args.json, args.png, args.pdf)): raise FileExistsError("analysis output exists")
    if sha(RAW) != RAW_SHA: raise ValueError("raw SHA mismatch")
    raw = json.loads(RAW.read_text())
    sources = {"full_identity": raw["full972_kd168"]["identity"]["per_case"], "full_flip8": raw["full972_kd168"]["flip8"]["per_case"], "gcsam_identity": raw["gcsam184"]["identity"]["per_case"]}
    ids = [x["case_id"] for x in sources["full_identity"]]
    if any([x["case_id"] for x in rows] != ids for rows in sources.values()): raise ValueError("case order mismatch")
    rows = []
    for values in zip(*(sources[name] for name in sources), strict=True):
        fi, ff, gc = values; target = int(fi["target_wt_voxels"])
        if any(int(x["target_wt_voxels"]) != target for x in values): raise ValueError("target mismatch")
        rows.append({"case_id": fi["case_id"], "target_voxels": target, "log10_target_voxels": float(np.log10(target)), "full_identity_dice": fi["locator_dsc"], "full_flip8_dice": ff["locator_dsc"], "gcsam_identity_dice": gc["locator_dsc"], "flip8_minus_identity_dice": ff["locator_dsc"] - fi["locator_dsc"], "gcsam_minus_full_identity_dice": gc["locator_dsc"] - fi["locator_dsc"], "full_identity_volume_ratio": fi["predicted_wt_voxels"] / target, "full_flip8_volume_ratio": ff["predicted_wt_voxels"] / target, "gcsam_identity_volume_ratio": gc["predicted_wt_voxels"] / target, "full_identity_point": bool(fi["automatic_point_hits_gt"]), "full_flip8_point": bool(ff["automatic_point_hits_gt"]), "gcsam_identity_point": bool(gc["automatic_point_hits_gt"])})
    ordered = sorted(range(95), key=lambda i: (rows[i]["target_voxels"], rows[i]["case_id"]))
    quartiles = [ordered[:24], ordered[24:48], ordered[48:72], ordered[72:]]
    quartile_summary = []
    for number, indices in enumerate(quartiles, 1):
        values = [rows[i] for i in indices]
        quartile_summary.append({"quartile": number, "cases": len(values), "target_voxels_min": min(x["target_voxels"] for x in values), "target_voxels_max": max(x["target_voxels"] for x in values), "target_voxels_median": statistics.median(x["target_voxels"] for x in values), "full_identity_dice": statistics.fmean(x["full_identity_dice"] for x in values), "full_flip8_dice": statistics.fmean(x["full_flip8_dice"] for x in values), "gcsam_identity_dice": statistics.fmean(x["gcsam_identity_dice"] for x in values), "flip8_minus_identity_dice": statistics.fmean(x["flip8_minus_identity_dice"] for x in values), "gcsam_minus_full_identity_dice": statistics.fmean(x["gcsam_minus_full_identity_dice"] for x in values), "full_identity_point_hits": sum(x["full_identity_point"] for x in values), "full_flip8_point_hits": sum(x["full_flip8_point"] for x in values), "gcsam_identity_point_hits": sum(x["gcsam_identity_point"] for x in values)})
    log_volume = np.asarray([x["log10_target_voxels"] for x in rows])
    fields = ["full_identity_dice", "full_flip8_dice", "gcsam_identity_dice", "flip8_minus_identity_dice", "gcsam_minus_full_identity_dice", "full_identity_volume_ratio", "full_flip8_volume_ratio", "gcsam_identity_volume_ratio"]
    correlations = {field: {"spearman_rho": float(spearmanr(log_volume, [x[field] for x in rows]).statistic), "two_sided_p_descriptive": float(spearmanr(log_volume, [x[field] for x in rows]).pvalue)} for field in fields}
    point_transitions = {"gcsam_lost_vs_full_identity": [x["case_id"] for x in rows if x["full_identity_point"] and not x["gcsam_identity_point"]], "gcsam_gained_vs_full_identity": [x["case_id"] for x in rows if x["gcsam_identity_point"] and not x["full_identity_point"]], "flip8_lost_vs_identity": [x["case_id"] for x in rows if x["full_identity_point"] and not x["full_flip8_point"]], "flip8_gained_vs_identity": [x["case_id"] for x in rows if x["full_flip8_point"] and not x["full_identity_point"]]}
    result = {"schema": "aflow.brats_africa95.posthoc_error_audit", "schema_version": 1, "status": "descriptive_only_no_tuning_or_selection", "source_raw_sha256": RAW_SHA, "quartile_definition": "sort all95 by target voxels then split 24/24/24/23; defined after result only for description", "quartiles": quartile_summary, "spearman_vs_log10_target_volume": correlations, "point_transitions": point_transitions, "per_case": rows, "new_model_inference": False, "threshold_or_case_selection": False, "interpretation_guard": "nominal p-values are descriptive and not multiplicity-adjusted; do not use them for promotion or parameter selection", "analyzer_sha256": sha(Path(__file__))}
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    colors = {"full_identity": "#3566a8", "full_flip8": "#d4842f", "gcsam_identity": "#2b8c64"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for name, field, label in (("full_identity", "full_identity_dice", "Full972 identity"), ("full_flip8", "full_flip8_dice", "Full972 Flip8"), ("gcsam_identity", "gcsam_identity_dice", "GCSAM identity")):
        axes[0].scatter(log_volume, [x[field] for x in rows], s=17, alpha=.62, color=colors[name], label=label)
    axes[0].set(xlabel="log10 target WT voxels (128³)", ylabel="Dice", title="Absolute Dice vs target size"); axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].scatter(log_volume, [x["gcsam_minus_full_identity_dice"] for x in rows], s=20, alpha=.7, color=colors["gcsam_identity"], label="GCSAM − Full identity")
    axes[1].scatter(log_volume, [x["flip8_minus_identity_dice"] for x in rows], s=18, alpha=.6, color=colors["full_flip8"], label="Flip8 − identity")
    axes[1].set(xlabel="log10 target WT voxels (128³)", ylabel="paired Dice difference", title="Frozen paired differences"); axes[1].legend(frameon=False, fontsize=8)
    for name, field, label in (("full_identity", "full_identity_volume_ratio", "Full972 identity"), ("full_flip8", "full_flip8_volume_ratio", "Full972 Flip8"), ("gcsam_identity", "gcsam_identity_volume_ratio", "GCSAM identity")):
        axes[2].scatter(log_volume, [x[field] for x in rows], s=17, alpha=.62, color=colors[name], label=label)
    axes[2].axhline(1, color="black", lw=1, ls="--"); axes[2].set(xlabel="log10 target WT voxels (128³)", ylabel="predicted / target volume", title="Systematic under-segmentation", ylim=(0, min(3, max(x["gcsam_identity_volume_ratio"] for x in rows) * 1.05)))
    fig.suptitle("BraTS-Africa95 post-hoc error audit (descriptive; no tuning)", fontsize=12)
    fig.savefig(args.png, dpi=180); fig.savefig(args.pdf); plt.close(fig)
    print(json.dumps({"quartiles": quartile_summary, "correlations": correlations, "point_transitions": {k: len(v) for k,v in point_transitions.items()}, "json_sha256": sha(args.json), "png_sha256": sha(args.png), "pdf_sha256": sha(args.pdf)}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
