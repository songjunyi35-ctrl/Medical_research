#!/usr/bin/env python3
"""Post-hoc saved-count error audit for the frozen WAW152 result."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "archive/research_records/2026-09-06_waw_gliomai152_frozen_panel/RAW_WAW152.json"
COHORT = ROOT / "archive/reviews/2026-09-06_waw_gliomai_source_audit/FROZEN_COHORT.json"
WAW_INDEX = ROOT / "archive/research_records/2026-09-06_waw_gliomai152_frozen_panel/CACHE_INDEX_FINAL.json"
TRAIN_INDEX = ROOT / "workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train/index.json"
EXPECTED = {
    RAW: "ef35301a78f6e47ac0574e5727aa7c5015b2f30cd73e1c8c45487fa8591dde91",
    COHORT: "25de15904d8db5e6d6e304bc73058dba4f43e7534ae0a533f4f06636613a263a",
    WAW_INDEX: "a3831454785715ba48f845115571f99b8adfe61011d29fadaa0b91a4929adc7e",
    TRAIN_INDEX: "7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd",
}
SEED = 20261453
SAMPLES = 10000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def interval(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(SAMPLES, len(values)))
    return [float(value) for value in np.quantile(values[draws].mean(axis=1), (0.025, 0.975))]


def metrics(rows: list[dict]) -> dict[str, np.ndarray]:
    intersection = np.asarray([row["intersection_voxels"] for row in rows], dtype=np.float64)
    predicted = np.asarray([row["predicted_wt_voxels"] for row in rows], dtype=np.float64)
    target = np.asarray([row["target_wt_voxels"] for row in rows], dtype=np.float64)
    precision = np.divide(intersection, predicted, out=np.ones_like(intersection), where=predicted > 0)
    recall = intersection / target
    return {
        "dice": np.asarray([row["locator_dsc"] for row in rows], dtype=np.float64),
        "precision": precision,
        "recall": recall,
        "predicted": predicted,
        "target": target,
        "prediction_target_ratio": predicted / target,
        "point": np.asarray([row["automatic_point_hits_gt"] for row in rows], dtype=bool),
    }


def describe(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def group_deltas(indices: np.ndarray, full: dict, flip: dict, gcsam: dict) -> dict:
    return {
        "cases": int(len(indices)),
        "target_voxels_mean": float(np.mean(full["target"][indices])),
        "full_identity_dice": float(np.mean(full["dice"][indices])),
        "full_flip8_dice": float(np.mean(flip["dice"][indices])),
        "gcsam_identity_dice": float(np.mean(gcsam["dice"][indices])),
        "flip8_minus_full_identity": float(np.mean(flip["dice"][indices] - full["dice"][indices])),
        "gcsam_minus_full_identity": float(np.mean(gcsam["dice"][indices] - full["dice"][indices])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()
    if args.json.exists() or args.png.exists():
        raise FileExistsError("post-hoc output exists")
    for path, expected in EXPECTED.items():
        if sha256(path) != expected:
            raise ValueError(f"source SHA mismatch: {path}")
    raw = json.loads(RAW.read_text())
    cohort = json.loads(COHORT.read_text())
    waw_index = json.loads(WAW_INDEX.read_text())
    train_index = json.loads(TRAIN_INDEX.read_text())
    rows = {
        "full_identity": raw["full972_kd168"]["identity"]["per_case"],
        "full_flip8": raw["full972_kd168"]["flip8"]["per_case"],
        "gcsam_identity": raw["gcsam184"]["identity"]["per_case"],
    }
    case_ids = [row["case_id"] for row in rows["full_identity"]]
    if any([row["case_id"] for row in model_rows] != case_ids for model_rows in rows.values()):
        raise ValueError("paired case order")
    if case_ids != waw_index["case_ids"] or len(case_ids) != 152:
        raise ValueError("cache/report case order")
    values = {name: metrics(model_rows) for name, model_rows in rows.items()}
    full = values["full_identity"]
    flip = values["full_flip8"]
    gcsam = values["gcsam_identity"]
    for index, case_id in enumerate(case_ids):
        if waw_index["details"][case_id]["resized_wt"] != int(full["target"][index]):
            raise ValueError("saved target/cache mismatch")

    grade_by_case = {row["case_id"]: row["who_grade"] for row in cohort["cases"]}
    grades = np.asarray([grade_by_case[case_id] for case_id in case_ids])
    target_order = np.argsort(full["target"], kind="stable")
    quartiles = np.array_split(target_order, 4)
    train_target = np.sort(np.asarray(list(train_index["per_case_positive_voxels"].values()), dtype=float))

    def percentile(array: np.ndarray) -> np.ndarray:
        return np.searchsorted(train_target, array, side="right") / len(train_target)

    comparisons = {}
    for offset, (name, candidate) in enumerate((("flip8_minus_full", flip), ("gcsam_minus_full", gcsam))):
        comparisons[name] = {
            "dice_delta": float(np.mean(candidate["dice"] - full["dice"])),
            "precision_delta": float(np.mean(candidate["precision"] - full["precision"])),
            "precision_delta_95ci_posthoc": interval(candidate["precision"] - full["precision"], SEED + offset * 10 + 1),
            "recall_delta": float(np.mean(candidate["recall"] - full["recall"])),
            "recall_delta_95ci_posthoc": interval(candidate["recall"] - full["recall"], SEED + offset * 10 + 2),
            "prediction_voxel_delta_mean": float(np.mean(candidate["predicted"] - full["predicted"])),
            "point_gained": int(np.sum(candidate["point"] & ~full["point"])),
            "point_lost": int(np.sum(~candidate["point"] & full["point"])),
            "spearman_log10_target_vs_dice_delta": {
                "rho": float(spearmanr(np.log10(full["target"]), candidate["dice"] - full["dice"]).statistic),
                "nominal_p_uncorrected": float(spearmanr(np.log10(full["target"]), candidate["dice"] - full["dice"]).pvalue),
            },
        }

    result = {
        "schema": "aflow.waw_gliomai152.posthoc_error_audit",
        "schema_version": 2,
        "status": "posthoc_descriptive_no_model_or_cache_array_access_no_selection",
        "cases": 152,
        "models": {
            name: {
                "dice": describe(model["dice"]),
                "precision": describe(model["precision"]),
                "recall": describe(model["recall"]),
                "predicted_voxels": describe(model["predicted"]),
                "target_voxels": describe(model["target"]),
                "prediction_target_ratio": describe(model["prediction_target_ratio"]),
                "point_hits": int(np.sum(model["point"])),
            }
            for name, model in values.items()
        },
        "comparisons": comparisons,
        "target_size_quartiles": {
            f"Q{number}_small_to_large": group_deltas(indices, full, flip, gcsam)
            for number, indices in enumerate(quartiles, start=1)
        },
        "who_grade": {
            str(grade): group_deltas(np.flatnonzero(grades == grade), full, flip, gcsam)
            for grade in (2, 3, 4)
        },
        "train972_empirical_size_support": {
            "train_target_percentiles": {
                "p50": float(np.quantile(train_target, 0.5)),
                "p75": float(np.quantile(train_target, 0.75)),
                "p90": float(np.quantile(train_target, 0.9)),
                "p95": float(np.quantile(train_target, 0.95)),
                "max": float(np.max(train_target)),
            },
            "waw_target_percentile": describe(percentile(full["target"])),
            "full_prediction_percentile": describe(percentile(full["predicted"])),
            "flip8_prediction_percentile": describe(percentile(flip["predicted"])),
            "gcsam_prediction_percentile": describe(percentile(gcsam["predicted"])),
            "waw_targets_above_train_p90": int(np.sum(full["target"] > np.quantile(train_target, 0.9))),
            "waw_targets_above_train_p95": int(np.sum(full["target"] > np.quantile(train_target, 0.95))),
            "waw_targets_above_train_max": int(np.sum(full["target"] > np.max(train_target))),
        },
        "bootstrap": {"samples": SAMPLES, "base_seed": SEED, "posthoc": True},
        "forbidden_use": "no grade/size/scanner router, threshold, checkpoint/view selection, or cohort retest",
        "new_model_inference": False,
        "image_or_checkpoint_access": False,
        "source_sha256": {str(path.relative_to(ROOT)): expected for path, expected in EXPECTED.items()},
        "analyzer_sha256": sha256(Path(__file__)),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    labels = ["Full identity", "Full Flip8", "GCSAM identity"]
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    models = [full, flip, gcsam]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4), constrained_layout=True)
    x = np.arange(2)
    width = 0.24
    for index, (label, color, model) in enumerate(zip(labels, colors, models, strict=True)):
        axes[0].bar(x + (index - 1) * width, [np.mean(model["precision"]), np.mean(model["recall"])], width, label=label, color=color)
    axes[0].set(xticks=x, xticklabels=["Precision", "Recall"], ylim=(0.55, 0.9), title="Overlap components")
    axes[0].legend(frameon=False, fontsize=8)

    qx = np.arange(4)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].plot(qx, [result["target_size_quartiles"][f"Q{i}_small_to_large"]["flip8_minus_full_identity"] for i in range(1, 5)], marker="o", color=colors[1], label="Flip8 − Full")
    axes[1].plot(qx, [result["target_size_quartiles"][f"Q{i}_small_to_large"]["gcsam_minus_full_identity"] for i in range(1, 5)], marker="o", color=colors[2], label="GCSAM − Full")
    axes[1].set(xticks=qx, xticklabels=["Q1 small", "Q2", "Q3", "Q4 large"], ylabel="Dice delta", title="Post-hoc target-size quartiles")
    axes[1].legend(frameon=False, fontsize=8)

    gx = np.arange(3)
    axes[2].axhline(0, color="black", lw=1)
    axes[2].plot(gx, [result["who_grade"][str(i)]["flip8_minus_full_identity"] for i in (2, 3, 4)], marker="o", color=colors[1], label="Flip8 − Full")
    axes[2].plot(gx, [result["who_grade"][str(i)]["gcsam_minus_full_identity"] for i in (2, 3, 4)], marker="o", color=colors[2], label="GCSAM − Full")
    axes[2].set(xticks=gx, xticklabels=["Grade 2", "Grade 3", "Grade 4"], ylabel="Dice delta", title="Post-hoc WHO grade")
    axes[2].legend(frameon=False, fontsize=8)

    axes[3].hist(percentile(full["target"]), bins=np.linspace(0, 1, 11), alpha=0.55, label="WAW target", color="#8172b2")
    axes[3].hist(percentile(full["predicted"]), bins=np.linspace(0, 1, 11), alpha=0.45, label="Full prediction", color=colors[0])
    axes[3].hist(percentile(gcsam["predicted"]), bins=np.linspace(0, 1, 11), alpha=0.35, label="GCSAM prediction", color=colors[2])
    axes[3].set(xlabel="Empirical percentile in train972 size", ylabel="Cases", title="Target/prediction vs train support")
    axes[3].legend(frameon=False, fontsize=8)
    fig.suptitle("WAW152 post-hoc saved-count audit (descriptive; no tuning)")
    fig.savefig(args.png, dpi=180)
    plt.close(fig)
    print(json.dumps({"json_sha256": sha256(args.json), "png_sha256": sha256(args.png)}, indent=2))


if __name__ == "__main__":
    main()
