#!/usr/bin/env python
"""Inter-observer segmentation reproducibility analysis for the revision sample.

This script compares Reader 1 reference/final masks with Reader 2 independent
masks without modifying source files and without resampling. It writes all
outputs to RESULTS_DIR.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage


# By default the script expects to live in the release's `code` directory.
# Set WAW_GLIOMA_RELEASE_DIR to run it from another location.
BASE_DIR = Path(os.environ.get("WAW_GLIOMA_RELEASE_DIR", Path(__file__).resolve().parent.parent))
READER1_PRE_DIR = BASE_DIR / "3_1_preop_masks_041125"
READER1_POST_DIR = BASE_DIR / "3_2_postop_masks_041125"
READER2_PRE_DIR = BASE_DIR / "revisions" / "final_dice" / "pre"
READER2_POST_DIR = BASE_DIR / "revisions" / "final_dice" / "post"
MAPPING_CSV = BASE_DIR / "revisions" / "mask_rename_matching" / "reader2_to_reader1_best_mask_matches_manual_corrected.csv"
RESULTS_DIR = BASE_DIR / "revisions" / "interobserver_reproducibility_results"

EXPECTED_SAMPLE_SIZE_PER_GROUP = 15
AFFINE_ATOL = 1e-4

# Keep semantic labels explicit and conservative. If the project protocol uses
# different names, edit only this mapping; metric calculations are label-number based.
LABEL_NAMES = {
    1: "label_1",
    2: "label_2",
    3: "label_3",
    4: "label_4",
}


@dataclass(frozen=True)
class MaskPair:
    group: str
    case_id: str
    reader1_case_id: str
    reader1_exam_index: str
    reader1_mask: Path
    reader2_mask: Path
    match_confidence: str
    manual_override: str


def is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def index_reader2(root: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not is_nifti(path):
            continue
        case_id = path.parent.name
        if case_id in indexed:
            duplicates.append(case_id)
        indexed[case_id] = path
    if duplicates:
        raise RuntimeError(f"Duplicate Reader 2 cases in {root}: {sorted(set(duplicates))}")
    return indexed


def build_pairs() -> tuple[list[MaskPair], dict[str, int]]:
    r2_pre = index_reader2(READER2_PRE_DIR)
    r2_post = index_reader2(READER2_POST_DIR)
    mapping = pd.read_csv(MAPPING_CSV)

    pairs: list[MaskPair] = []
    for _, row in mapping.iterrows():
        reader1_mask = Path(str(row["reader1_candidate_mask"]))
        reader2_mask = Path(str(row["reader2_mask"]))
        pairs.append(
            MaskPair(
                group=str(row["group"]),
                case_id=str(row["reader2_case_id"]),
                reader1_case_id=str(row["reader1_candidate_case_id"]),
                reader1_exam_index=str(row["reader1_candidate_exam_index"]),
                reader1_mask=reader1_mask,
                reader2_mask=reader2_mask,
                match_confidence=str(row.get("match_confidence", "")),
                manual_override=str(row.get("manual_override", "")),
            )
        )

    counts = {
        "r2_pre_cases": len(r2_pre),
        "r2_post_cases": len(r2_post),
        "mapped_pairs": len(pairs),
        "mapped_pre_pairs": sum(1 for pair in pairs if pair.group == "PRE"),
        "mapped_post_pairs": sum(1 for pair in pairs if pair.group == "POST"),
    }
    return pairs, counts


def load_mask(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if not np.all(np.isclose(data, np.rint(data))):
        raise RuntimeError(f"Non-integer labels found in {path}")
    return image, np.rint(data).astype(np.int16)


def labels_present(data: np.ndarray) -> list[int]:
    return sorted(int(v) for v in np.unique(data) if int(v) != 0)


def affine_orientation_code(image: nib.Nifti1Image) -> str:
    return "".join(nib.aff2axcodes(image.affine))


def qc_pair(pair: MaskPair) -> dict[str, object]:
    row: dict[str, object] = {
        "group": pair.group,
        "case_id": pair.case_id,
        "reader2_old_case_id": pair.case_id,
        "reader1_renamed_case_id": pair.reader1_case_id,
        "reader1_exam_index": pair.reader1_exam_index,
        "reader1_mask": str(pair.reader1_mask),
        "reader2_mask": str(pair.reader2_mask),
        "match_confidence": pair.match_confidence,
        "manual_override": pair.manual_override,
        "labels_present_reader1": "",
        "labels_present_reader2": "",
        "image dimensions": "",
        "voxel spacing": "",
        "affine/orientation compatibility": "NA",
        "QC status": "FAIL",
        "qc_details": "",
    }
    if not pair.reader1_mask.exists() or not pair.reader2_mask.exists():
        row["qc_details"] = "missing Reader 1 or Reader 2 mask"
        return row

    img1, data1 = load_mask(pair.reader1_mask)
    img2, data2 = load_mask(pair.reader2_mask)
    shape_ok = img1.shape == img2.shape
    spacing1 = tuple(float(x) for x in img1.header.get_zooms()[:3])
    spacing2 = tuple(float(x) for x in img2.header.get_zooms()[:3])
    spacing_ok = np.allclose(spacing1, spacing2, atol=1e-5)
    affine_ok = np.allclose(img1.affine, img2.affine, atol=AFFINE_ATOL)
    orient_ok = affine_orientation_code(img1) == affine_orientation_code(img2)

    labels1 = labels_present(data1)
    labels2 = labels_present(data2)
    row.update(
        {
            "labels_present_reader1": ";".join(map(str, labels1)),
            "labels_present_reader2": ";".join(map(str, labels2)),
            "image dimensions": "x".join(map(str, img1.shape)),
            "voxel spacing": ";".join(f"{v:.6g}" for v in spacing1),
            "affine/orientation compatibility": bool(shape_ok and spacing_ok and affine_ok and orient_ok),
        }
    )

    details = []
    if not shape_ok:
        details.append(f"shape mismatch R1={img1.shape} R2={img2.shape}")
    if not spacing_ok:
        details.append(f"spacing mismatch R1={spacing1} R2={spacing2}")
    if not affine_ok:
        details.append("affine mismatch")
    if not orient_ok:
        details.append(f"orientation mismatch R1={affine_orientation_code(img1)} R2={affine_orientation_code(img2)}")
    if data1.size and np.count_nonzero(data1) == 0:
        details.append("Reader 1 mask empty")
    if data2.size and np.count_nonzero(data2) == 0:
        details.append("Reader 2 mask empty")
    if set(labels1) ^ set(labels2):
        details.append(f"labels differ R1={labels1} R2={labels2}")

    row["QC status"] = "PASS" if shape_ok and spacing_ok and affine_ok and orient_ok else "FAIL"
    row["qc_details"] = "; ".join(details)
    return row


def dice_score(a: np.ndarray, b: np.ndarray) -> float:
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return math.nan
    return 2.0 * int(np.logical_and(a, b).sum()) / denom


def surface_voxels(mask: np.ndarray) -> np.ndarray:
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_and(mask, np.logical_not(eroded))


def hd95_mm(a: np.ndarray, b: np.ndarray, spacing: tuple[float, float, float]) -> float:
    if not a.any() or not b.any():
        return math.nan
    surf_a = surface_voxels(a)
    surf_b = surface_voxels(b)
    dt_a = ndimage.distance_transform_edt(~surf_a, sampling=spacing)
    dt_b = ndimage.distance_transform_edt(~surf_b, sampling=spacing)
    distances = np.concatenate([dt_b[surf_a], dt_a[surf_b]])
    if distances.size == 0:
        return 0.0
    return float(np.percentile(distances, 95))


def calculate_case_metrics(pair: MaskPair, all_labels: list[int]) -> list[dict[str, object]]:
    img1, data1 = load_mask(pair.reader1_mask)
    img2, data2 = load_mask(pair.reader2_mask)
    spacing = tuple(float(x) for x in img1.header.get_zooms()[:3])
    voxel_volume_ml = float(np.prod(spacing)) / 1000.0
    rows = []
    for label in all_labels:
        m1 = data1 == label
        m2 = data2 == label
        present1 = bool(m1.any())
        present2 = bool(m2.any())
        concordant_absence = not present1 and not present2
        discordant_presence = present1 != present2

        vol1 = float(m1.sum() * voxel_volume_ml)
        vol2 = float(m2.sum() * voxel_volume_ml)
        abs_diff = abs(vol1 - vol2)
        rel_diff = math.nan if (vol1 + vol2) == 0 else 200.0 * abs_diff / (vol1 + vol2)

        if concordant_absence:
            dice = math.nan
            hd95 = math.nan
            hd95_reason = "concordant_absence"
        elif discordant_presence:
            dice = 0.0
            hd95 = math.nan
            hd95_reason = "discordant_presence"
        else:
            dice = dice_score(m1, m2)
            hd95 = hd95_mm(m1, m2, spacing)
            hd95_reason = ""

        rows.append(
            {
                "group": pair.group,
                "case_id": pair.case_id,
                "reader2_old_case_id": pair.case_id,
                "reader1_renamed_case_id": pair.reader1_case_id,
                "reader1_exam_index": pair.reader1_exam_index,
                "label": label,
                "label_name": LABEL_NAMES.get(label, f"label_{label}"),
                "reader1_present": present1,
                "reader2_present": present2,
                "concordant_absence": concordant_absence,
                "discordant_presence": discordant_presence,
                "dice": dice,
                "hd95_mm": hd95,
                "hd95_reason": hd95_reason,
                "reader1_volume_ml": vol1,
                "reader2_volume_ml": vol2,
                "absolute_volume_difference_ml": abs_diff,
                "relative_volume_difference_percent": rel_diff,
            }
        )
    return rows


def summarize(case_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in ["PRE", "POST", "Overall"]:
        group_df = case_df if group == "Overall" else case_df[case_df["group"] == group]
        for label in sorted(case_df["label"].unique()):
            df = group_df[group_df["label"] == label]
            evaluable = df[~df["concordant_absence"]]
            dice = evaluable["dice"].dropna()
            hd95 = evaluable["hd95_mm"].dropna()
            rel = evaluable["relative_volume_difference_percent"].dropna()
            rows.append(
                {
                    "Group": group,
                    "Label": LABEL_NAMES.get(int(label), f"label_{label}"),
                    "Label value": int(label),
                    "n evaluable": int(len(evaluable)),
                    "Dice, median (IQR)": format_median_iqr(dice),
                    "HD95 [mm], median (IQR)": format_median_iqr(hd95),
                    "Relative volume difference [%], median (IQR)": format_median_iqr(rel),
                    "Concordant absence, n": int(df["concordant_absence"].sum()),
                    "Discordant presence, n": int(df["discordant_presence"].sum()),
                    "dice_median": float(dice.median()) if len(dice) else math.nan,
                    "dice_q1": float(dice.quantile(0.25)) if len(dice) else math.nan,
                    "dice_q3": float(dice.quantile(0.75)) if len(dice) else math.nan,
                    "hd95_median": float(hd95.median()) if len(hd95) else math.nan,
                    "hd95_q1": float(hd95.quantile(0.25)) if len(hd95) else math.nan,
                    "hd95_q3": float(hd95.quantile(0.75)) if len(hd95) else math.nan,
                    "relative_volume_difference_median": float(rel.median()) if len(rel) else math.nan,
                    "relative_volume_difference_q1": float(rel.quantile(0.25)) if len(rel) else math.nan,
                    "relative_volume_difference_q3": float(rel.quantile(0.75)) if len(rel) else math.nan,
                    "relative_volume_difference_min": float(rel.min()) if len(rel) else math.nan,
                    "relative_volume_difference_max": float(rel.max()) if len(rel) else math.nan,
                    "dice_n": int(len(dice)),
                    "dice_min": float(dice.min()) if len(dice) else math.nan,
                    "dice_max": float(dice.max()) if len(dice) else math.nan,
                    "hd95_n": int(len(hd95)),
                    "hd95_min": float(hd95.min()) if len(hd95) else math.nan,
                    "hd95_max": float(hd95.max()) if len(hd95) else math.nan,
                    "relative_volume_difference_n": int(len(rel)),
                }
            )
    return pd.DataFrame(rows)


def format_median_iqr(series: pd.Series) -> str:
    series = series.dropna()
    if len(series) == 0:
        return "NA"
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return f"{series.median():.3f} ({q1:.3f}-{q3:.3f})"


def make_plots(case_df: pd.DataFrame) -> None:
    for metric, ylabel, stem in [
        ("dice", "Dice coefficient", "dice_distribution_by_label"),
        ("hd95_mm", "HD95 [mm]", "hd95_distribution_by_label"),
    ]:
        plot_df = case_df.dropna(subset=[metric]).copy()
        labels = sorted(plot_df["label"].unique())
        fig, ax = plt.subplots(figsize=(8, 4.8))
        positions = []
        data = []
        tick_positions = []
        tick_labels = []
        pos = 1
        for label in labels:
            for group in ["PRE", "POST"]:
                values = plot_df[(plot_df["label"] == label) & (plot_df["group"] == group)][metric].values
                data.append(values)
                positions.append(pos)
                pos += 1
            tick_positions.append(pos - 1.5)
            tick_labels.append(LABEL_NAMES.get(int(label), f"label_{label}"))
            pos += 0.75
        ax.boxplot(data, positions=positions, widths=0.6, showfliers=True)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Label; paired boxes are PRE then POST")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"{stem}.png", dpi=300)
        fig.savefig(RESULTS_DIR / f"{stem}.pdf")
        plt.close(fig)


def print_report(pairing_df: pd.DataFrame, case_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    pre_matched = int(((pairing_df["group"] == "PRE") & pairing_df["reader1_mask"].ne("") & pairing_df["reader2_mask"].ne("")).sum())
    post_matched = int(((pairing_df["group"] == "POST") & pairing_df["reader1_mask"].ne("") & pairing_df["reader2_mask"].ne("")).sum())
    unmatched = int((pairing_df["reader1_mask"].eq("") | pairing_df["reader2_mask"].eq("")).sum())
    geom_mismatch = int((pairing_df["affine/orientation compatibility"] == False).sum())
    empty_masks = int(pairing_df["qc_details"].str.contains("mask empty", na=False).sum())
    unexpected = sorted(set(case_df["label"]) - set(LABEL_NAMES))
    completed = int(case_df["dice"].notna().sum())

    print("\nQuality-control report")
    print(f"PRE cases matched: {pre_matched}")
    print(f"POST cases matched: {post_matched}")
    print(f"Unmatched cases: {unmatched}")
    print(f"Geometry/affine mismatches: {geom_mismatch}")
    print(f"Empty masks: {empty_masks}")
    print(f"Unexpected labels: {len(unexpected)} {unexpected}")
    print(f"Comparisons with numeric Dice completed: {completed}")

    overall = summary_df[summary_df["Group"] == "Overall"]
    print("\nMethods paragraph")
    print(
        "A random technical-validation sample comprising 15 pre-operative and 15 "
        "post-operative MRI examinations from unique patients was segmented "
        "independently by a second reader who was blinded to the final reference "
        "masks. Reader 1 and Reader 2 masks were paired using the manually "
        "verified rename mapping generated from mask-overlap QC. "
        "Before comparison, image dimensions, voxel spacing, orientation and affine "
        "matrices were checked, and no masks were resampled. Inter-observer "
        "agreement was quantified separately for each non-background label using "
        "the Dice similarity coefficient and the 95th percentile bidirectional "
        "surface Hausdorff distance in millimetres. Labels absent in both readers "
        "were recorded as concordant absences and excluded from Dice/HD95 "
        "distributions, whereas labels present in only one reader were recorded as "
        "discordant presence with Dice set to 0 and HD95 left undefined."
    )
    print("\nResults paragraph")
    parts = []
    for _, row in overall.iterrows():
        parts.append(
            f"{row['Label']} (label {row['Label value']}): Dice {row['Dice, median (IQR)']}, "
            f"HD95 {row['HD95 [mm], median (IQR)']} mm"
        )
    print(
        "All matched mask pairs passing geometric QC were included in the "
        "inter-observer reproducibility analysis. Across PRE and POST cases, "
        + "; ".join(parts)
        + ". Concordant absences and discordant-presence cases were tabulated "
        "separately for each label."
    )


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pairs, counts = build_pairs()

    pairing_rows = [qc_pair(pair) for pair in pairs]
    pairing_df = pd.DataFrame(pairing_rows)
    pairing_df.to_csv(RESULTS_DIR / "mask_pairing_manifest.csv", index=False)

    print("Directory structure inspected")
    for key, value in counts.items():
        print(f"{key}: {value}")
    print("\nPairing table")
    print(pairing_df[["group", "reader2_old_case_id", "reader1_renamed_case_id", "reader1_exam_index", "reader1_mask", "reader2_mask", "labels_present_reader1", "labels_present_reader2", "QC status", "qc_details"]].to_string(index=False))

    failed = pairing_df[pairing_df["QC status"] != "PASS"]
    if not failed.empty:
        print("\nStopping because at least one pair failed geometry/missing-file QC.")
        return 2

    all_labels = sorted(
        {
            int(label)
            for labels in pd.concat([pairing_df["labels_present_reader1"], pairing_df["labels_present_reader2"]])
            for label in str(labels).split(";")
            if label
        }
    )
    print(f"\nActual non-background label values found: {all_labels}")

    case_rows = []
    pair_lookup = {(row["group"], row["case_id"]): row for _, row in pairing_df.iterrows()}
    for pair in pairs:
        if pair_lookup[(pair.group, pair.case_id)]["QC status"] == "PASS":
            case_rows.extend(calculate_case_metrics(pair, all_labels))
    case_df = pd.DataFrame(case_rows)
    case_df.to_csv(RESULTS_DIR / "interobserver_reproducibility_case_level.csv", index=False)

    summary_df = summarize(case_df)
    summary_df.to_csv(RESULTS_DIR / "interobserver_reproducibility_summary.csv", index=False)
    make_plots(case_df)
    print_report(pairing_df, case_df, summary_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
