"""Compute rank eta-squared radiomics association with MRI field strength.

This is an exploratory cross-sectional analysis, not a test-retest or ICC
reproducibility analysis. It uses only uniquely linked radiomics rows and
compares 3T with 1.5T within each phase and mask label.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_PATTERN = re.compile(r"^original_(shape|firstorder|glcm|glrlm|glszm|gldm|ngtdm)_")
MIN_GROUP_SIZE = 10


def scanner_by_study(path: Path) -> pd.DataFrame:
    metadata = pd.read_excel(path)
    parts = metadata["file_name"].str.extract(r"^(?P<id>[^_]+)_(?P<exam>[^_]+)_")
    metadata = metadata.assign(study_id=parts["id"] + "_" + parts["exam"])
    result = metadata[["study_id", "FieldStrengthLabel"]].drop_duplicates()
    if (result.groupby("study_id").size() > 1).any():
        raise ValueError("Field strength differs within at least one study.")
    return result.rename(columns={"FieldStrengthLabel": "field_strength"})


def rank_eta_squared(values: pd.Series, groups: pd.Series) -> float | None:
    data = pd.DataFrame({"value": values, "group": groups}).replace([np.inf, -np.inf], np.nan).dropna()
    eligible = data["group"].value_counts()
    data = data[data["group"].isin(eligible[eligible >= MIN_GROUP_SIZE].index)]
    if data["group"].nunique() < 2 or data["value"].nunique() < 2:
        return None
    ranks = data["value"].rank(method="average")
    overall = ranks.mean()
    group_stats = data.assign(rank=ranks).groupby("group")["rank"].agg(["count", "mean"])
    between = float((group_stats["count"] * (group_stats["mean"] - overall) ** 2).sum())
    total = float(((ranks - overall) ** 2).sum())
    return between / total if total else None


def analyse(root: Path, phase: str, radiomics_file: str, metadata_file: str) -> list[dict[str, object]]:
    radiomics = pd.read_csv(root / radiomics_file, sep=";", low_memory=False)
    radiomics = radiomics[radiomics["linkage_status"].eq("matched")]
    radiomics = radiomics.merge(scanner_by_study(root / metadata_file), on="study_id", how="left")
    radiomics = radiomics[radiomics["field_strength"].isin(["3T", "1.5T"])]
    features = [column for column in radiomics.columns if FEATURE_PATTERN.match(column)]
    summaries = []
    for label, subset in radiomics.groupby("mask_label"):
        results = []
        for feature in features:
            eta = rank_eta_squared(pd.to_numeric(subset[feature], errors="coerce"), subset["field_strength"])
            if eta is not None:
                results.append(eta)
        if results:
            values = np.array(results)
            counts = subset["field_strength"].value_counts()
            summaries.append({
                "phase": phase,
                "mask_label": int(label),
                "n_radiomics_rows": int(len(subset)),
                "n_3T": int(counts.get("3T", 0)),
                "n_1_5T": int(counts.get("1.5T", 0)),
                "features_tested": int(len(values)),
                "median_rank_eta_squared": float(np.median(values)),
                "percent_rank_eta_squared_ge_0_06": float((values >= 0.06).mean() * 100),
                "percent_rank_eta_squared_ge_0_14": float((values >= 0.14).mean() * 100),
            })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("radiomics_scanner_association_summary.csv"))
    args = parser.parse_args()
    rows = []
    rows += analyse(args.release_dir, "preoperative", "4_radiomics_preop_041125.csv", "2_1_preop_scans_coreg_meta_041125.xlsx")
    rows += analyse(args.release_dir, "postoperative", "4_radiomics_postop_041125.csv", "2_2_postop_scans_coreg_meta_041125.xlsx")
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
