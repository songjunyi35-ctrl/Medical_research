"""Validate WAW-Glioma release linkage and report known availability gaps."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def metadata_studies(path: Path) -> set[str]:
    frame = pd.read_excel(path)
    fields = frame["file_name"].str.extract(r"^(?P<id>[^_]+)_(?P<exam>[^_]+)_")
    return set(fields["id"] + "_" + fields["exam"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_dir", type=Path)
    args = parser.parse_args()
    root = args.release_dir

    clinical = pd.read_csv(root / "1_0_clinical_molecular_data_041125.csv", sep=";", dtype=str)
    if clinical["id"].nunique() != len(clinical):
        raise ValueError("Clinical patient identifiers are not unique.")
    print(f"Clinical records: {len(clinical)}")

    expected = {
        "preoperative": metadata_studies(root / "2_1_preop_scans_coreg_meta_041125.xlsx"),
        "postoperative": metadata_studies(root / "2_2_postop_scans_coreg_meta_041125.xlsx"),
    }
    for phase, studies in expected.items():
        print(f"Coregistered {phase} studies: {len(studies)}")

    for phase, filename in {
        "preoperative": "4_radiomics_preop_041125.csv",
        "postoperative": "4_radiomics_postop_041125.csv",
    }.items():
        radiomics = pd.read_csv(root / filename, sep=";", low_memory=False)
        required = {"study_id", "mask_file", "linkage_status", "candidate_study_ids"}
        missing = required - set(radiomics.columns)
        if missing:
            raise ValueError(f"{filename} lacks required columns: {sorted(missing)}")
        matched = radiomics[radiomics["linkage_status"].eq("matched")]
        invalid = set(matched["study_id"]) - expected[phase]
        if invalid:
            raise ValueError(f"{filename} links to studies absent from metadata: {sorted(invalid)}")
        print(f"{phase} radiomics: {len(radiomics)} rows; {len(matched)} uniquely linked; "
              f"{len(radiomics) - len(matched)} flagged as non-unique or unresolved.")

    print("Known postoperative studies lacking a released mask: 289_4, 506_3")


if __name__ == "__main__":
    main()
