"""Create blank manual QC manifests from WAW-Glioma imaging metadata.

The output files document a rule-based visual review. They intentionally do not
replace human review with an unvalidated automated score.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FILES = {
    "preoperative": "2_1_preop_scans_coreg_meta_041125.xlsx",
    "postoperative": "2_2_postop_scans_coreg_meta_041125.xlsx",
}


def study_ids(metadata_path: Path) -> list[str]:
    frame = pd.read_excel(metadata_path)
    fields = frame["file_name"].str.extract(r"^(?P<id>[^_]+)_(?P<exam>[^_]+)_")
    return sorted((fields["id"] + "_" + fields["exam"]).drop_duplicates())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_dir", type=Path, help="Directory containing the released metadata XLSX files.")
    parser.add_argument("--output-dir", type=Path, default=Path("qc_manifests"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for phase, filename in FILES.items():
        for identifier in study_ids(args.release_dir / filename):
            rows.append({
                "study_id": identifier,
                "phase": phase,
                "reviewer": "",
                "review_date": "",
                "whole_brain_coverage": "",
                "gross_rotation_or_translation": "",
                "anatomical_landmark_alignment": "",
                "tumor_boundary_alignment": "",
                "decision": "",
                "exclusion_reason": "",
                "notes": "",
            })
    registration = pd.DataFrame(rows)
    registration.to_csv(args.output_dir / "registration_qc_manifest.csv", index=False)

    segmentation = registration[["study_id", "phase", "reviewer", "review_date"]].copy()
    segmentation["automated_initialization_reviewed"] = ""
    segmentation["manual_correction_performed"] = ""
    segmentation["final_radiologist_review"] = ""
    segmentation["decision"] = ""
    segmentation["notes"] = ""
    segmentation.to_csv(args.output_dir / "segmentation_qc_manifest.csv", index=False)
    print(f"Wrote {len(registration)} study rows to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
