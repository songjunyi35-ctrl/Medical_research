"""Audit RHUH80 label metadata only; never download or open a segmentation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "archive/reviews/2026-09-05_30h_chain_audit"
PREACCESS = REVIEW / "RHUH_IMAGE_QC_PREACCESS.json"
SUMS = REVIEW / "source_metadata/RHUH-GBM-nii-v1.sums"
OUTPUT = REVIEW / "RHUH_LABEL_INVENTORY_READINESS.json"
PREACCESS_SHA = "65892e4d90a577ff2623cd2786d5142aeed86be809a5b273faf245321a4481c9"
SUMS_SHA = "7c6ba3fa767bf169679db30caa854e671796416ca365fac6e57662796177c130"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if sha(PREACCESS) != PREACCESS_SHA or sha(SUMS) != SUMS_SHA:
        raise ValueError("RHUH official inventory inputs changed")
    preaccess = json.loads(PREACCESS.read_text())
    sums = {}
    for line in SUMS.read_text().splitlines():
        md5, path = line.split(maxsplit=1)
        if path in sums:
            raise ValueError("duplicate official sums path")
        sums[path] = md5
    selected = preaccess["selected"]
    rows = []
    for item in selected:
        label = item["label"]
        official_path = label["official_path"].lstrip("/")
        row = {
            "case_id": item["case_id"],
            "subject_id": item["subject_id"],
            "timepoint": item["timepoint"],
            "official_path": label["official_path"],
            "basename": label["basename"],
            "bytes": label["bytes"],
            "official_md5": label["official_md5"],
            "sums_md5": sums.get(official_path),
        }
        row["inventory_exact"] = row["sums_md5"] == row["official_md5"]
        rows.append(row)
    subjects = {}
    for row in rows:
        subjects.setdefault(row["subject_id"], []).append(row["timepoint"])
    paths = [row["official_path"] for row in rows]
    special = [row for row in rows if row["basename"] == "segmentation.nii.gz"]
    gates = {
        "cases_80": len(rows) == 80,
        "subjects_40": len(subjects) == 40,
        "each_subject_timepoints_1_2": all(sorted(times) == [1, 2] for times in subjects.values()),
        "unique_paths_80": len(set(paths)) == 80,
        "all_paths_in_official_sums": all(row["inventory_exact"] for row in rows),
        "bytes_exact": sum(row["bytes"] for row in rows) == 2355783,
        "one_frozen_naming_exception": len(special) == 1 and special[0]["case_id"] == "RHUH-0035_2",
        "no_local_label_directory": not (ROOT / "workspace/BraTS/rhuh_gbm_source_20260905/raw_labels").exists(),
    }
    report = {
        "schema": "aflow.rhuh80.label_inventory_readiness",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ready_metadata_only" if all(gates.values()) else "failed_metadata_gate",
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "cases": len(rows),
        "subjects": len(subjects),
        "label_bytes": sum(row["bytes"] for row in rows),
        "special_naming": special,
        "rows": rows,
        "preaccess_sha256": PREACCESS_SHA,
        "official_sums_sha256": SUMS_SHA,
        "runner_sha256": sha(Path(__file__)),
        "labels_downloaded": False,
        "label_voxel_access": False,
        "model_inference": False,
        "automatic_download": False,
    }
    with OUTPUT.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    if not report["all_gates_passed"]:
        raise RuntimeError("RHUH label inventory metadata gate failed")


if __name__ == "__main__":
    main()
