"""Audit historical teacher training identity against recent student-held-out inners."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return rows, {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    source = ROOT / "data/datasets/brats2024_t1ce_plain_locator_confirm_seed20260817_v1/train_cases.jsonl"
    rows, evidence = load(source)
    assert evidence["sha256"] == "a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16"
    cases = {row["case_id"] for row in rows}
    subjects = {row["subject_id"] for row in rows}
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "completed_metadata_only",
        "teacher_train": {**evidence, "scans": len(cases), "subjects": len(subjects)},
        "replications": [], "new_model_inference": False, "new_label_voxel_access": False,
        "interpretation": "student-fit-disjoint is not end-to-end teacher-training-disjoint; indirect leakage risk, not proof of direct student label access or numerical bias magnitude"}
    for path in sorted((ROOT / "data/datasets").glob("*20260904/*inner_cases.jsonl")):
        inner, item = load(path)
        inner_cases = {row["case_id"] for row in inner}
        inner_subjects = {row["subject_id"] for row in inner}
        report["replications"].append({**item, "scans": len(inner_cases), "subjects": len(inner_subjects),
            "teacher_training_case_overlap": len(inner_cases & cases),
            "teacher_training_subject_overlap": len(inner_subjects & subjects),
            "overlapping_case_ids": sorted(inner_cases & cases)})
    assert len(report["replications"]) == 4
    output = ROOT / "archive/reviews/2026-09-05_30h_chain_audit/TEACHER_INNER_EXPOSURE.json"
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "overlap_scans": [r["teacher_training_case_overlap"] for r in report["replications"]]}))


if __name__ == "__main__":
    main()
