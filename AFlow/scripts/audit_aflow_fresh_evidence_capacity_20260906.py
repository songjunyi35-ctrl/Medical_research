#!/usr/bin/env python3
"""Audit remaining never-evaluated train-pool subjects under immutable role policy."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl"
SOURCE_SHA = "6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65"
SPLIT = ROOT / "archive/research_records/2026-09-05_no_n4_clean_teacher_replication/SPLIT.json"
SPLIT_SHA = "68975292cc10ab581657a97789050a28cc03d4a3bc3b0991682b7284bfa9c3f2"
OUT = ROOT / "archive/reviews/2026-09-06_fresh_evidence_capacity"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict]: return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main() -> None:
    output = OUT / "CAPACITY_AUDIT.json"
    if output.exists(): raise FileExistsError(output)
    if sha(SOURCE) != SOURCE_SHA or sha(SPLIT) != SPLIT_SHA: raise ValueError("frozen source/split SHA")
    source = rows(SOURCE); counts = Counter(x["subject_id"] for x in source)
    if len(source) != 972 or len(counts) != 444: raise ValueError("source count")
    split = json.loads(SPLIT.read_text())
    # Reconstruct the exact pre-E/F exclusion set from the paths frozen in SPLIT.json.
    excluded = set(); evidence = []
    for item in split["exclusions"]:
        path = ROOT / item["path"]
        if sha(path) != item["sha256"]: raise ValueError("exclusion SHA " + item["path"])
        values = rows(path); subjects = {str(x.get("subject_id") or "-".join(x["case_id"].split("-")[:3])) for x in values}
        if len(subjects) != item["subjects"]: raise ValueError("exclusion subject count")
        excluded |= subjects; evidence.append({"path": item["path"], "sha256": item["sha256"], "subjects": len(subjects)})
    initial_eligible = set(counts) - excluded
    if len(initial_eligible) != 38 or sum(counts[x] for x in initial_eligible) != 87: raise ValueError("initial eligible mismatch")
    used = set(); folds = {}
    for fold in ("e", "f"):
        record = split["folds"][fold]["inner"]
        path = ROOT / record["path"] if not Path(record["path"]).is_absolute() else Path(record["path"])
        if sha(path) != record["sha256"]: raise ValueError("fold inner SHA")
        values = rows(path); subjects = {x["subject_id"] for x in values}; cases = {x["case_id"] for x in values}
        if len(cases) != 42 or len(subjects) != record["subjects"] or not subjects <= initial_eligible or used & subjects: raise ValueError("fold ownership")
        used |= subjects; folds[fold] = {"cases": len(cases), "subjects": len(subjects), "sha256": record["sha256"], "metrics_accessed": True}
    remaining = initial_eligible - used
    result = {"schema": "aflow.fresh_evidence_capacity.audit", "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "exhausted_for_new_two_fold_inner_under_immutable_roles", "source": {"scans": 972, "subjects": 444, "sha256": SOURCE_SHA}, "pre_ef": {"excluded_subjects": len(excluded), "eligible_subjects": len(initial_eligible), "eligible_scans": sum(counts[x] for x in initial_eligible), "exclusion_evidence": evidence}, "revealed_ef": folds, "remaining_never_evaluated": {"subjects": len(remaining), "scans": sum(counts[x] for x in remaining), "subject_ids": sorted(remaining), "case_counts": {x: counts[x] for x in sorted(remaining)}}, "feasibility": {"new_42_scan_fold": False, "two_new_42_scan_folds": False, "minimum_prior_fresh_scale": {"scans_per_fold": 42, "subjects_per_fold_at_least": 12}, "role_reassignment_required_to_continue_internal": True}, "interpretation": "under immutable data roles, prior fit cases cannot silently become validation and revealed E/F cannot evaluate a new method; new internal evidence requires explicit objective/role change or new data", "image_or_label_voxels_accessed": False, "model_inference": False, "metrics_recomputed": False, "runner_sha256": sha(Path(__file__))}
    OUT.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "initial": result["pre_ef"], "folds": folds, "remaining": result["remaining_never_evaluated"], "output_sha256": sha(output)}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
