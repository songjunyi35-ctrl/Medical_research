#!/usr/bin/env python3
"""Build two fresh train-only splits for direct-locator Flip8 TTA."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_sammed3d_t1ce_wt_inner_split import choose_exact_validation_subjects  # noqa: E402
from scripts.train_t1ce_wt_capg3d import sha256_file  # noqa: E402


SCHEMA = "aflow.brats2024.t1ce_wt_flip8_tta_replication_splits"
SOURCE_SHA256 = "6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65"
IDEA_SHA256 = "87b53daa6ae91f22d00d16a5b6eef4294b1ee34d12e451060b14417cae4dd095"
PREREG_SHA256 = "cd2fe143f1ddb46062aece1e144c7bfcb04a9f728a85d71765faf7687ed699ec"
SPLIT_SEEDS = {"a": 20261201, "b": 20261203}
TRAIN_SEEDS = {"a": 20261202, "b": 20261204}
BOOTSTRAP_SEEDS = {"a": 20261205, "b": 20261206}


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    cases = [str(row.get("case_id", "")) for row in rows]
    subjects = [str(row.get("subject_id", "")) for row in rows]
    if len(rows) != 972 or len(set(cases)) != 972 or len(set(subjects)) != 444:
        raise ValueError("source must contain972 unique cases/444 subjects")
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("source contains non-train role")
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]], role: str, replicate: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({
                "case_id": str(row["case_id"]),
                "subject_id": str(row["subject_id"]),
                "role": role,
                "replicate": replicate,
            }, sort_keys=True) + "\n")


def summarize(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "cases": len(rows),
        "subjects": len({str(row["subject_id"]) for row in rows}),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    for path, expected in ((args.source, SOURCE_SHA256), (args.idea, IDEA_SHA256), (args.prereg, PREREG_SHA256)):
        if sha256_file(path) != expected:
            raise ValueError(f"frozen input SHA mismatch: {path}")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing split directory: {output}")
    rows = read_rows(args.source.resolve())
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subject[str(row["subject_id"])].append(row)
    counts = {subject: len(items) for subject, items in by_subject.items()}
    output.mkdir(parents=True, exist_ok=False)
    records: dict[str, Any] = {}
    inner_cases: dict[str, set[str]] = {}
    inner_subjects: dict[str, set[str]] = {}
    for replicate, seed in SPLIT_SEEDS.items():
        chosen = choose_exact_validation_subjects(counts, seed=seed, target_cases=108)
        inner = [row for row in rows if str(row["subject_id"]) in chosen]
        fit = [row for row in rows if str(row["subject_id"]) not in chosen]
        fit_case_set = {str(row["case_id"]) for row in fit}
        inner_case_set = {str(row["case_id"]) for row in inner}
        fit_subject_set = {str(row["subject_id"]) for row in fit}
        inner_subject_set = {str(row["subject_id"]) for row in inner}
        if len(fit) != 864 or len(inner) != 108:
            raise RuntimeError(f"replicate {replicate} is not864/108")
        if fit_case_set & inner_case_set or fit_subject_set & inner_subject_set:
            raise RuntimeError(f"replicate {replicate} leaks case/subject")
        fit_path = output / f"replicate_{replicate}_fit_cases.jsonl"
        inner_path = output / f"replicate_{replicate}_inner_cases.jsonl"
        write_rows(fit_path, fit, "flip8_tta_replication_fit", replicate)
        write_rows(inner_path, inner, "flip8_tta_replication_inner", replicate)
        records[replicate] = {
            "split_seed": seed,
            "train_seed": TRAIN_SEEDS[replicate],
            "bootstrap_seed": BOOTSTRAP_SEEDS[replicate],
            "fit": summarize(fit_path, fit),
            "inner": summarize(inner_path, inner),
            "checks": {"case_overlap": 0, "subject_overlap": 0, "union_cases": 972},
        }
        inner_cases[replicate] = inner_case_set
        inner_subjects[replicate] = inner_subject_set
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "frozen_before_implementation_model_training_or_inner_access",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(args.source.resolve()), "sha256": SOURCE_SHA256, "cases": 972, "subjects": 444},
        "documents": {"idea_sha256": IDEA_SHA256, "prereg_sha256": PREREG_SHA256},
        "replications": records,
        "cross_replication": {
            "inner_case_overlap": len(inner_cases["a"] & inner_cases["b"]),
            "inner_subject_overlap": len(inner_subjects["a"] & inner_subjects["b"]),
            "same_inner_case_set": inner_cases["a"] == inner_cases["b"],
            "same_inner_subject_set": inner_subjects["a"] == inner_subjects["b"],
        },
        "access": {"train_only_inner_metrics": False, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False, "ucsd_external75": False},
    }
    (output / "split_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--idea", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
