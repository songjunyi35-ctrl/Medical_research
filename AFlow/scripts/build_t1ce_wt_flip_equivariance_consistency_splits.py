#!/usr/bin/env python3
"""Build two mutually fresh train-only splits for flip-consistency training."""
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


SCHEMA = "aflow.brats2024.t1ce_wt_flip_equivariance_consistency_splits"
SOURCE_SHA256 = "6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65"
IDEA_SHA256 = "a71b75ff8c7b775fc0606141c5b1c6258aeaf4670ec990c819dedb13f85e1175"
PREREG_SHA256 = "7ae74cd1891a4a67efa064ca2dffd4560064657661fd4d824902aff02bac5809"
OLD_INNER_SHA256 = {
    "a": "a96abf516ae8f69dd02bc12906c0a381b7fafd991073fe500d3d49de36577e81",
    "b": "0d85256e43d1de2ee6b71587a210f193af8bb50be2a9d378ab1f4b59ceee5b88",
}
SPLIT_SEEDS = {"c": 20261211, "d": 20261213}
TRAIN_SEEDS = {"c": 20261212, "d": 20261214}
BOOTSTRAP_SEEDS = {"c": 20261215, "d": 20261216}


def read_source(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    cases = [str(row.get("case_id", "")) for row in rows]
    subjects = [str(row.get("subject_id", "")) for row in rows]
    if len(rows) != 972 or len(set(cases)) != 972 or len(set(subjects)) != 444:
        raise ValueError("source must contain972 unique cases/444 subjects")
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("source contains non-train role")
    return rows


def read_old_subjects(path: Path, expected_sha256: str) -> set[str]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"old inner SHA mismatch: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 108:
        raise ValueError("old inner must contain108 scans")
    return {str(row["subject_id"]) for row in rows}


def choose_fresh_subjects(
    counts: dict[str, int], *, excluded: set[str], seed: int
) -> set[str]:
    eligible = {subject: count for subject, count in counts.items() if subject not in excluded}
    chosen = choose_exact_validation_subjects(eligible, seed=seed, target_cases=108)
    if chosen & excluded or sum(counts[subject] for subject in chosen) != 108:
        raise RuntimeError("fresh subject selection violated exclusion/count")
    return chosen


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
    rows = read_source(args.source.resolve())
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subject[str(row["subject_id"])].append(row)
    counts = {subject: len(items) for subject, items in by_subject.items()}
    old_a = read_old_subjects(args.old_inner_a.resolve(), OLD_INNER_SHA256["a"])
    old_b = read_old_subjects(args.old_inner_b.resolve(), OLD_INNER_SHA256["b"])
    old_union = old_a | old_b
    chosen_c = choose_fresh_subjects(counts, excluded=old_union, seed=SPLIT_SEEDS["c"])
    chosen_d = choose_fresh_subjects(counts, excluded=old_union | chosen_c, seed=SPLIT_SEEDS["d"])
    if chosen_c & chosen_d:
        raise RuntimeError("C/D inner subjects overlap")

    output.mkdir(parents=True, exist_ok=False)
    records: dict[str, Any] = {}
    chosen_by_rep = {"c": chosen_c, "d": chosen_d}
    for replicate in ("c", "d"):
        chosen = chosen_by_rep[replicate]
        inner = [row for row in rows if str(row["subject_id"]) in chosen]
        fit = [row for row in rows if str(row["subject_id"]) not in chosen]
        fit_cases = {str(row["case_id"]) for row in fit}
        inner_cases = {str(row["case_id"]) for row in inner}
        fit_subjects = {str(row["subject_id"]) for row in fit}
        inner_subjects = {str(row["subject_id"]) for row in inner}
        if len(fit) != 864 or len(inner) != 108:
            raise RuntimeError(f"replicate {replicate} is not864/108")
        if fit_cases & inner_cases or fit_subjects & inner_subjects:
            raise RuntimeError(f"replicate {replicate} leaks case/subject")
        if inner_subjects & old_union:
            raise RuntimeError(f"replicate {replicate} reuses an old inner subject")
        fit_path = output / f"replicate_{replicate}_fit_cases.jsonl"
        inner_path = output / f"replicate_{replicate}_inner_cases.jsonl"
        write_rows(fit_path, fit, "flip_equivariance_consistency_fit", replicate)
        write_rows(inner_path, inner, "flip_equivariance_consistency_inner", replicate)
        records[replicate] = {
            "split_seed": SPLIT_SEEDS[replicate],
            "train_seed": TRAIN_SEEDS[replicate],
            "bootstrap_seed": BOOTSTRAP_SEEDS[replicate],
            "fit": summarize(fit_path, fit),
            "inner": summarize(inner_path, inner),
            "checks": {"case_overlap": 0, "subject_overlap": 0, "union_cases": 972, "old_inner_subject_overlap": 0},
        }
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "frozen_before_implementation_model_training_or_inner_access",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(args.source.resolve()), "sha256": SOURCE_SHA256, "cases": 972, "subjects": 444},
        "documents": {"idea_sha256": IDEA_SHA256, "prereg_sha256": PREREG_SHA256},
        "excluded_old_inners": {
            "a": {"path": str(args.old_inner_a.resolve()), "sha256": OLD_INNER_SHA256["a"], "subjects": len(old_a)},
            "b": {"path": str(args.old_inner_b.resolve()), "sha256": OLD_INNER_SHA256["b"], "subjects": len(old_b)},
            "union_subjects": len(old_union),
        },
        "replications": records,
        "cross_replication": {"inner_case_overlap": 0, "inner_subject_overlap": 0},
        "access": {"train_only_inner_metrics": False, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False, "ucsd_external75": False},
    }
    (output / "split_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    record = ROOT / "archive/research_records/2026-09-04_flip_equivariance_consistency_stability"
    old = ROOT / "data/datasets/brats2024_t1ce_flip8_tta_replications_20260904"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--idea", type=Path, default=record / "IDEA.md")
    parser.add_argument("--prereg", type=Path, default=record / "PREREG.md")
    parser.add_argument("--old-inner-a", type=Path, default=old / "replicate_a_inner_cases.jsonl")
    parser.add_argument("--old-inner-b", type=Path, default=old / "replicate_b_inner_cases.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
