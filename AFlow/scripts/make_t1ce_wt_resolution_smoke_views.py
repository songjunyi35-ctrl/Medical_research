#!/usr/bin/env python3
"""Derive tiny, diagnostic-only case views from a frozen resolution split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.train_t1ce_wt_capg3d import sha256_file


def read_rows(path: Path, expected_role: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or any(row.get("role") != expected_role for row in rows):
        raise ValueError(f"unexpected role in {path}")
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--inner-tune", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit-cases", type=int, default=8)
    parser.add_argument("--inner-tune-cases", type=int, default=4)
    args = parser.parse_args()
    for path in (args.fit, args.inner_tune, args.output_dir):
        lowered = str(path.resolve()).lower()
        if any(token in lowered for token in ("selector", "confirmation", "sealed", "test270")):
            raise ValueError(f"forbidden data role in path: {path}")
    fit = read_rows(args.fit, "fit")[: args.fit_cases]
    tune = read_rows(args.inner_tune, "inner_tune")[: args.inner_tune_cases]
    if len(fit) != args.fit_cases or len(tune) != args.inner_tune_cases:
        raise ValueError("frozen split is too small for requested smoke views")
    if {row["subject_id"] for row in fit} & {row["subject_id"] for row in tune}:
        raise ValueError("smoke views have subject leakage")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fit_path = args.output_dir / "smoke_fit_cases.jsonl"
    tune_path = args.output_dir / "smoke_inner_tune_cases.jsonl"
    write_rows(fit_path, fit)
    write_rows(tune_path, tune)
    manifest = {
        "schema": "aflow.brats2024.t1ce_wt_resolution_smoke_views",
        "diagnostic_only": True,
        "fit": {"path": fit_path.name, "cases": len(fit), "sha256": sha256_file(fit_path)},
        "inner_tune": {
            "path": tune_path.name,
            "cases": len(tune),
            "sha256": sha256_file(tune_path),
        },
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
