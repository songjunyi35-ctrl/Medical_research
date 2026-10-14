#!/usr/bin/env python3
"""Create the frozen paired statistical report for Domain-SAGE 3D."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.train_sammed3d_t1ce_wt_residual_refiner import sha256_file, write_json_atomic


SCHEMA = "aflow.sammed3d.t1ce_wt_domain_sage_pair_analysis"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def subject_id(case_id: str) -> str:
    return case_id.rsplit("-", 1)[0]


def run(args: argparse.Namespace) -> dict[str, Any]:
    plain_reload = load_json(args.plain_reload)
    domain_reload = load_json(args.domain_reload)
    plain_done = load_json(args.plain_train_done)
    domain_done = load_json(args.domain_train_done)
    for payload in (plain_reload, domain_reload, plain_done, domain_done):
        if payload.get("test_access") is not False:
            raise ValueError("all inputs must explicitly deny test access")
    plain_rows = plain_reload["validation"]["per_case"]
    domain_rows = domain_reload["validation"]["per_case"]
    plain_ids = [row["case_id"] for row in plain_rows]
    domain_ids = [row["case_id"] for row in domain_rows]
    if plain_ids != domain_ids:
        raise ValueError("plain and domain case order differs")
    plain_values = np.asarray([row["final_dsc"] for row in plain_rows], dtype=np.float64)
    domain_values = np.asarray([row["final_dsc"] for row in domain_rows], dtype=np.float64)
    differences = domain_values - plain_values

    grouped: dict[str, list[float]] = defaultdict(list)
    plain_grouped: dict[str, list[float]] = defaultdict(list)
    domain_grouped: dict[str, list[float]] = defaultdict(list)
    for case_id, plain, domain, difference in zip(
        plain_ids, plain_values, domain_values, differences, strict=True
    ):
        key = subject_id(case_id)
        grouped[key].append(float(difference))
        plain_grouped[key].append(float(plain))
        domain_grouped[key].append(float(domain))
    subjects = sorted(grouped)
    subject_differences = np.asarray(
        [np.mean(grouped[key]) for key in subjects], dtype=np.float64
    )
    plain_subject = np.asarray(
        [np.mean(plain_grouped[key]) for key in subjects], dtype=np.float64
    )
    domain_subject = np.asarray(
        [np.mean(domain_grouped[key]) for key in subjects], dtype=np.float64
    )
    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap_parts: list[np.ndarray] = []
    remaining = args.bootstrap_resamples
    while remaining:
        count = min(remaining, 5000)
        indices = rng.integers(
            0, len(subjects), size=(count, len(subjects)), endpoint=False
        )
        bootstrap_parts.append(subject_differences[indices].mean(axis=1))
        remaining -= count
    bootstrap = np.concatenate(bootstrap_parts)
    ci = np.quantile(bootstrap, [0.025, 0.975])

    domains = np.asarray(
        [row["pseudo_domain"] for row in domain_rows], dtype=np.int64
    )
    domain_subgroups = []
    for domain in sorted(set(domains.tolist())):
        mask = domains == domain
        domain_subgroups.append(
            {
                "pseudo_domain": int(domain),
                "cases": int(mask.sum()),
                "plain_scan_dsc": float(plain_values[mask].mean()),
                "domain_sage_scan_dsc": float(domain_values[mask].mean()),
                "paired_difference": float(differences[mask].mean()),
            }
        )

    scan_difference = float(differences.mean())
    subject_difference = float(subject_differences.mean())
    pass_gate = bool(
        scan_difference >= args.required_scan_gain and subject_difference >= 0.0
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "comparison": "Domain-SAGE minus matched plain SAGE",
        "cases": len(plain_ids),
        "subjects": len(subjects),
        "scan": {
            "plain_dsc": float(plain_values.mean()),
            "domain_sage_dsc": float(domain_values.mean()),
            "paired_difference": scan_difference,
            "improved_cases": int(np.sum(differences > 0)),
            "worsened_cases": int(np.sum(differences < 0)),
            "tied_cases": int(np.sum(differences == 0)),
        },
        "subject": {
            "plain_dsc": float(plain_subject.mean()),
            "domain_sage_dsc": float(domain_subject.mean()),
            "paired_difference": subject_difference,
            "bootstrap_unit": "subject after within-subject scan mean",
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_resamples": args.bootstrap_resamples,
            "ci95": [float(ci[0]), float(ci[1])],
        },
        "preregistered_gate": {
            "required_scan_gain": args.required_scan_gain,
            "requires_nonnegative_subject_gain": True,
            "passed": pass_gate,
            "decision": "advance to second seed" if pass_gate else "negative ablation; do not tune",
        },
        "pseudo_domain_subgroups_descriptive_only": domain_subgroups,
        "training": {
            "plain": {
                "best_epoch": plain_done["selection"]["epoch"],
                "completed_epochs": len(plain_done["history"]),
                "elapsed_seconds": plain_done["elapsed_seconds"],
            },
            "domain": {
                "best_epoch": domain_done["selection"]["epoch"],
                "completed_epochs": len(domain_done["history"]),
                "elapsed_seconds": domain_done["elapsed_seconds"],
                "best_validation_domain_accuracy": domain_done["history"][
                    domain_done["selection"]["epoch"] - 1
                ]["validation"]["domain_accuracy"],
                "final_validation_domain_accuracy": domain_done["history"][-1][
                    "validation"
                ]["domain_accuracy"],
            },
        },
        "evidence": {
            "plain_train_done": {
                "path": str(args.plain_train_done.resolve()),
                "sha256": sha256_file(args.plain_train_done),
            },
            "domain_train_done": {
                "path": str(args.domain_train_done.resolve()),
                "sha256": sha256_file(args.domain_train_done),
            },
            "plain_reload": {
                "path": str(args.plain_reload.resolve()),
                "sha256": sha256_file(args.plain_reload),
            },
            "domain_reload": {
                "path": str(args.domain_reload.resolve()),
                "sha256": sha256_file(args.domain_reload),
            },
        },
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_atomic(args.output, report)
    print(
        f"DOMAIN_SAGE_ANALYSIS_DONE scan_diff={scan_difference:+.9f} "
        f"subject_diff={subject_difference:+.9f} ci=[{ci[0]:+.9f},{ci[1]:+.9f}] "
        f"passed={pass_gate}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain-train-done", type=Path, required=True)
    parser.add_argument("--domain-train-done", type=Path, required=True)
    parser.add_argument("--plain-reload", type=Path, required=True)
    parser.add_argument("--domain-reload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260818)
    parser.add_argument("--bootstrap-resamples", type=int, default=100000)
    parser.add_argument("--required-scan-gain", type=float, default=0.001)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
