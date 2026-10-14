#!/usr/bin/env python3
"""Authorized paired Full972 baseline/Gamma final using the frozen Stage A implementation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_t1ce_wt_capg3d import sha256_file  # noqa: E402
import scripts.train_t1ce_wt_gamma_aug_stage_a as stage_a  # noqa: E402

SCHEMA = "aflow.sammed3d.t1ce_wt_gamma_aug_stage_b_final"
CHECKPOINT_SCHEMA = SCHEMA + ".checkpoint"
SEED = 20261082
GAMMA_SEED = 20261182
FULL972_SHA256 = "6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65"
STAGE_A_DECISION_SHA256 = "97bfd6b405737168debf327bbba576a29886600e224e23a7cba1b35df9fe210b"


def validate_decision(path: Path) -> dict:
    if sha256_file(path) != STAGE_A_DECISION_SHA256:
        raise ValueError("Gamma Stage A decision SHA mismatch")
    value = json.loads(path.read_text())
    if value.get("schema") != "aflow.brats.t1ce_wt_gamma_aug_stage_a_decision" or value.get("status") != "passed" or value.get("full972_final_authorized") is not True or value.get("gates") != {"passed": 6, "failed": 0, "all_gates_passed": True} or value.get("access") != {"train_only_inner": True, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False}:
        raise ValueError("Gamma Stage A decision does not authorize final")
    return value


def configure_stage_b(wrapper: Path) -> None:
    original_configure = stage_a.configure_base
    stage_a.SCHEMA = SCHEMA
    stage_a.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    stage_a.SEED = SEED
    stage_a.GAMMA_SEED = GAMMA_SEED
    stage_a.FIT_SHA256 = FULL972_SHA256
    stage_a.SPLIT_SHA256 = STAGE_A_DECISION_SHA256
    stage_a.__file__ = str(wrapper.resolve())

    def configure(args):
        original_configure(args)
        stage_a.base.FIT_CASES = 972
        stage_a.base.FIT_SUBJECTS = 444

    stage_a.configure_base = configure


def main() -> None:
    record = ROOT / "archive/research_records/2026-09-02_full972_gamma_aug_stability"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=stage_a.ARMS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--fit-case-ids", type=Path, default=ROOT / "data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl")
    parser.add_argument("--stage-a-decision", dest="split_manifest", type=Path, default=record / "STAGE_A_DECISION.json")
    parser.add_argument("--idea", type=Path, default=record / "IDEA.md")
    parser.add_argument("--prereg", type=Path, default=record / "PREREG.md")
    parser.add_argument("--native-artifact", type=Path, default=ROOT / "archive/research_records/2026-08-21_voxel_kd_data972_native_sampling/NATIVE_WT_VOLUMES.json")
    parser.add_argument("--historical-teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--ema-teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--stop-epoch", type=int, default=168)
    parser.add_argument("--schedule-total-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--base-lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-fit-cases", type=int, default=32)
    args = parser.parse_args()
    validate_decision(args.split_manifest)
    configure_stage_b(Path(__file__))
    stage_a.run(args)


if __name__ == "__main__": main()
