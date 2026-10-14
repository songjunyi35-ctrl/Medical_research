#!/usr/bin/env python3
"""V3 CLI wrapper: fixed architecture fields plus final JSON print dependency."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import scripts.train_t1ce_wt_plain_voxel_kd_continuation as base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("control", "kd"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--train-case-ids", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--ema-teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260846)
    parser.add_argument("--start-global-epoch", type=int, default=168)
    parser.add_argument("--continuation-epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=base.TEMPERATURE)
    parser.add_argument("--kd-weight", type=float, required=True)
    parser.add_argument("--init-filters", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()
    if args.init_filters != 16 or args.dropout != 0.1:
        raise ValueError("frozen architecture requires init_filters16/dropout0.1")
    base.json = json
    base.run(args)


if __name__ == "__main__":
    main()
