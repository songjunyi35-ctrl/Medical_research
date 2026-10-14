#!/usr/bin/env python3
"""Validation-only reload replay using the trainer's legacy NumPy RNG stream.

The independent checkpoint evaluator intentionally derives a separate PCG64
stream for each case.  Training validation instead resets NumPy's legacy
MT19937 stream once, then consumes it across case-sorted validation rows.  This
small wrapper keeps the frozen evaluator implementation unchanged while
replaying that trainer RNG contract on the frozen 108-case validation split.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import evaluate_sammed3d_t1ce_wt_checkpoint as evaluator  # noqa: E402


VALIDATION_MANIFEST = (
    ROOT
    / "data/datasets/"
    "brats2024_sammed3d_t1ce_wt_inner90_10_seed20260723_v1/"
    "validation_t1ce_wt.jsonl"
)
VALIDATION_MANIFEST_SHA256 = (
    "fc750ab15266b8a53c6d9a7ff62bd043b10cc21733e9b68354a2a989fd1f488c"
)
SELECTED_CHECKPOINT = (
    ROOT
    / "workspace/BraTS/sammed3d_t1ce_wt/"
    "fv_lora_r4_aug_train972_val108_e50_s20260725/"
    "sam_model_val_dice_best.pth"
)
SELECTED_CHECKPOINT_SHA256 = (
    "c1d1bd79791aa7beeffc541095472dec968630cc70ea7967128c86552bd7f1ec"
)


class LegacyNumpyStream:
    """Expose Generator.integers while consuming an MT19937 RandomState."""

    def __init__(self, seed: int) -> None:
        self._random_state = np.random.RandomState(seed)

    def integers(self, high: int) -> int:
        return int(self._random_state.randint(high))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=SELECTED_CHECKPOINT)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=SELECTED_CHECKPOINT_SHA256,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--output", required=True, type=Path)
    parsed = parser.parse_args()

    args = argparse.Namespace(
        manifest=VALIDATION_MANIFEST,
        expected_manifest_sha256=VALIDATION_MANIFEST_SHA256,
        expected_split="validation",
        max_cases=None,
        checkpoint=parsed.checkpoint,
        expected_checkpoint_sha256=parsed.expected_checkpoint_sha256,
        spatial_preprocessing=evaluator.WHOLE_RESIZE,
        device=parsed.device,
        seed=parsed.seed,
        output=parsed.output,
    )

    stream = LegacyNumpyStream(parsed.seed)
    original_default_rng = evaluator.np.random.default_rng
    evaluator.np.random.default_rng = lambda _seed=None: stream
    try:
        report = evaluator.evaluate(args)
    finally:
        evaluator.np.random.default_rng = original_default_rng

    report["rng"] = {
        "mode": "trainer_legacy_numpy_stream",
        "algorithm": "MT19937 RandomState",
        "seed": parsed.seed,
        "scope": "one stream consumed over case-sorted validation rows",
    }
    report["purpose"] = (
        "validation-only diagnostic replay of trainer epoch validation RNG; "
        "not an independent case-seeded estimate"
    )
    evaluator.write_json_atomic(parsed.output.resolve(), report)
    print(evaluator.json.dumps(report["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
