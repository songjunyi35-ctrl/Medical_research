#!/usr/bin/env python3
"""Paired fit864 AdamW/Lookahead5 training using the frozen Full972 trainer core."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import WeightedRandomSampler as TorchWeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lookahead_optimizer import LookaheadOptimizer  # noqa: E402
from scripts.train_t1ce_wt_capg3d import write_json_atomic  # noqa: E402
import scripts.train_t1ce_wt_lawa6_stage_a as base  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_lookahead5_stage_a"
CHECKPOINT_SCHEMA = SCHEMA + ".checkpoint"
ARMS = ("adamw_control", "lookahead5_candidate")
SEED = 20261071
FIT_SHA256 = "ee16a66d9a7efb8b3c5a14536d03bba7e5eb0eb984e4f333b314ec61570439b3"
SPLIT_SHA256 = "adf5d61280402d167d2eaa92f09b355d91385ee51d5b0f7e989cc7b7b21a096b"
IDEA_SHA256 = "b71bd5c2a018ee638d3517fa179f42fbe21fa2d197336f7a451888bbec814f36"
PREREG_SHA256 = "cbbdcfdb8a3424e573dbec531edad84b8344abc8827113128eaa4d34720807d9"
FORMAL_TOTAL_STEPS = 168 * 121

_ORIGINAL_ADAMW = torch.optim.AdamW
_ORIGINAL_AUGMENT = base.augment_batch
_ORIGINAL_CHECKPOINT_PAYLOAD = base.checkpoint_payload
_AUDIT: dict[str, Any] = {"sampler_epoch_sha256": [], "first_augmented_batch": None}
_LOOKAHEAD: LookaheadOptimizer | None = None


class AuditedWeightedRandomSampler(TorchWeightedRandomSampler):
    def __iter__(self):
        indices = list(super().__iter__())
        _AUDIT["sampler_epoch_sha256"].append(base.tensor_sha256(torch.tensor(indices, dtype=torch.int64)))
        return iter(indices)


def audited_augment(image: torch.Tensor, label: torch.Tensor, *, max_shift: int):
    result_image, result_label = _ORIGINAL_AUGMENT(image, label, max_shift=max_shift)
    if _AUDIT["first_augmented_batch"] is None:
        _AUDIT["first_augmented_batch"] = {
            "image_sha256": base.tensor_sha256(result_image),
            "label_sha256": base.tensor_sha256(result_label),
        }
    return result_image, result_label


def configure_base(args: argparse.Namespace) -> None:
    global _LOOKAHEAD
    _AUDIT["sampler_epoch_sha256"] = []
    _AUDIT["first_augmented_batch"] = None
    _LOOKAHEAD = None
    base.SCHEMA = SCHEMA
    base.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    base.ARMS = ARMS
    base.ARM_SEED = {arm: SEED for arm in ARMS}
    base.FIT_SHA256 = FIT_SHA256
    base.SPLIT_SHA256 = SPLIT_SHA256
    base.IDEA_SHA256 = IDEA_SHA256
    base.PREREG_SHA256 = PREREG_SHA256
    base.FIT_CASES = 864
    base.FIT_SUBJECTS = 392
    base.__file__ = str(Path(__file__).resolve())
    base.WeightedRandomSampler = AuditedWeightedRandomSampler
    base.augment_batch = audited_augment

    planned_total_steps = 6 if args.smoke else FORMAL_TOTAL_STEPS
    optimizer_metadata = {
        "adamw_control": {"outer": None, "deployment_weights": "AdamW fast weights at step20328"},
        "lookahead5_candidate": {
            "outer": "Lookahead",
            "k": 5,
            "alpha": 0.5,
            "total_steps": planned_total_steps,
            "expected_slow_updates": planned_total_steps // 5,
            "trailing_fast_steps": planned_total_steps % 5,
            "deployment_weights": "latest slow weights; trailing fast steps are not deployed",
        },
    }[args.arm]

    def tagged_checkpoint_payload(**kwargs):
        payload = _ORIGINAL_CHECKPOINT_PAYLOAD(**kwargs)
        payload["optimizer_factor"] = optimizer_metadata
        return payload

    base.checkpoint_payload = tagged_checkpoint_payload
    if args.arm == "lookahead5_candidate":
        total_steps = planned_total_steps

        def factory(parameters, *factory_args, **factory_kwargs):
            global _LOOKAHEAD
            inner = _ORIGINAL_ADAMW(parameters, *factory_args, **factory_kwargs)
            _LOOKAHEAD = LookaheadOptimizer(inner, k=5, alpha=0.5, total_steps=total_steps)
            return _LOOKAHEAD

        base.torch.optim.AdamW = factory
    else:
        base.torch.optim.AdamW = _ORIGINAL_ADAMW


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.arm not in ARMS or args.seed != SEED:
        raise ValueError("Lookahead Stage A arm/seed differs from preregistration")
    configure_base(args)
    report = base.run(args)
    expected_epochs = 6 if args.smoke else 168
    expected_steps = 6 if args.smoke else FORMAL_TOTAL_STEPS
    optimizer_audit: dict[str, Any]
    if args.arm == "lookahead5_candidate":
        if _LOOKAHEAD is None:
            raise RuntimeError("Lookahead optimizer was not constructed")
        optimizer_audit = _LOOKAHEAD.audit()
        if not optimizer_audit["all_counts_exact"]:
            raise RuntimeError("Lookahead count/final-slow audit failed")
    else:
        optimizer_audit = {
            "optimizer": "AdamW",
            "total_steps": expected_steps,
            "fast_steps": expected_steps,
            "slow_updates": 0,
            "final_slow_loaded": False,
            "all_counts_exact": True,
        }
    pairing = {
        "schema": SCHEMA + ".pairing_audit",
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm,
        "seed": args.seed,
        "smoke": args.smoke,
        "initial_state_sha256": report["initial_state_sha256"],
        "sampling_weight_sha256": report["sampling_contract"]["native_weight_sha256"],
        "sampler_epoch_sha256": _AUDIT["sampler_epoch_sha256"],
        "sampler_epoch_count": len(_AUDIT["sampler_epoch_sha256"]),
        "first_augmented_batch": _AUDIT["first_augmented_batch"],
        "optimizer_audit": optimizer_audit,
        "gates": {
            "sampler_epoch_count_exact": len(_AUDIT["sampler_epoch_sha256"]) == expected_epochs,
            "first_augmented_batch_recorded": _AUDIT["first_augmented_batch"] is not None,
            "optimizer_counts_exact": optimizer_audit["all_counts_exact"] is True,
            "train_report_gates_passed": report["all_gates_passed"] is True,
        },
        "access": {"train_only_inner": False, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False},
    }
    pairing["all_gates_passed"] = all(pairing["gates"].values())
    write_json_atomic(args.output_dir / "PAIRING_AUDIT.json", pairing)
    if not pairing["all_gates_passed"]:
        raise RuntimeError("Lookahead Stage A pairing-side audit failed")
    return report


def main() -> None:
    record = ROOT / "archive/research_records/2026-09-02_full972_lookahead5_stability"
    split = ROOT / "data/datasets/brats2024_t1ce_lookahead5_inner_seed20261070_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--fit-case-ids", type=Path, default=split / "fit_cases.jsonl")
    parser.add_argument("--split-manifest", type=Path, default=split / "split_manifest.json")
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
    run(args)


if __name__ == "__main__":
    main()
