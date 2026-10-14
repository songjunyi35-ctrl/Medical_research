#!/usr/bin/env python3
"""Paired fit864 baseline/Gamma training using the frozen Full972 trainer core."""
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

from scripts.gamma_augmentation import RetainedStatsGamma  # noqa: E402
from scripts.train_t1ce_wt_capg3d import write_json_atomic  # noqa: E402
import scripts.train_t1ce_wt_lawa6_stage_a as base  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_gamma_aug_stage_a"
CHECKPOINT_SCHEMA = SCHEMA + ".checkpoint"
ARMS = ("baseline_control", "gamma_candidate")
SEED = 20261081
GAMMA_SEED = 20261181
FIT_SHA256 = "f8efe7eb10f1b66b7bd3fbfc3b5c1286e63594c6e8f51be3c89469f6848bf8bb"
SPLIT_SHA256 = "3287acd890e64aea5ad301ff7f7bf525f957845486249f0ad8aeac8e8e5613fc"
IDEA_SHA256 = "a0935b570590c14211b44457d3b41664c21dd45b14732b861f57b84d9dd0f9fb"
PREREG_SHA256 = "061238cfc064a9ff57c9b4134e68ebf3849ca9ba9a8b392a014c0cc5e4c67b13"

_ORIGINAL_AUGMENT = base.augment_batch
_ORIGINAL_CHECKPOINT_PAYLOAD = base.checkpoint_payload
_AUDIT: dict[str, Any] = {"sampler_epoch_sha256": [], "first_pre_gamma_batch": None, "first_label_unchanged": None, "gamma_global_rng_unchanged": None}
_GAMMA: RetainedStatsGamma | None = None
_ARM = ""


class AuditedWeightedRandomSampler(TorchWeightedRandomSampler):
    def __iter__(self):
        indices = list(super().__iter__())
        _AUDIT["sampler_epoch_sha256"].append(base.tensor_sha256(torch.tensor(indices, dtype=torch.int64)))
        return iter(indices)


def audited_augment(image: torch.Tensor, label: torch.Tensor, *, max_shift: int):
    global _GAMMA
    base_image, base_label = _ORIGINAL_AUGMENT(image, label, max_shift=max_shift)
    if _AUDIT["first_pre_gamma_batch"] is None:
        _AUDIT["first_pre_gamma_batch"] = {"image_sha256": base.tensor_sha256(base_image), "label_sha256": base.tensor_sha256(base_label)}
    if _ARM != "gamma_candidate":
        return base_image, base_label
    if _GAMMA is None:
        _GAMMA = RetainedStatsGamma(base_image.device, seed=GAMMA_SEED)
    first_call = _GAMMA.total_samples == 0
    if first_call:
        cpu_before = torch.random.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state(base_image.device).clone()
        label_before = base.tensor_sha256(base_label)
    result_image, result_label = _GAMMA(base_image, base_label)
    if first_call:
        _AUDIT["gamma_global_rng_unchanged"] = bool(
            torch.equal(cpu_before, torch.random.get_rng_state())
            and torch.equal(cuda_before, torch.cuda.get_rng_state(base_image.device))
        )
        _AUDIT["first_label_unchanged"] = label_before == base.tensor_sha256(result_label)
    return result_image, result_label


def configure_base(args: argparse.Namespace) -> None:
    global _GAMMA, _ARM
    _ARM = args.arm
    _GAMMA = None
    _AUDIT.update({"sampler_epoch_sha256": [], "first_pre_gamma_batch": None, "first_label_unchanged": None, "gamma_global_rng_unchanged": None})
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
    factor = {
        "baseline_control": {"gamma": False},
        "gamma_candidate": {"gamma": True, "probability": 0.3, "range": [0.7, 1.5], "invert": False, "retain_mean_population_std": True, "generator_seed": GAMMA_SEED, "position": "after existing augmentation"},
    }[args.arm]

    def tagged_checkpoint_payload(**kwargs):
        payload = _ORIGINAL_CHECKPOINT_PAYLOAD(**kwargs)
        payload["augmentation_factor"] = factor
        return payload

    base.checkpoint_payload = tagged_checkpoint_payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.arm not in ARMS or args.seed != SEED:
        raise ValueError("Gamma Stage A arm/seed differs from preregistration")
    configure_base(args)
    report = base.run(args)
    expected_epochs = 6 if args.smoke else 168
    gamma_audit = None if _GAMMA is None else _GAMMA.audit()
    gamma_gates = {"applicable": args.arm == "gamma_candidate"}
    if args.arm == "gamma_candidate":
        assert gamma_audit is not None
        fraction_ok = 0.0 < gamma_audit["applied_fraction"] < 1.0 if args.smoke else 0.25 <= gamma_audit["applied_fraction"] <= 0.35
        gamma_gates.update({
            "applied_fraction_ok": fraction_ok,
            "gamma_range_ok": 0.7 <= gamma_audit["gamma_min"] < 1.0 or 1.0 < gamma_audit["gamma_min"] <= 1.5,
            "gamma_max_range_ok": 0.7 <= gamma_audit["gamma_max"] < 1.0 or 1.0 < gamma_audit["gamma_max"] <= 1.5,
            "statistics_preserved": gamma_audit["max_mean_abs_error"] <= 1e-4 and gamma_audit["max_population_std_abs_error"] <= 1e-4,
            "first_label_unchanged": _AUDIT["first_label_unchanged"] is True,
            "global_rng_unchanged_by_gamma": _AUDIT["gamma_global_rng_unchanged"] is True,
        })
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
        "first_pre_gamma_batch": _AUDIT["first_pre_gamma_batch"],
        "gamma_audit": gamma_audit,
        "gamma_gates": gamma_gates,
        "gates": {
            "sampler_epoch_count_exact": len(_AUDIT["sampler_epoch_sha256"]) == expected_epochs,
            "first_pre_gamma_batch_recorded": _AUDIT["first_pre_gamma_batch"] is not None,
            "gamma_contract_passed": all(value for key, value in gamma_gates.items() if key != "applicable") if args.arm == "gamma_candidate" else True,
            "train_report_gates_passed": report["all_gates_passed"] is True,
        },
        "access": {"train_only_inner": False, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False},
    }
    pairing["all_gates_passed"] = all(pairing["gates"].values())
    write_json_atomic(args.output_dir / "PAIRING_AUDIT.json", pairing)
    if not pairing["all_gates_passed"]:
        raise RuntimeError("Gamma Stage A pairing-side audit failed")
    return report


def main() -> None:
    record = ROOT / "archive/research_records/2026-09-02_full972_gamma_aug_stability"
    split = ROOT / "data/datasets/brats2024_t1ce_gamma_aug_inner_seed20261080_v1"
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
