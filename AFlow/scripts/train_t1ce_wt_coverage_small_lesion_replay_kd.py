#!/usr/bin/env python3
"""Coverage-first small-lesion replay versus native replacement voxel-KD."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Sized

import torch
from torch.utils.data import Sampler
from torch.utils.data import WeightedRandomSampler as TorchWeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_t1ce_wt_plain_voxel_kd_data972 as core  # noqa: E402
import scripts.train_t1ce_wt_plain_voxel_kd_data972_native_sampling as native  # noqa: E402
from scripts.train_t1ce_wt_capg3d import sha256_file  # noqa: E402


BASE_WRITE_JSON_ATOMIC = core.write_json_atomic


SCHEMA = "aflow.sammed3d.t1ce_wt_coverage_small_lesion_replay_kd_train"
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_coverage_small_lesion_replay_kd_checkpoint"
ARMS = ("native_replacement_control", "coverage_replay_candidate")
ARM_COUNTS = {arm: (972, 444) for arm in ARMS}
ARM_MANIFEST_SHA256 = {
    arm: "6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65"
    for arm in ARMS
}

IDEA_SHA256 = "9a3bd06d40446db804648d7bf0db23d379266fae80e71fdb188493610565d6d6"
PREREG_SHA256 = "61643d8793455446c10128da00a2f796409a5bc3437b87654b6d645e40cd7d08"
PREREG_JSON_SHA256 = "4c46f17b64b7edcd6bfa2856a2c5c44a52e0bb413ee7d84d4e4e2a6ee7efb37a"
SAMPLER_SPEC_SHA256 = "57e71af8215dd148505e98aeaba5ffed1e7086fe968f1185cd84f89c3ac5eecb"
SAMPLER_AMENDMENT_SHA256 = "95435d1220d54cd3d728a0d6b540fd7ceba69a5c4dd023a8e4818e7bbe5a36a9"
SEED_AMENDMENT_SHA256 = "fec579ef84133d43f03ffa75464ce5310ee0c28b266d98f1d3466f0ec98a4e65"
REPORT_AMENDMENT_SHA256 = "a9066ac52f91b903ed5861415eb26b42846abe9a7bf5bf4c4f3ee55ab02f2e22"
WRITER_AMENDMENT_SHA256 = "8e911103c431864257055312ca34ff8cb9e8b88c999c4356e3b73055562d1bf5"

POPULATION = 972
REQUESTED_DRAWS = 756
CONSUMED_DRAWS = 752
COVERAGE_DRAWS = 680
REPLAY_CONSUMED = 72
REPLAY_DROPPED = 4
COVERAGE_SEED_OFFSET = 1001
REPLAY_SEED_OFFSET = 2003
ORDER_SEED_OFFSET = 3007


def tensor_index_sha256(values: list[int]) -> str:
    tensor = torch.tensor(values, dtype=torch.int64)
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


class CoverageSmallLesionReplaySampler(Sampler[int]):
    """Persistent coverage queue plus a low-budget weighted replay tail."""

    def __init__(
        self,
        weights: torch.Tensor,
        *,
        num_samples: int,
        replacement: bool,
        seed: int,
    ) -> None:
        weights = torch.as_tensor(weights, dtype=torch.float64, device="cpu").contiguous()
        if weights.ndim != 1 or weights.numel() != POPULATION:
            raise ValueError(f"candidate weights must contain exactly {POPULATION} values")
        if not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
            raise ValueError("candidate weights must be finite and strictly positive")
        if num_samples != REQUESTED_DRAWS:
            raise ValueError(f"candidate num_samples must be {REQUESTED_DRAWS}")
        if replacement is not True:
            raise ValueError("candidate replay contract requires replacement=True input")
        self.weights = weights
        self.num_samples = num_samples
        self.seed = int(seed)
        self.coverage_generator = torch.Generator().manual_seed(
            self.seed + COVERAGE_SEED_OFFSET
        )
        self.replay_generator = torch.Generator().manual_seed(
            self.seed + REPLAY_SEED_OFFSET
        )
        self.order_generator = torch.Generator().manual_seed(
            self.seed + ORDER_SEED_OFFSET
        )
        self._previous_coverage: list[int] | None = None
        self.epoch_audits: list[dict[str, Any]] = []
        self.epoch_coverage_indices: list[list[int]] = []

    def __len__(self) -> int:
        return self.num_samples

    def _take_coverage(self) -> tuple[list[int], int | None, int | None]:
        if self._previous_coverage is None:
            selected = torch.randperm(
                POPULATION, generator=self.coverage_generator
            )[:COVERAGE_DRAWS].tolist()
            retained_count, complement_count = None, None
        else:
            previous = set(self._previous_coverage)
            complement = [index for index in range(POPULATION) if index not in previous]
            if len(complement) != POPULATION - COVERAGE_DRAWS:
                raise RuntimeError("previous coverage set has the wrong cardinality")
            previous_values = torch.tensor(self._previous_coverage, dtype=torch.int64)
            retained_order = torch.randperm(
                COVERAGE_DRAWS, generator=self.coverage_generator
            )[:388]
            retained = previous_values[retained_order].tolist()
            selected = complement + retained
            shuffle = torch.randperm(
                COVERAGE_DRAWS, generator=self.coverage_generator
            ).tolist()
            selected = [selected[index] for index in shuffle]
            retained_count, complement_count = len(retained), len(complement)
        if len(selected) != COVERAGE_DRAWS or len(set(selected)) != COVERAGE_DRAWS:
            raise RuntimeError("coverage transition did not produce 680 unique indices")
        self._previous_coverage = selected
        return selected, retained_count, complement_count

    def __iter__(self) -> Iterator[int]:
        coverage, retained_count, complement_count = self._take_coverage()
        replay = torch.multinomial(
            self.weights,
            REPLAY_CONSUMED + REPLAY_DROPPED,
            replacement=True,
            generator=self.replay_generator,
        ).tolist()
        effective = coverage + replay[:REPLAY_CONSUMED]
        shuffle = torch.randperm(
            CONSUMED_DRAWS, generator=self.order_generator
        ).tolist()
        consumed = [effective[index] for index in shuffle]
        output = consumed + replay[REPLAY_CONSUMED:]
        if len(output) != REQUESTED_DRAWS:
            raise RuntimeError("candidate sampler produced the wrong length")
        epoch_number = len(self.epoch_audits) + 1
        self.epoch_coverage_indices.append(coverage)
        two_epoch_union = None
        if epoch_number >= 2:
            two_epoch_union = len(
                set(self.epoch_coverage_indices[-2])
                | set(self.epoch_coverage_indices[-1])
            )
        self.epoch_audits.append({
            "sampler_epoch": epoch_number,
            "coverage_draws": len(coverage),
            "coverage_unique_within_epoch": len(set(coverage)),
            "coverage_retained_from_previous": retained_count,
            "coverage_complement_from_previous": complement_count,
            "replay_consumed": REPLAY_CONSUMED,
            "replay_dropped": REPLAY_DROPPED,
            "requested_draws": len(output),
            "coverage_index_sha256": tensor_index_sha256(coverage),
            "consumed_index_sha256": tensor_index_sha256(consumed),
            "dropped_index_sha256": tensor_index_sha256(output[-REPLAY_DROPPED:]),
            "adjacent_two_epoch_coverage_union": two_epoch_union,
        })
        return iter(output)


def build_sampler(
    arm: str,
    weights: torch.Tensor,
    *,
    num_samples: int,
    replacement: bool,
    generator: torch.Generator | None,
    seed: int,
) -> Sampler[int]:
    if arm == "native_replacement_control":
        return TorchWeightedRandomSampler(
            weights,
            num_samples=num_samples,
            replacement=replacement,
            generator=generator,
        )
    if arm == "coverage_replay_candidate":
        return CoverageSmallLesionReplaySampler(
            weights,
            num_samples=num_samples,
            replacement=replacement,
            seed=seed,
        )
    raise ValueError(f"unexpected arm: {arm}")


_ACTIVE_SAMPLER: Sampler[int] | None = None
_ACTIVE_ARM: str | None = None


def method_documents() -> dict[str, str]:
    docs = native.method_documents()
    docs.update({
        "idea_sha256": IDEA_SHA256,
        "prereg_sha256": PREREG_SHA256,
        "prereg_json_sha256": PREREG_JSON_SHA256,
        "sampler_spec_sha256": SAMPLER_SPEC_SHA256,
        "sampler_amendment_sha256": SAMPLER_AMENDMENT_SHA256,
        "seed_compatibility_amendment_sha256": SEED_AMENDMENT_SHA256,
        "self_contained_report_amendment_sha256": REPORT_AMENDMENT_SHA256,
        "final_writer_amendment_sha256": WRITER_AMENDMENT_SHA256,
    })
    return docs


def sampler_contract() -> dict[str, Any]:
    if _ACTIVE_ARM is None:
        raise RuntimeError("sampler contract requested before arm configuration")
    shared = {
        "population_count": POPULATION,
        "sampler_num_samples": REQUESTED_DRAWS,
        "consumed_draws_per_epoch": CONSUMED_DRAWS,
        "dropped_draws_per_epoch": REPLAY_DROPPED,
        "optimizer_steps_per_epoch": 94,
        "native_weight_sha256": native.CANDIDATE_WEIGHT_SHA256,
        "weight_source": "source-grid WT voxels inside frozen image-only brain crop",
        "weight_formula": "clip(sqrt(median(selected_native_volumes)/native_volume_i),0.5,4.0)",
        "weight_dtype": "torch.float64_cpu",
        "native_volume_artifact_sha256": native.NATIVE_ARTIFACT_SHA256,
        "native_volume_builder_sha256": native.NATIVE_BUILDER_SHA256,
    }
    if _ACTIVE_ARM == "native_replacement_control":
        return shared | {
            "strategy": "native_volume_weighted_replacement",
            "replacement": True,
            "generator_seed": 20260884,
        }
    audit = []
    if isinstance(_ACTIVE_SAMPLER, CoverageSmallLesionReplaySampler):
        audit = copy.deepcopy(_ACTIVE_SAMPLER.epoch_audits)
    return shared | {
        "strategy": "adjacent_complement_coverage_680_plus_native_weighted_replay_72",
        "replacement": "coverage_uses_previous_epoch_complement_plus_unique_retention; replay_with_replacement",
        "coverage_consumed_per_epoch": COVERAGE_DRAWS,
        "replay_consumed_per_epoch": REPLAY_CONSUMED,
        "replay_dropped_per_epoch": REPLAY_DROPPED,
        "coverage_seed": 20260884 + COVERAGE_SEED_OFFSET,
        "replay_seed": 20260884 + REPLAY_SEED_OFFSET,
        "order_seed": 20260884 + ORDER_SEED_OFFSET,
        "epoch_audits": audit,
    }


def patch_contract(value: Any) -> Any:
    if isinstance(value, dict):
        result = {key: patch_contract(item) for key, item in value.items()}
        if "sampling_contract" in result and isinstance(result["sampling_contract"], dict):
            result["sampling_contract"].update(sampler_contract())
        if "method_documents" in result:
            result["method_documents"] = method_documents()
        if "data_role" in result and isinstance(result["data_role"], dict):
            result["data_role"]["role"] = _ACTIVE_ARM
            result["data_role"]["retired_selector_confirmation_reclassified_as_train"] = True
        return result
    if isinstance(value, list):
        return [patch_contract(item) for item in value]
    return value


def candidate_sampler_gates(expected_epochs: int) -> dict[str, bool]:
    if not isinstance(_ACTIVE_SAMPLER, CoverageSmallLesionReplaySampler):
        return {"control_has_no_candidate_sampler_audit": True}
    audits = _ACTIVE_SAMPLER.epoch_audits
    coverages = _ACTIVE_SAMPLER.epoch_coverage_indices
    return {
        "sampler_epoch_count_exact": len(audits) == expected_epochs,
        "all_epoch_counts_exact": all(
            row["coverage_draws"] == COVERAGE_DRAWS
            and row["coverage_unique_within_epoch"] == COVERAGE_DRAWS
            and row["replay_consumed"] == REPLAY_CONSUMED
            and row["replay_dropped"] == REPLAY_DROPPED
            and row["requested_draws"] == REQUESTED_DRAWS
            for row in audits
        ),
        "all_adjacent_two_epoch_coverage_unions_complete": all(
            len(set(coverages[index - 1]) | set(coverages[index])) == POPULATION
            for index in range(1, len(coverages))
        ),
        "all_post_first_transitions_are_292_complement_plus_388_retained": all(
            row["coverage_complement_from_previous"] == 292
            and row["coverage_retained_from_previous"] == 388
            for row in audits[1:]
        ),
    }


def validate_with_seed_compatibility(
    args: argparse.Namespace,
    inherited_validator: Any,
) -> tuple[list[str], str]:
    frozen_seed = int(args.seed)
    if frozen_seed != 20260884:
        raise ValueError("current method requires frozen seed20260884")
    try:
        args.seed = 20260846
        result = inherited_validator(args)
    finally:
        args.seed = frozen_seed
    if args.seed != 20260884:
        raise RuntimeError("seed compatibility layer failed to restore current seed")
    return result


def configure(args: argparse.Namespace) -> None:
    global _ACTIVE_ARM, _ACTIVE_SAMPLER
    _ACTIVE_ARM = args.arm
    _ACTIVE_SAMPLER = None
    native.SCHEMA = SCHEMA
    native.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    native.ARMS = ARMS
    native.ARM_COUNTS = ARM_COUNTS
    native.ARM_MANIFEST_SHA256 = ARM_MANIFEST_SHA256
    native.IDEA_SHA256 = IDEA_SHA256
    native.PREREG_SHA256 = PREREG_SHA256
    native.AMENDMENT_001_SHA256 = SAMPLER_SPEC_SHA256
    native._ACTIVE_CASE_IDS = core.read_case_ids(args.train_case_ids) or []
    if len(native._ACTIVE_CASE_IDS) != POPULATION:
        raise ValueError("both arms require the frozen 972-case manifest")
    native._ACTIVE_ARM = args.arm
    native.configure_base(args)
    core.__file__ = str(Path(__file__).resolve())
    if sha256_file(args.prereg_json) != PREREG_JSON_SHA256:
        raise ValueError("frozen machine preregistration SHA mismatch")
    if sha256_file(args.sampler_amendment) != SAMPLER_AMENDMENT_SHA256:
        raise ValueError("frozen sampler amendment SHA mismatch")
    if sha256_file(args.seed_amendment) != SEED_AMENDMENT_SHA256:
        raise ValueError("frozen seed compatibility amendment SHA mismatch")
    if sha256_file(args.report_amendment) != REPORT_AMENDMENT_SHA256:
        raise ValueError("frozen self-contained report amendment SHA mismatch")
    if sha256_file(args.writer_amendment) != WRITER_AMENDMENT_SHA256:
        raise ValueError("frozen final writer amendment SHA mismatch")

    inherited_validator = core.validate_frozen_inputs
    def current_validator(value: argparse.Namespace) -> tuple[list[str], str]:
        return validate_with_seed_compatibility(value, inherited_validator)
    core.validate_frozen_inputs = current_validator

    def sampler_factory(
        weights: torch.Tensor,
        num_samples: int,
        replacement: bool = True,
        generator: torch.Generator | None = None,
    ) -> Sampler[int]:
        global _ACTIVE_SAMPLER
        _ACTIVE_SAMPLER = build_sampler(
            args.arm,
            weights,
            num_samples=num_samples,
            replacement=replacement,
            generator=generator,
            seed=args.seed,
        )
        return _ACTIVE_SAMPLER

    core.WeightedRandomSampler = sampler_factory

    def write_json_current(path: Path, value: Any) -> None:
        BASE_WRITE_JSON_ATOMIC(path, patch_contract(value))
    core.write_json_atomic = write_json_current

    save_checkpoint_native = core.save_checkpoint
    def save_checkpoint_current(*positional: Any, **keywords: Any) -> dict[str, Any]:
        item = save_checkpoint_native(*positional, **keywords)
        path = Path(item["path"])
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint = patch_contract(checkpoint)
        checkpoint["schema"] = CHECKPOINT_SCHEMA
        torch.save(checkpoint, path)
        item["sha256"] = sha256_file(path)
        return item
    core.save_checkpoint = save_checkpoint_current


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    configure(args)
    args.amendment = args.amendment3
    args.feasibility = args.native_artifact
    report = core.run(args)
    expected_sampler_epochs = 1 if args.mode == "smoke" else args.continuation_epochs
    gates = candidate_sampler_gates(expected_sampler_epochs)
    report["sampling_contract"] = sampler_contract()
    report["method_documents"] = method_documents()
    report = patch_contract(report)
    report["sampler_gates"] = gates
    report["all_gates_passed"] = bool(report["all_gates_passed"] and all(gates.values()))
    report["status"] = "passed" if args.mode == "smoke" and report["all_gates_passed"] else (
        "complete" if report["all_gates_passed"] else "failed"
    )
    output = args.output_dir / ("SMOKE.json" if args.mode == "smoke" else "TRAIN_DONE.json")
    core.write_json_atomic(output, report)
    if not report["all_gates_passed"]:
        raise RuntimeError("coverage/replay sampler gates failed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--train-case-ids", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--ema-teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    record = ROOT / "archive/research_records/2026-08-24_coverage_small_lesion_replay_kd"
    prior = ROOT / "archive/research_records/2026-08-21_voxel_kd_data972_native_sampling"
    parser.add_argument("--idea", type=Path, default=record / "IDEA.md")
    parser.add_argument("--prereg", type=Path, default=record / "PREREG.md")
    parser.add_argument("--prereg-json", type=Path, default=record / "PREREG.json")
    parser.add_argument("--amendment1", type=Path, default=record / "SAMPLER_SPEC.md")
    parser.add_argument("--sampler-amendment", type=Path, default=record / "AMENDMENT_001_PRE_GPU_COMPLEMENT_COVERAGE.md")
    parser.add_argument("--seed-amendment", type=Path, default=record / "AMENDMENT_002_PRE_MODEL_SEED_VALIDATOR_COMPATIBILITY.md")
    parser.add_argument("--report-amendment", type=Path, default=record / "AMENDMENT_003_PRE_FORMAL_SELF_CONTAINED_SMOKE_CONTRACT.md")
    parser.add_argument("--writer-amendment", type=Path, default=record / "AMENDMENT_004_PRE_FORMAL_FINAL_WRITER_ORDER.md")
    parser.add_argument("--amendment2", type=Path, default=prior / "AMENDMENT_002_CANONICAL_RAS_BEFORE_CROP.md")
    parser.add_argument("--amendment3", type=Path, default=prior / "AMENDMENT_003_FREEZE_NATIVE_ARTIFACT.md")
    parser.add_argument("--amendment4", type=Path, default=prior / "AMENDMENT_004_HISTORICAL_ADAMW_DEFAULTS.md")
    parser.add_argument("--native-artifact", type=Path, default=prior / "NATIVE_WT_VOLUMES.json")
    parser.add_argument("--native-builder", type=Path, default=ROOT / "scripts/build_t1ce_wt_native_volume_artifact.py")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260884)
    parser.add_argument("--start-global-epoch", type=int, default=168)
    parser.add_argument("--continuation-epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=core.TEMPERATURE)
    parser.add_argument("--kd-weight", type=float, default=core.KD_WEIGHT)
    parser.add_argument("--sampler-draws", type=int, default=core.SAMPLER_DRAWS)
    parser.add_argument("--smoke-steps", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
