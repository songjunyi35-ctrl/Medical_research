#!/usr/bin/env python3
"""Native-volume-weighted cache-matched 756/972 voxel-KD experiment."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_t1ce_wt_plain_voxel_kd_data972 as base  # noqa: E402
from scripts.build_t1ce_wt_native_volume_artifact import native_sampling_weights  # noqa: E402
from scripts.train_t1ce_wt_capg3d import sha256_file  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_voxel_kd_data972_native_train"
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_voxel_kd_data972_native_checkpoint"
ARMS = ("native756_control", "native972_candidate")
ARM_COUNTS = {"native756_control": (756, 346), "native972_candidate": (972, 444)}
ARM_MANIFEST_SHA256 = {
    "native756_control": "a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16",
    "native972_candidate": "6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65",
}
IDEA_SHA256 = "b90b5b340d58b5589859695b0c7515e6a256e9ae4bd8e6a9824cae07a1539856"
PREREG_SHA256 = "1a0d4cd398b1233ceac4571b900783f34e0a84e74a4b836b3ef1502050281d0f"
AMENDMENT_001_SHA256 = "5a84420082b4165f8db2cc4e8861d95c5a2b3191fb9b51f8948b3d150fb11b72"
AMENDMENT_002_SHA256 = "9d59b490b533244fba01dba1554a13d80ffdd67cc672659743a116a879154b38"
AMENDMENT_003_SHA256 = "bce7f00eed898a7d2825b03536f3cf99335b4390315359cd189ec368a57462a8"
AMENDMENT_004_SHA256 = "b1d03a41295fa860c620d8815479e8d1046bbfd052027a80afe253c8d32eee88"
NATIVE_ARTIFACT_SHA256 = "32be09856d71e78f512792c8bcf8f5f9ee137e6ba1e4358862ff2a61ba828c31"
NATIVE_BUILDER_SHA256 = "80874d4745d7ae5b0ced90471f0fa59d9c6345a597b27ec02a253eb6a19c010a"
CONTROL_WEIGHT_SHA256 = "757e463d8e60f96ef6429613f6783fb3aa1014b580932959f9ed354d37072124"
CANDIDATE_WEIGHT_SHA256 = "81d3fa6a907f791c4d0b97efe95ece5e057d15b85673463cadbeecc2b3f148ec"


_ACTIVE_ARTIFACT: dict[str, Any] | None = None
_ACTIVE_ARM: str | None = None


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def method_documents() -> dict[str, str]:
    return {
        "idea_sha256": IDEA_SHA256,
        "prereg_sha256": PREREG_SHA256,
        "amendment_001_sha256": AMENDMENT_001_SHA256,
        "amendment_002_sha256": AMENDMENT_002_SHA256,
        "amendment_003_sha256": AMENDMENT_003_SHA256,
        "amendment_004_sha256": AMENDMENT_004_SHA256,
        "native_artifact_sha256": NATIVE_ARTIFACT_SHA256,
        "native_builder_sha256": NATIVE_BUILDER_SHA256,
    }


def native_sampling_contract(arm: str, case_ids: list[str]) -> tuple[torch.Tensor, dict[str, Any]]:
    if _ACTIVE_ARTIFACT is None:
        raise RuntimeError("native artifact is not configured")
    volumes = _ACTIVE_ARTIFACT["native_wt_voxels"]
    weights = native_sampling_weights(case_ids, volumes)
    expected = CONTROL_WEIGHT_SHA256 if arm == "native756_control" else CANDIDATE_WEIGHT_SHA256
    actual = tensor_sha256(weights)
    if actual != expected:
        raise ValueError(f"native sampling weight SHA mismatch for {arm}")
    return weights, {
        "weight_source": "source-grid WT voxels inside frozen image-only brain crop",
        "weight_formula": "clip(sqrt(median(selected_native_volumes)/native_volume_i),0.5,4.0)",
        "weight_dtype": "torch.float64_cpu",
        "native_volume_artifact_sha256": NATIVE_ARTIFACT_SHA256,
        "native_volume_builder_sha256": NATIVE_BUILDER_SHA256,
        "native_weight_sha256": actual,
        "native_weight_min": float(weights.min()),
        "native_weight_max": float(weights.max()),
    }


class NativeVolumeBrainCropDataset(base.BrainCropDataset):
    def sampling_weights(self) -> torch.Tensor:
        if _ACTIVE_ARM is None:
            raise RuntimeError("native arm is not configured")
        weights, _ = native_sampling_contract(_ACTIVE_ARM, self.case_ids)
        return weights


def patch_native_contract(value: Any) -> Any:
    """Return a JSON/checkpoint structure with the frozen native contract attached."""

    if isinstance(value, dict):
        result = {key: patch_native_contract(item) for key, item in value.items()}
        if "sampling_contract" in result and isinstance(result["sampling_contract"], dict):
            if _ACTIVE_ARM is None or _ACTIVE_ARTIFACT is None:
                raise RuntimeError("native contract patch requires active arm/artifact")
            expected_count = ARM_COUNTS[_ACTIVE_ARM][0]
            artifact_ids = list(_ACTIVE_ARTIFACT["native_wt_voxels"])
            if _ACTIVE_ARM == "native756_control":
                # The selected control IDs are supplied by the manifest; weight SHA is the key gate.
                native_sha = CONTROL_WEIGHT_SHA256
            else:
                if len(artifact_ids) != expected_count:
                    raise ValueError("native artifact candidate population mismatch")
                native_sha = CANDIDATE_WEIGHT_SHA256
            result["sampling_contract"].update({
                "weight_source": "source-grid WT voxels inside frozen image-only brain crop",
                "weight_formula": "clip(sqrt(median(selected_native_volumes)/native_volume_i),0.5,4.0)",
                "weight_dtype": "torch.float64_cpu",
                "native_volume_artifact_sha256": NATIVE_ARTIFACT_SHA256,
                "native_volume_builder_sha256": NATIVE_BUILDER_SHA256,
                "native_weight_sha256": native_sha,
            })
        if "method_documents" in result:
            result["method_documents"] = method_documents()
        return result
    if isinstance(value, list):
        return [patch_native_contract(item) for item in value]
    return value


def configure_base(args: argparse.Namespace) -> None:
    global _ACTIVE_ARTIFACT, _ACTIVE_ARM
    _ACTIVE_ARM = args.arm
    artifact = json.loads(args.native_artifact.read_text(encoding="utf-8"))
    if artifact.get("schema") != "aflow.brats.t1ce_wt_native_volume_artifact":
        raise ValueError("unexpected native-volume artifact schema")
    if artifact.get("status") != "passed" or artifact.get("audit", {}).get("all_gates_passed") is not True:
        raise ValueError("native-volume artifact did not pass all gates")
    if artifact.get("audit", {}).get("control_weight_values_exact") is not True:
        raise ValueError("native control weights are not historically exact")
    _ACTIVE_ARTIFACT = artifact

    frozen = {
        args.idea: IDEA_SHA256,
        args.prereg: PREREG_SHA256,
        args.amendment1: AMENDMENT_001_SHA256,
        args.amendment2: AMENDMENT_002_SHA256,
        args.amendment3: AMENDMENT_003_SHA256,
        args.amendment4: AMENDMENT_004_SHA256,
        args.native_artifact: NATIVE_ARTIFACT_SHA256,
        args.native_builder: NATIVE_BUILDER_SHA256,
    }
    for path, expected in frozen.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen native input SHA mismatch: {path}")

    base.__file__ = str(Path(__file__).resolve())
    base.SCHEMA = SCHEMA
    base.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    base.ARMS = ARMS
    base.ARM_COUNTS = ARM_COUNTS
    base.ARM_MANIFEST_SHA256 = ARM_MANIFEST_SHA256
    base.IDEA_SHA256 = IDEA_SHA256
    base.PREREG_SHA256 = PREREG_SHA256
    # The inherited preflight has four document slots. Bind the latest amendment/artifact
    # there and independently verify all earlier amendments above.
    base.AMENDMENT_SHA256 = AMENDMENT_003_SHA256
    base.FEASIBILITY_SHA256 = NATIVE_ARTIFACT_SHA256
    base.BrainCropDataset = NativeVolumeBrainCropDataset

    original_adamw = base.torch.optim.AdamW
    def historical_adamw(parameters: Any, *positional: Any, **keywords: Any) -> Any:
        keywords["foreach"] = None
        keywords["fused"] = None
        return original_adamw(parameters, *positional, **keywords)
    base.torch.optim.AdamW = historical_adamw

    original_write_json_atomic = base.write_json_atomic
    def write_json_native(path: Path, value: Any) -> None:
        original_write_json_atomic(path, patch_native_contract(value))
    base.write_json_atomic = write_json_native

    original_save_checkpoint = base.save_checkpoint
    def save_checkpoint_native(*positional: Any, **keywords: Any) -> dict[str, Any]:
        contract = copy.deepcopy(keywords["sampling_contract"])
        _, native = native_sampling_contract(args.arm, list(_ACTIVE_CASE_IDS))
        contract.update(native)
        keywords["sampling_contract"] = contract
        item = original_save_checkpoint(*positional, **keywords)
        path = Path(item["path"])
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint = patch_native_contract(checkpoint)
        checkpoint["schema"] = CHECKPOINT_SCHEMA
        checkpoint["native_sampling"] = native
        torch.save(checkpoint, path)
        item["sha256"] = sha256_file(path)
        return item
    base.save_checkpoint = save_checkpoint_native


_ACTIVE_CASE_IDS: list[str] = []


def run(args: argparse.Namespace) -> dict[str, Any]:
    global _ACTIVE_CASE_IDS
    if args.arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    _ACTIVE_CASE_IDS = base.read_case_ids(args.train_case_ids) or []
    if len(_ACTIVE_CASE_IDS) != ARM_COUNTS[args.arm][0]:
        raise ValueError("native arm manifest count mismatch")
    configure_base(args)
    # Map the inherited four-document CLI contract to the new frozen sources.
    args.amendment = args.amendment3
    args.feasibility = args.native_artifact
    report = base.run(args)
    return patch_native_contract(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--train-case-ids", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--ema-teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    record = ROOT / "archive/research_records/2026-08-21_voxel_kd_data972_native_sampling"
    parser.add_argument("--idea", type=Path, default=record / "IDEA.md")
    parser.add_argument("--prereg", type=Path, default=record / "PREREG.md")
    parser.add_argument("--amendment1", type=Path, default=record / "AMENDMENT_001_BRAINCROP_NATIVE_VOLUME_DEFINITION.md")
    parser.add_argument("--amendment2", type=Path, default=record / "AMENDMENT_002_CANONICAL_RAS_BEFORE_CROP.md")
    parser.add_argument("--amendment3", type=Path, default=record / "AMENDMENT_003_FREEZE_NATIVE_ARTIFACT.md")
    parser.add_argument("--amendment4", type=Path, default=record / "AMENDMENT_004_HISTORICAL_ADAMW_DEFAULTS.md")
    parser.add_argument("--native-artifact", type=Path, default=record / "NATIVE_WT_VOLUMES.json")
    parser.add_argument("--native-builder", type=Path, default=ROOT / "scripts/build_t1ce_wt_native_volume_artifact.py")
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
    parser.add_argument("--kd-weight", type=float, default=base.KD_WEIGHT)
    parser.add_argument("--sampler-draws", type=int, default=base.SAMPLER_DRAWS)
    parser.add_argument("--smoke-steps", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
