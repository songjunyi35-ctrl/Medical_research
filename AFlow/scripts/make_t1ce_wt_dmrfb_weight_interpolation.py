#!/usr/bin/env python3
"""Generate the three frozen KD200-to-DMRFB216 interpolation checkpoints."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator  # noqa: E402
from sammed3d.t1ce_wt_dmrfb_adapter import (  # noqa: E402
    DMRFBAdapterLocator,
    FORMAL_DEPLOY_PARAMETERS,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_dmrfb_weight_interpolation_generation_receipt"
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_dmrfb_weight_interpolation_checkpoint"
IDEA_SHA256 = "858b963e637e35af0825d6813d6da26c1ef64417f3b496dcad52da23ec485119"
PREREG_SHA256 = "2674c8f400ca2822f8a58b7ecc2fc9947034d2453f5b4987819fd10ac5453db3"
KD200_SHA256 = "56f54898ded11a0208e75f8000d57d4d053e80d6dedd760402730ad345c111b7"
DMRFB216_SHA256 = "df53d9bd61f1489746ae1240fffba1f8a3897d0de1416122a389e1d420def684"
INTERIOR_ALPHAS = (0.25, 0.50, 0.75)
EXPECTED_STATE_KEYS = 90
FIXED_INPUT_SEED = 20261016
FIXED_INPUT_SHAPE = (1, 1, 16, 16, 16)
FORBIDDEN_OUTPUT_TOKENS = ("validation", "test", "outer", "confirmation")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in state:
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_state_geometry(
    endpoint0: Mapping[str, torch.Tensor],
    endpoint1: Mapping[str, torch.Tensor],
    *,
    expected_keys: int | None = EXPECTED_STATE_KEYS,
) -> dict[str, Any]:
    if tuple(endpoint0) != tuple(endpoint1):
        missing = sorted(set(endpoint0) - set(endpoint1))
        extra = sorted(set(endpoint1) - set(endpoint0))
        raise ValueError(f"endpoint state keys differ: endpoint1_missing={missing}, endpoint1_extra={extra}")
    if expected_keys is not None and len(endpoint0) != int(expected_keys):
        raise ValueError(
            f"state key count {len(endpoint0)} differs from expected {expected_keys}"
        )
    floating: list[str] = []
    nonfloating: list[str] = []
    total = 0
    for name in endpoint0:
        left, right = endpoint0[name], endpoint1[name]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise TypeError(f"state value {name} is not a tensor")
        if left.shape != right.shape:
            raise ValueError(f"state shape mismatch for {name}: {left.shape} != {right.shape}")
        if left.dtype != right.dtype:
            raise ValueError(f"state dtype mismatch for {name}: {left.dtype} != {right.dtype}")
        total += int(left.numel())
        if left.is_floating_point():
            if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
                raise ValueError(f"non-finite floating endpoint tensor: {name}")
            floating.append(name)
        else:
            if not torch.equal(left, right):
                raise ValueError(f"non-floating endpoint tensor differs: {name}")
            nonfloating.append(name)
    return {
        "state_keys": len(endpoint0),
        "total_values": total,
        "floating_keys": floating,
        "nonfloating_keys": nonfloating,
        "endpoint0_state_sha256": state_sha256(endpoint0),
        "endpoint1_state_sha256": state_sha256(endpoint1),
    }


def interpolate_state_dict(
    endpoint0: Mapping[str, torch.Tensor],
    endpoint1: Mapping[str, torch.Tensor],
    *,
    alpha: float,
    expected_keys: int | None = EXPECTED_STATE_KEYS,
) -> OrderedDict[str, torch.Tensor]:
    value = float(alpha)
    if value not in INTERIOR_ALPHAS:
        raise ValueError(f"alpha must be one of {INTERIOR_ALPHAS}")
    validate_state_geometry(endpoint0, endpoint1, expected_keys=expected_keys)
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name in endpoint0:
        left = endpoint0[name].detach().cpu().contiguous()
        right = endpoint1[name].detach().cpu().contiguous()
        if left.is_floating_point():
            interpolated = (
                (1.0 - value) * left.to(torch.float32)
                + value * right.to(torch.float32)
            ).to(left.dtype)
            if not bool(torch.isfinite(interpolated).all()):
                raise RuntimeError(f"interpolation produced non-finite tensor: {name}")
            result[name] = interpolated.contiguous()
        else:
            # Geometry validation already proved exact equality.
            result[name] = left.clone()
    return result


def _load_payload(path: Path, expected_sha256: str, description: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{description} SHA-256 mismatch")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f"{description} is not a checkpoint dictionary")
    return value


def build_alpha0_model(kd200: Mapping[str, Any]) -> DMRFBAdapterLocator:
    model = DMRFBAdapterLocator.from_plain_checkpoint(
        kd200, arm="dmrfb_dilation2"
    ).cpu().eval()
    if torch.count_nonzero(model.adapter.projection.weight).item() != 0:
        raise RuntimeError("alpha0 projection weight is not zero")
    if model.adapter.projection.bias is None or torch.count_nonzero(
        model.adapter.projection.bias
    ).item() != 0:
        raise RuntimeError("alpha0 projection bias is not zero")
    if sum(parameter.numel() for parameter in model.parameters()) != FORMAL_DEPLOY_PARAMETERS:
        raise RuntimeError("alpha0 deploy parameter count mismatch")
    return model


def load_alpha1_model(
    checkpoint: Mapping[str, Any],
) -> DMRFBAdapterLocator:
    if checkpoint.get("schema") != "aflow.sammed3d.t1ce_wt_dmrfb_adapter_checkpoint":
        raise ValueError("alpha1 checkpoint schema mismatch")
    if checkpoint.get("arm") != "dmrfb_dilation2" or checkpoint.get("global_epoch") != 216:
        raise ValueError("alpha1 checkpoint is not DMRFB dilation2 epoch216")
    for field in (
        "validation_access",
        "outer_selector_access",
        "confirmation_access",
        "test_access",
    ):
        if checkpoint.get(field) is not False:
            raise ValueError(f"alpha1 checkpoint {field} must be false")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("alpha1 checkpoint lacks model_state_dict")
    model = DMRFBAdapterLocator(
        arm="dmrfb_dilation2", init_filters=16, dropout=0.1
    ).cpu().eval()
    model.load_state_dict(state, strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != FORMAL_DEPLOY_PARAMETERS:
        raise RuntimeError("alpha1 deploy parameter count mismatch")
    return model


def fixed_input() -> torch.Tensor:
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(FIXED_INPUT_SEED)
        return torch.randn(FIXED_INPUT_SHAPE, dtype=torch.float32)


@torch.inference_mode()
def finite_logits(model: DMRFBAdapterLocator, image: torch.Tensor) -> torch.Tensor:
    value = model.eval()(image).final_logits.float().cpu()
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError("fixed-input logits are non-finite")
    return value


def alpha_token(alpha: float) -> str:
    if float(alpha) not in INTERIOR_ALPHAS:
        raise ValueError("unexpected interpolation alpha")
    return f"{int(round(100 * float(alpha))):03d}"


def generate(args: argparse.Namespace) -> dict[str, Any]:
    for path, expected, description in (
        (args.idea, IDEA_SHA256, "idea"),
        (args.preregistration, PREREG_SHA256, "preregistration"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"{description} SHA-256 mismatch")
    lowered = str(args.output_dir.resolve()).lower()
    if any(token in lowered for token in FORBIDDEN_OUTPUT_TOKENS):
        raise ValueError("interpolation output path contains a forbidden data-role token")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse interpolation directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    kd200 = _load_payload(args.kd200_checkpoint, KD200_SHA256, "KD200")
    dmrfb216 = _load_payload(args.dmrfb216_checkpoint, DMRFB216_SHA256, "DMRFB216")
    endpoint0_model = build_alpha0_model(kd200)
    endpoint1_model = load_alpha1_model(dmrfb216)
    endpoint0 = endpoint0_model.deploy_state_dict(cpu=True)
    endpoint1 = endpoint1_model.deploy_state_dict(cpu=True)
    geometry = validate_state_geometry(endpoint0, endpoint1)
    if geometry["state_keys"] != EXPECTED_STATE_KEYS:
        raise RuntimeError("frozen interpolation state key count changed")

    image = fixed_input()
    # Alpha0 must remain the exact original KD200 function.
    architecture = kd200["architecture"]
    plain = AutomaticPromptLocator(
        center_aware=False,
        init_filters=int(architecture["init_filters"]),
        dropout=float(architecture["dropout"]),
    ).cpu().eval()
    plain.load_state_dict(kd200["model_state_dict"], strict=True)
    alpha0_logits = finite_logits(endpoint0_model, image)
    with torch.inference_mode():
        plain_logits = plain(image).segmentation_logits.float().cpu()
    alpha0_plain_difference = float((alpha0_logits - plain_logits).abs().max())
    if alpha0_plain_difference != 0.0:
        raise RuntimeError("alpha0 DMRFB model is not exactly KD200")
    finite_logits(endpoint1_model, image)

    checkpoints: dict[str, Any] = {}
    for alpha in INTERIOR_ALPHAS:
        state = interpolate_state_dict(endpoint0, endpoint1, alpha=alpha)
        reference = DMRFBAdapterLocator(
            arm="dmrfb_dilation2", init_filters=16, dropout=0.1
        ).cpu().eval()
        reference.load_state_dict(state, strict=True)
        reference_logits = finite_logits(reference, image)
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "schema_version": 1,
            "alpha": alpha,
            "arm": "dmrfb_dilation2",
            "resolution": 128,
            "model_state_dict": state,
            "architecture": {
                "class": "DMRFBAdapterLocator",
                "backbone": "MONAI SegResNet",
                "center_aware": False,
                "init_filters": 16,
                "dropout": 0.1,
                "input_channels": ["T1ce"],
                "outputs": ["WT segmentation logit"],
                "deploy_parameter_count": FORMAL_DEPLOY_PARAMETERS,
            },
            "interpolation": {
                "formula": "(1-alpha)*endpoint0+alpha*endpoint1 in float32 then cast",
                "alpha_grid": list(INTERIOR_ALPHAS),
                "endpoint0_checkpoint_sha256": KD200_SHA256,
                "endpoint1_checkpoint_sha256": DMRFB216_SHA256,
                "endpoint0_state_sha256": geometry["endpoint0_state_sha256"],
                "endpoint1_state_sha256": geometry["endpoint1_state_sha256"],
            },
            "state_geometry": geometry,
            "interpolated_state_sha256": state_sha256(state),
            "idea_sha256": IDEA_SHA256,
            "preregistration_sha256": PREREG_SHA256,
            "generator_sha256": sha256_file(Path(__file__).resolve()),
            "fixed_input_sha256": tensor_sha256(image),
            "fixed_input_logits_finite": True,
            "checkpoint_reload_max_abs_difference": None,
            "source_dmrfb_validation_access": True,
            "new_interpolation_validation_access": False,
            "validation_access": False,
            "outer_selector_access": False,
            "confirmation_access": False,
            "test_access": False,
        }
        token = alpha_token(alpha)
        path = args.output_dir / f"plain_128_dmrfb_interpolation_alpha{token}.pth"
        torch.save(payload, path)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        reload_model = DMRFBAdapterLocator(
            arm="dmrfb_dilation2", init_filters=16, dropout=0.1
        ).cpu().eval()
        reload_model.load_state_dict(loaded["model_state_dict"], strict=True)
        reload_logits = finite_logits(reload_model, image)
        difference = float((reference_logits - reload_logits).abs().max())
        if difference != 0.0:
            raise RuntimeError(f"alpha{alpha} strict reload logits differ")
        loaded["checkpoint_reload_max_abs_difference"] = difference
        torch.save(loaded, path)
        checkpoints[str(alpha)] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "alpha": alpha,
            "state_keys": geometry["state_keys"],
            "parameter_count": FORMAL_DEPLOY_PARAMETERS,
            "checkpoint_reload_max_abs_difference": difference,
            "fixed_input_logits_finite": True,
            "interpolated_state_sha256": loaded["interpolated_state_sha256"],
        }

    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "all_gates_passed": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "alphas": list(INTERIOR_ALPHAS),
        "checkpoints": checkpoints,
        "state_geometry": geometry,
        "alpha0_plain_max_abs_difference": alpha0_plain_difference,
        "fixed_input_sha256": tensor_sha256(image),
        "idea_sha256": IDEA_SHA256,
        "preregistration_sha256": PREREG_SHA256,
        "endpoint0_checkpoint_sha256": KD200_SHA256,
        "endpoint1_checkpoint_sha256": DMRFB216_SHA256,
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "source_dmrfb_validation_access": True,
        "new_interpolation_validation_access": False,
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    write_json_atomic(args.output_dir / "GENERATION.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idea", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--kd200-checkpoint", type=Path, required=True)
    parser.add_argument("--dmrfb216-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = generate(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "alphas": result["alphas"],
                "checkpoints": result["checkpoints"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
