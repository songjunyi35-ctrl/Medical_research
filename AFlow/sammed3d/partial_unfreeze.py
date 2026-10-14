"""Fail-closed parameter policy for the T1ce/WT last-four-block arm.

This module does not alter the SAM-Med3D architecture or checkpoint state
keys.  It only selects the exact base parameters that may receive gradients,
builds two explicit AdamW parameter groups, and emits a serializable policy
record that a dedicated trainer can bind into every checkpoint.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch
from torch import nn

from sammed3d import peft


POLICY_SCHEMA = "aflow.sammed3d.t1ce_wt_last_four_partial_unfreeze"
POLICY_VERSION = 1
MODEL_TYPE = "vit_b_ori"
ENCODER_DEPTH = 12
TRAINABLE_ENCODER_BLOCKS = (8, 9, 10, 11)
FROZEN_ENCODER_BLOCKS = tuple(range(8))
ENCODER_LR = 1e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 0.1

# Counts are architecture contracts, not observations to be silently updated.
EXPECTED_ENCODER_NUMEL = 28_367_616
EXPECTED_HEAD_NUMEL = 7_584_304
EXPECTED_TOTAL_NUMEL = 35_951_920
EXPECTED_MODEL_NUMEL = 100_509_232


@dataclass(frozen=True)
class ParameterSelection:
    """The selected tensors and their checkpointable protocol record."""

    encoder_parameters: tuple[nn.Parameter, ...]
    head_parameters: tuple[nn.Parameter, ...]
    record: dict[str, Any]

    @property
    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.encoder_parameters + self.head_parameters

    def optimizer_groups(self) -> list[dict[str, Any]]:
        return [
            {
                "params": list(self.encoder_parameters),
                "lr": ENCODER_LR,
                "weight_decay": WEIGHT_DECAY,
                "group_name": "image_encoder_blocks_8_11",
            },
            {
                "params": list(self.head_parameters),
                "lr": HEAD_LR,
                "weight_decay": WEIGHT_DECAY,
                "group_name": "prompt_encoder_and_mask_decoder",
            },
        ]


def _parameter_fingerprint(named_parameters: Sequence[tuple[str, nn.Parameter]]) -> str:
    rows = [
        {
            "name": name,
            "numel": parameter.numel(),
            "shape": list(parameter.shape),
        }
        for name, parameter in named_parameters
    ]
    encoded = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _select_named_parameters(
    model: nn.Module,
) -> tuple[
    list[tuple[str, nn.Parameter]],
    list[tuple[str, nn.Parameter]],
]:
    blocks = getattr(getattr(model, "image_encoder", None), "blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError("model.image_encoder.blocks must be an nn.ModuleList")
    if len(blocks) != ENCODER_DEPTH:
        raise RuntimeError(
            f"expected exactly {ENCODER_DEPTH} image-encoder blocks, got {len(blocks)}"
        )
    if list(peft.iter_qv_lora_modules(model)):
        raise RuntimeError("partial-unfreeze arm forbids simultaneous encoder LoRA")
    if isinstance(getattr(model, "target_embed", None), peft.TargetEmbedding):
        raise RuntimeError("T1ce/WT partial-unfreeze arm forbids target embeddings")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for index in TRAINABLE_ENCODER_BLOCKS:
        for parameter in blocks[index].parameters():
            parameter.requires_grad_(True)
    for module in (model.prompt_encoder, model.mask_decoder):
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    encoder_prefixes = tuple(
        f"image_encoder.blocks.{index}."
        for index in TRAINABLE_ENCODER_BLOCKS
    )
    encoder_named = []
    head_named = []
    unexpected_trainable = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(encoder_prefixes):
            encoder_named.append((name, parameter))
        elif name.startswith(("prompt_encoder.", "mask_decoder.")):
            head_named.append((name, parameter))
        else:
            unexpected_trainable.append(name)
    if unexpected_trainable:
        raise RuntimeError(
            "unexpected trainable parameters: " + ", ".join(unexpected_trainable)
        )
    if not encoder_named or not head_named:
        raise RuntimeError("encoder and head parameter groups must both be non-empty")

    selected_ids = [id(parameter) for _, parameter in encoder_named + head_named]
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("optimizer parameter groups contain shared tensors")

    trainable_names = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    selected_names = {
        name for name, _ in encoder_named + head_named
    }
    if trainable_names != selected_names:
        raise RuntimeError("optimizer groups do not exactly cover trainable parameters")
    return encoder_named, head_named


def configure_vit_b_ori_last_four(model: nn.Module) -> ParameterSelection:
    """Freeze everything except blocks 8--11, prompt encoder, and mask decoder."""
    encoder_named, head_named = _select_named_parameters(model)
    encoder_numel = sum(parameter.numel() for _, parameter in encoder_named)
    head_numel = sum(parameter.numel() for _, parameter in head_named)
    total_numel = sum(parameter.numel() for parameter in model.parameters())
    observed = {
        "encoder": encoder_numel,
        "head": head_numel,
        "trainable": encoder_numel + head_numel,
        "model": total_numel,
    }
    expected = {
        "encoder": EXPECTED_ENCODER_NUMEL,
        "head": EXPECTED_HEAD_NUMEL,
        "trainable": EXPECTED_TOTAL_NUMEL,
        "model": EXPECTED_MODEL_NUMEL,
    }
    if observed != expected:
        raise RuntimeError(
            f"vit_b_ori parameter-count contract mismatch: "
            f"observed={observed}, expected={expected}"
        )

    record = {
        "schema": POLICY_SCHEMA,
        "schema_version": POLICY_VERSION,
        "model_type": MODEL_TYPE,
        "image_encoder": {
            "update": "base_weight_partial_unfreeze",
            "depth": ENCODER_DEPTH,
            "trainable_blocks": list(TRAINABLE_ENCODER_BLOCKS),
            "frozen_blocks": list(FROZEN_ENCODER_BLOCKS),
            "non_block_parameters_trainable": False,
            "lora_enabled": False,
        },
        "heads": {
            "prompt_encoder_trainable": True,
            "mask_decoder_trainable": True,
            "target_embedding_enabled": False,
        },
        "optimizer": {
            "type": "AdamW",
            "groups": [
                {
                    "name": "image_encoder_blocks_8_11",
                    "lr": ENCODER_LR,
                    "weight_decay": WEIGHT_DECAY,
                    "parameter_tensors": len(encoder_named),
                    "parameter_numel": encoder_numel,
                    "parameter_fingerprint_sha256": _parameter_fingerprint(
                        encoder_named
                    ),
                },
                {
                    "name": "prompt_encoder_and_mask_decoder",
                    "lr": HEAD_LR,
                    "weight_decay": WEIGHT_DECAY,
                    "parameter_tensors": len(head_named),
                    "parameter_numel": head_numel,
                    "parameter_fingerprint_sha256": _parameter_fingerprint(
                        head_named
                    ),
                },
            ],
        },
        "counts": {
            "trainable_parameters": encoder_numel + head_numel,
            "total_parameters": total_numel,
        },
    }
    return ParameterSelection(
        encoder_parameters=tuple(parameter for _, parameter in encoder_named),
        head_parameters=tuple(parameter for _, parameter in head_named),
        record=record,
    )


def build_adamw(selection: ParameterSelection) -> torch.optim.AdamW:
    """Build and immediately validate the two discriminative-LR groups."""
    optimizer = torch.optim.AdamW(selection.optimizer_groups())
    validate_optimizer(optimizer, selection.record)
    return optimizer


def validate_optimizer(
    optimizer: torch.optim.Optimizer,
    policy_record: Mapping[str, Any],
) -> None:
    expected_groups = policy_record["optimizer"]["groups"]
    if len(optimizer.param_groups) != len(expected_groups):
        raise RuntimeError("optimizer must contain exactly two parameter groups")
    seen: set[int] = set()
    for actual, expected in zip(optimizer.param_groups, expected_groups):
        if actual.get("group_name") != expected["name"]:
            raise RuntimeError("optimizer parameter-group name mismatch")
        if float(actual["lr"]) != float(expected["lr"]):
            raise RuntimeError("optimizer parameter-group learning-rate mismatch")
        if float(actual["weight_decay"]) != float(expected["weight_decay"]):
            raise RuntimeError("optimizer parameter-group weight-decay mismatch")
        parameters = actual["params"]
        if len(parameters) != int(expected["parameter_tensors"]):
            raise RuntimeError("optimizer parameter-tensor count mismatch")
        if sum(parameter.numel() for parameter in parameters) != int(
            expected["parameter_numel"]
        ):
            raise RuntimeError("optimizer parameter-numel mismatch")
        for parameter in parameters:
            if id(parameter) in seen:
                raise RuntimeError("optimizer parameter groups overlap")
            seen.add(id(parameter))
            if not parameter.requires_grad:
                raise RuntimeError("optimizer contains a frozen parameter")


def validate_checkpoint_payload(
    payload: Mapping[str, Any],
    expected_policy_record: Mapping[str, Any],
) -> None:
    """Require exact policy metadata and optimizer-group state on resume."""
    checkpoint_metadata = payload.get("checkpoint_metadata")
    if not isinstance(checkpoint_metadata, Mapping):
        raise RuntimeError("checkpoint is missing checkpoint_metadata")
    saved_policy = checkpoint_metadata.get("partial_unfreeze_policy")
    if saved_policy != expected_policy_record:
        raise RuntimeError("checkpoint partial-unfreeze policy mismatch")

    optimizer_state = payload.get("optimizer_state_dict")
    if not isinstance(optimizer_state, Mapping):
        raise RuntimeError("checkpoint is missing optimizer_state_dict")
    saved_groups = optimizer_state.get("param_groups")
    expected_groups = expected_policy_record["optimizer"]["groups"]
    if not isinstance(saved_groups, list) or len(saved_groups) != len(expected_groups):
        raise RuntimeError("checkpoint optimizer parameter-group count mismatch")
    for actual, expected in zip(saved_groups, expected_groups):
        if actual.get("group_name") != expected["name"]:
            raise RuntimeError("checkpoint optimizer parameter-group name mismatch")
        if float(actual["lr"]) != float(expected["lr"]):
            raise RuntimeError("checkpoint optimizer learning-rate mismatch")
        if float(actual["weight_decay"]) != float(expected["weight_decay"]):
            raise RuntimeError("checkpoint optimizer weight-decay mismatch")
        if len(actual.get("params", [])) != int(expected["parameter_tensors"]):
            raise RuntimeError("checkpoint optimizer parameter-tensor count mismatch")
