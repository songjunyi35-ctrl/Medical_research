from __future__ import annotations

import copy

import pytest
import torch

from sammed3d import engine, partial_unfreeze, peft


@pytest.fixture()
def model():
    return engine.sam_model_registry3D[engine.MODEL_TYPE](checkpoint=None)


def test_last_four_policy_has_exact_trainable_set_and_counts(model) -> None:
    selection = partial_unfreeze.configure_vit_b_ori_last_four(model)
    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    assert selection.record["counts"] == {
        "trainable_parameters": 35_951_920,
        "total_parameters": 100_509_232,
    }
    assert sum(p.numel() for p in selection.encoder_parameters) == 28_367_616
    assert sum(p.numel() for p in selection.head_parameters) == 7_584_304
    assert trainable
    assert all(
        name.startswith(
            tuple(
                f"image_encoder.blocks.{index}."
                for index in partial_unfreeze.TRAINABLE_ENCODER_BLOCKS
            )
        )
        or name.startswith(("prompt_encoder.", "mask_decoder."))
        for name in trainable
    )
    assert not any(
        name.startswith("image_encoder.blocks.7.")
        or name.startswith("image_encoder.patch_embed.")
        or name.startswith("image_encoder.neck.")
        or name == "image_encoder.pos_embed"
        for name in trainable
    )


def test_discriminative_adamw_groups_are_exact_and_disjoint(model) -> None:
    selection = partial_unfreeze.configure_vit_b_ori_last_four(model)
    optimizer = partial_unfreeze.build_adamw(selection)

    assert [group["group_name"] for group in optimizer.param_groups] == [
        "image_encoder_blocks_8_11",
        "prompt_encoder_and_mask_decoder",
    ]
    assert [group["lr"] for group in optimizer.param_groups] == [1e-5, 1e-4]
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.1, 0.1]
    encoder_ids = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
    head_ids = {id(parameter) for parameter in optimizer.param_groups[1]["params"]}
    assert encoder_ids.isdisjoint(head_ids)


def test_policy_rejects_lora_co_training(model) -> None:
    peft.inject_qv_lora(model, rank=4, alpha=8, block_indexes=(8,))

    with pytest.raises(RuntimeError, match="forbids simultaneous encoder LoRA"):
        partial_unfreeze.configure_vit_b_ori_last_four(model)


def test_checkpoint_policy_and_optimizer_groups_are_fail_closed(model) -> None:
    selection = partial_unfreeze.configure_vit_b_ori_last_four(model)
    optimizer = partial_unfreeze.build_adamw(selection)
    payload = {
        "optimizer_state_dict": optimizer.state_dict(),
        "checkpoint_metadata": {
            "partial_unfreeze_policy": selection.record,
        },
    }
    partial_unfreeze.validate_checkpoint_payload(payload, selection.record)

    tampered = copy.deepcopy(payload)
    tampered["optimizer_state_dict"]["param_groups"][0]["lr"] = 2e-5
    with pytest.raises(RuntimeError, match="learning-rate mismatch"):
        partial_unfreeze.validate_checkpoint_payload(
            tampered,
            selection.record,
        )


def test_policy_rejects_wrong_encoder_depth(model) -> None:
    model.image_encoder.blocks = torch.nn.ModuleList(
        list(model.image_encoder.blocks[:11])
    )

    with pytest.raises(RuntimeError, match="exactly 12"):
        partial_unfreeze.configure_vit_b_ori_last_four(model)
