from __future__ import annotations

import argparse
import csv

import pytest
import torch

from sammed3d import engine, partial_unfreeze, peft
from scripts import train_sammed3d_brats as generic
from scripts import train_sammed3d_t1ce_wt_partial_unfreeze as runner


def test_machine_prereg_and_parent_md_are_exactly_bound() -> None:
    prereg, path = runner.load_and_validate_prereg()

    assert runner.sha256_file(path) == runner.PREREG_SHA256
    assert prereg["authorization"] == {
        "development_training": True,
        "test_access": False,
    }
    assert prereg["parent_md_sha256"] == (
        "111968f92c09f35272a5ddd35741c3ad1a5b2803a09f307b2a6993473e156156"
    )
    assert (
        prereg["optimization"]["rng_reseed_after_model_setup"] is True
    )
    assert "closed-loop contrast" in prereg["decision"]["estimand_scope"]
    assert "C2/C3 model-specific adaptive XOR" in (
        prereg["decision"]["prompt_trajectory"]
    )


def test_smoke_output_resolution_does_not_create_directory(tmp_path) -> None:
    prereg, _ = runner.load_and_validate_prereg()
    output = tmp_path / "smoke_fv_partial_last4_unit"
    args = argparse.Namespace(
        mode="smoke",
        output_dir=output,
        device="cpu",
        preflight_only=False,
    )

    resolved = runner.resolve_output(args, prereg)

    assert resolved == output
    assert not output.exists()


def test_independent_history_writer_has_fixed_complete_schema(tmp_path) -> None:
    row = {
        field: index for index, field in enumerate(runner.HISTORY_FIELDS)
    }
    path = tmp_path / "history.csv"

    runner.append_history_csv(path, row)

    with path.open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == 1
    assert tuple(saved[0]) == runner.HISTORY_FIELDS
    with pytest.raises(RuntimeError, match="fixed CSV schema"):
        runner.append_history_csv(path, {"epoch": 2})


def test_gradient_gate_requires_finite_signal_per_group() -> None:
    encoder = torch.nn.Parameter(torch.ones(1))
    head = torch.nn.Parameter(torch.ones(1))
    selection = partial_unfreeze.ParameterSelection(
        encoder_parameters=(encoder,),
        head_parameters=(head,),
        record={},
    )
    encoder.grad = torch.ones_like(encoder)
    head.grad = torch.ones_like(head)
    runner.ensure_finite_gradients(selection)

    head.grad = None
    with pytest.raises(RuntimeError, match="prompt_and_mask_heads has no gradients"):
        runner.ensure_finite_gradients(selection)


def test_training_rng_is_reseeded_after_model_setup(monkeypatch) -> None:
    observed: list[int] = []
    monkeypatch.setattr(generic, "seed_everything", observed.append)

    runner.reseed_after_model_setup()

    assert observed == [runner.TRAIN_SEED]


def test_checkpoint_strict_reload_reproduces_policy_and_groups(tmp_path) -> None:
    model = engine.sam_model_registry3D[engine.MODEL_TYPE](checkpoint=None)
    selection = partial_unfreeze.configure_vit_b_ori_last_four(model)
    optimizer = partial_unfreeze.build_adamw(selection)
    adapter_config = peft.make_adapter_config(
        target_token=False,
        lora_rank=0,
        model_type=engine.MODEL_TYPE,
    )
    path = tmp_path / "selected.pth"
    prereg_path = runner._resolve(runner.PREREG_PATH)
    args = argparse.Namespace(mode="smoke")
    best = {
        "val_dice": {"value": 0.1, "epoch": 1},
        "selection": {"metric": "val_dice", "value": 0.1, "epoch": 1},
    }
    generic.save_checkpoint(
        path,
        model,
        optimizer,
        None,
        0,
        args,
        [],
        best,
        adapter_config,
        checkpoint_metadata=runner.checkpoint_metadata(
            kind="selected_best",
            epoch=1,
            metric_value=0.1,
            selected=True,
            selection=selection,
            prereg_path=prereg_path,
            mode="smoke",
        ),
    )

    record = runner.strict_reload_checkpoint(path, selection)

    assert record["strict_state_load"] is True
    assert record["partial_unfreeze_policy_exact"] is True
    assert record["optimizer_groups_exact"] is True
