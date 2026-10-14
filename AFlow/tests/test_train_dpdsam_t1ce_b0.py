import json
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from scripts.train_dpdsam_t1ce_b0 import (
    BEST_CHECKPOINT_NAME,
    TRAINING_SCHEMA,
    TRAINING_SCHEMA_VERSION,
    canonical_json_sha256,
    capture_rng_state,
    dice_bce_loss,
    read_rows,
    restore_rng_state,
    restore_training_checkpoint,
    summarize_dual_grid_cases,
    validate_resume_payload,
    validation_selection_score,
)


def test_dice_bce_loss_is_finite_and_backpropagates() -> None:
    logits = torch.zeros(2, 1, 4, 4, 4, requires_grad=True)
    target = torch.zeros_like(logits)
    target[:, :, 1:3, 1:3, 1:3] = 1

    loss, parts = dice_bce_loss(logits, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert parts["bce"] > 0
    assert parts["dice_loss"] > 0
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_read_rows_uses_fixed_hash_subset_and_split_contract(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [
        {
            "case_id": f"BraTS-GLI-{index:05d}-100",
            "subject_id": f"BraTS-GLI-{index:05d}",
            "split": "train",
            "schema": "aflow.brats2024.dpdsam_t1ce_binary_split",
            "modality": "t1ce",
            "image_path": f"/{index}-t1c.nii",
            "label_path": f"/{index}-seg.nii",
            "foreground_labels": [1, 2, 3],
            "excluded_labels": [4],
        }
        for index in range(5)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows)))

    selected = read_rows(
        path, 2, expected_split="train", subset_seed=17
    )

    assert len(selected) == 2
    assert selected == read_rows(path, 2, expected_split="train", subset_seed=17)


def test_rng_state_round_trip_restores_python_numpy_torch_and_loader_generator() -> None:
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    generator = torch.Generator().manual_seed(104)
    state = capture_rng_state(generator)

    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
        torch.rand(4, generator=generator),
    )
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    generator.manual_seed(999)

    restore_rng_state(state, generator)

    assert random.random() == expected[0]
    assert float(np.random.random()) == expected[1]
    assert torch.equal(torch.rand(4), expected[2])
    assert torch.equal(torch.rand(4, generator=generator), expected[3])


class _WorkerRandomDataset(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int):
        return (
            index,
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
        )


def _random_loader_epoch(generator: torch.Generator) -> list[list[torch.Tensor]]:
    loader = DataLoader(
        _WorkerRandomDataset(),
        batch_size=2,
        shuffle=True,
        num_workers=0,
        generator=generator,
        persistent_workers=False,
    )
    return [[value.clone() for value in batch] for batch in loader]


def test_loader_generator_restore_replays_shuffle_and_dataset_rng() -> None:
    random.seed(201)
    np.random.seed(202)
    torch.manual_seed(203)
    generator = torch.Generator().manual_seed(204)
    _random_loader_epoch(generator)
    state = capture_rng_state(generator)
    expected = _random_loader_epoch(generator)

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    generator.manual_seed(999)
    restore_rng_state(state, generator)
    resumed = _random_loader_epoch(generator)

    assert len(resumed) == len(expected)
    for resumed_batch, expected_batch in zip(resumed, expected):
        assert len(resumed_batch) == len(expected_batch)
        for resumed_value, expected_value in zip(resumed_batch, expected_batch):
            assert torch.equal(resumed_value, expected_value)


def _implementation_manifest() -> dict:
    model_metadata = {
        "arm": "B0_static_prompt",
        "adapter_config": {"schema_version": 1, "rank": 0},
    }
    return {
        "schema": TRAINING_SCHEMA,
        "schema_version": TRAINING_SCHEMA_VERSION,
        "args": {
            "seed": 20260722,
            "batch_size": 2,
            "checkpoint": "/frozen/base.pth",
        },
        "model": model_metadata,
        "data": {
            "train_manifest_sha256": "train-sha",
            "validation_manifest_sha256": "validation-sha",
            "split_manifest_sha256": "split-sha",
        },
        "optimizer": {
            "class": "torch.optim.AdamW",
            "learning_rate": 4e-4,
            "weight_decay": 0.1,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "amsgrad": False,
        },
        "source_checkpoint_sha256": "checkpoint-sha",
    }


class _TinyWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sam_model = nn.Linear(3, 2)


def _resume_payload(
    model: _TinyWrapper,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    manifest: dict,
) -> dict:
    return {
        "training_schema": TRAINING_SCHEMA,
        "training_schema_version": TRAINING_SCHEMA_VERSION,
        "model_state_dict": {
            name: value.detach().clone() for name, value in model.sam_model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": None,
        "epoch": 1,
        "history": [{"epoch": 1, "elapsed_seconds": 1.0}],
        "best": {"dsc": 0.25, "epoch": 1, "checkpoint": BEST_CHECKPOINT_NAME},
        "implementation_manifest": manifest,
        "implementation_manifest_sha256": canonical_json_sha256(manifest),
        "adapter_config": manifest["model"]["adapter_config"],
        "dpdsam_binary_config": manifest["model"],
        "rng_state": capture_rng_state(generator),
    }


def test_restore_checkpoint_restores_model_optimizer_history_and_rng() -> None:
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    generator = torch.Generator().manual_seed(14)
    manifest = _implementation_manifest()
    model = _TinyWrapper()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=4e-4, weight_decay=0.1
    )
    loss = model.sam_model(torch.ones(2, 3)).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    saved_weight = model.sam_model.weight.detach().clone()
    payload = _resume_payload(model, optimizer, generator, manifest)
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(3)
    expected_loader = torch.rand(3, generator=generator)

    with torch.no_grad():
        model.sam_model.weight.zero_()
    optimizer.param_groups[0]["lr"] = 0.25
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    generator.manual_seed(99)

    epoch, history, best = restore_training_checkpoint(
        payload,
        model=model,  # type: ignore[arg-type]
        optimizer=optimizer,
        scaler=None,
        generator=generator,
        implementation_manifest=manifest,
    )

    assert epoch == 1
    assert history == payload["history"]
    assert best == payload["best"]
    assert torch.equal(model.sam_model.weight, saved_weight)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(4e-4)
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)
    assert torch.equal(torch.rand(3, generator=generator), expected_loader)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 20260723),
        ("batch_size", 1),
        ("checkpoint", "/different/base.pth"),
    ],
)
def test_resume_rejects_critical_argument_drift(field: str, value) -> None:
    manifest = _implementation_manifest()
    model = _TinyWrapper()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)
    generator = torch.Generator().manual_seed(14)
    payload = _resume_payload(model, optimizer, generator, manifest)
    changed = json.loads(json.dumps(manifest))
    changed["args"][field] = value

    with pytest.raises(ValueError, match="manifest hash mismatch"):
        validate_resume_payload(payload, changed)


def test_resume_rejects_manifest_model_and_optimizer_drift() -> None:
    manifest = _implementation_manifest()
    model = _TinyWrapper()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)
    generator = torch.Generator().manual_seed(14)

    payload = _resume_payload(model, optimizer, generator, manifest)
    payload["adapter_config"] = {"schema_version": 1, "rank": 4}
    with pytest.raises(ValueError, match="adapter architecture mismatch"):
        validate_resume_payload(payload, manifest)

    payload = _resume_payload(model, optimizer, generator, manifest)
    payload["optimizer_state_dict"]["param_groups"][0]["weight_decay"] = 0.0
    with pytest.raises(ValueError, match="weight-decay mismatch"):
        restore_training_checkpoint(
            payload,
            model=model,  # type: ignore[arg-type]
            optimizer=optimizer,
            scaler=None,
            generator=generator,
            implementation_manifest=manifest,
        )


def _grid_metric(
    dsc: float, iou: float, hd95: float, prediction: bool, target: bool
) -> dict:
    return {
        "dsc": dsc,
        "iou": iou,
        "hd95_mm": hd95,
        "prediction_present": prediction,
        "target_present": target,
        "spacing_mm": [1.0, 1.0, 1.0],
        "empty_policy": "frozen",
    }


def test_dual_grid_summary_and_selection_use_model_grid_only() -> None:
    cases = [
        {
            "case_id": "BraTS-GLI-00001-100",
            "model_grid": _grid_metric(0.4, 0.25, 8.0, True, True),
            "source_grid": _grid_metric(0.9, 0.82, 2.0, True, True),
            "restore_mode": "nearest-exact",
            "headline_metric": "model_grid.dsc",
        },
        {
            "case_id": "BraTS-GLI-00002-100",
            "model_grid": _grid_metric(1.0, 1.0, 0.0, False, False),
            "source_grid": _grid_metric(0.0, 0.0, 9.0, False, True),
            "restore_mode": "nearest-exact",
            "headline_metric": "model_grid.dsc",
        },
    ]
    summary = summarize_dual_grid_cases(cases)

    assert summary["model_grid"] == {
        "dsc": pytest.approx(0.7),
        "iou": pytest.approx(0.625),
        "hd95_mm": pytest.approx(4.0),
        "both_empty": 1,
        "one_empty": 0,
    }
    assert summary["source_grid"]["dsc"] == pytest.approx(0.45)
    assert summary["source_grid"]["one_empty"] == 1
    assert validation_selection_score(summary) == pytest.approx(0.7)
    assert BEST_CHECKPOINT_NAME == "best_model_grid_dsc.pth"
