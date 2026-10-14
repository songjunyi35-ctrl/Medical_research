from argparse import Namespace
from collections import Counter, defaultdict
import csv
import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sammed3d import peft
from scripts import train_sammed3d_brats as supervised


@pytest.mark.parametrize(
    ("amp_dtype", "no_amp", "device", "expected"),
    [
        ("fp16", False, "cuda", torch.float16),
        ("bf16", False, "cuda", torch.bfloat16),
        ("none", False, "cuda", None),
        ("bf16", True, "cuda", None),
        ("fp16", False, "cpu", None),
    ],
)
def test_resolve_amp_dtype(amp_dtype, no_amp, device, expected):
    args = Namespace(amp_dtype=amp_dtype, no_amp=no_amp)

    assert supervised.resolve_amp_dtype(args, torch.device(device)) == expected


def test_grad_scaler_is_created_only_for_cuda_fp16(monkeypatch):
    sentinel = object()
    calls = []

    def fake_grad_scaler(device_type):
        calls.append(device_type)
        return sentinel

    monkeypatch.setattr(supervised.torch.amp, "GradScaler", fake_grad_scaler)

    assert supervised.make_grad_scaler(torch.device("cuda"), torch.float16) is sentinel
    assert supervised.make_grad_scaler(torch.device("cuda"), torch.bfloat16) is None
    assert supervised.make_grad_scaler(torch.device("cuda"), None) is None
    assert supervised.make_grad_scaler(torch.device("cpu"), torch.float16) is None
    assert calls == ["cuda"]


def test_fixed_validation_rng_repeats_and_restores_training_rng():
    supervised.seed_everything(17)
    outer_numpy_state = np.random.get_state()
    outer_torch_state = torch.random.get_rng_state()

    with supervised.fixed_rng_seed(9000, torch.device("cpu")):
        first = (np.random.randint(1000, size=5), torch.rand(5))
    with supervised.fixed_rng_seed(9000, torch.device("cpu")):
        second = (np.random.randint(1000, size=5), torch.rand(5))

    np.testing.assert_array_equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    restored_numpy_state = np.random.get_state()
    assert restored_numpy_state[0] == outer_numpy_state[0]
    np.testing.assert_array_equal(restored_numpy_state[1], outer_numpy_state[1])
    assert restored_numpy_state[2:] == outer_numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), outer_torch_state)


def _factorial_sequence(extra_initializations: int):
    supervised.seed_everything(11)
    for _ in range(extra_initializations):
        nn.Embedding(6, 8)
        np.random.random(7)

    # This is the post-model-construction reset used by main().
    supervised.seed_everything(2026)
    loader = DataLoader(
        TensorDataset(torch.arange(18)),
        batch_size=3,
        shuffle=True,
        generator=supervised.make_seeded_generator(2027),
    )
    order = [int(value) for batch in loader for value in batch[0]]
    click_draws = np.random.randint(10_000, size=8).tolist()
    model_draws = torch.rand(8).tolist()
    return order, click_draws, model_draws


def test_post_construction_reseed_isolates_factorial_rng_consumption():
    assert _factorial_sequence(0) == _factorial_sequence(5)


def test_prepare_output_dir_rejects_nonempty_by_default(tmp_path):
    output_dir = tmp_path / "run"
    supervised.prepare_output_dir(output_dir, allow_nonempty=False)
    (output_dir / "history.csv").write_text("epoch\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="output directory is not empty"):
        supervised.prepare_output_dir(output_dir, allow_nonempty=False)

    supervised.prepare_output_dir(output_dir, allow_nonempty=True)


def _rows(target_counts):
    return [
        {
            "target": target,
            "case_id": f"{target}-{index:02d}",
            "image_path": f"/{target}/{index}/image.npy",
            "label_path": f"/{target}/{index}/label.npy",
        }
        for target, count in target_counts.items()
        for index in range(count)
    ]


def _patient_rows(spec):
    rows = []
    for target, patients in spec.items():
        for patient_id, timepoints in patients.items():
            for timepoint in range(timepoints):
                case_id = f"{patient_id}-{timepoint:03d}"
                rows.append(
                    {
                        "target": target,
                        "patient_id": patient_id,
                        "case_id": case_id,
                        "image_path": f"/{case_id}/image.npy",
                        "label_path": f"/{target}/{case_id}/label.npy",
                    }
                )
    return rows


def test_target_balanced_subset_is_deterministic_and_not_target_sort_biased():
    rows = _rows({"A": 7, "B": 7, "C": 7})

    first = supervised.filter_rows(
        rows,
        targets=None,
        max_cases=6,
        balance_targets=True,
        seed=2026,
    )
    second = supervised.filter_rows(
        list(reversed(rows)),
        targets=None,
        max_cases=6,
        balance_targets=True,
        seed=2026,
    )

    assert first == second
    assert Counter(row["target"] for row in first) == {"A": 2, "B": 2, "C": 2}
    sorted_subset = supervised.filter_rows(rows, None, 6, balance_targets=False)
    assert Counter(row["target"] for row in sorted_subset) == {"A": 6}


def test_rows_per_target_and_single_target_legacy_order():
    balanced = supervised.filter_rows(
        _rows({"A": 4, "B": 4, "C": 4}),
        None,
        None,
        balance_targets=True,
        rows_per_target=1,
        seed=77,
    )
    assert Counter(row["target"] for row in balanced) == {"A": 1, "B": 1, "C": 1}

    single = list(reversed(_rows({"A": 5})))
    limited = supervised.filter_rows(
        single,
        None,
        2,
        balance_targets=True,
        seed=77,
    )
    assert [row["case_id"] for row in limited] == ["A-00", "A-01"]


def test_balanced_sampler_has_equal_target_mass_and_seeded_order():
    rows = _rows({"A": 6, "B": 2, "C": 1})
    weights = supervised.target_sampling_weights(rows, "balanced")
    mass = defaultdict(float)
    for row, weight in zip(rows, weights.tolist()):
        mass[row["target"]] += weight

    assert mass == pytest.approx({"A": 1.0, "B": 1.0, "C": 1.0})
    first = list(supervised.make_train_sampler(rows, "balanced", 99, 30))
    second = list(supervised.make_train_sampler(rows, "balanced", 99, 30))
    assert first == second
    assert len(first) == 30


def test_target_patient_weights_balance_targets_and_patients_within_target():
    rows = _patient_rows(
        {
            "A": {"p1": 3, "p2": 1},
            "B": {"p3": 2},
        }
    )
    weights = supervised.target_sampling_weights(
        rows,
        "target_patient_balanced",
    ).tolist()
    target_mass = defaultdict(float)
    cell_mass = defaultdict(float)
    for row, weight in zip(rows, weights):
        target_mass[row["target"]] += weight
        cell_mass[(row["target"], row["patient_id"])] += weight

    assert target_mass == pytest.approx({"A": 1.0, "B": 1.0})
    assert cell_mass == pytest.approx(
        {("A", "p1"): 0.5, ("A", "p2"): 0.5, ("B", "p3"): 1.0}
    )
    p1_weights = [
        weight
        for row, weight in zip(rows, weights)
        if row["target"] == "A" and row["patient_id"] == "p1"
    ]
    assert p1_weights == pytest.approx([1.0 / 6.0] * 3)


def test_target_patient_sampling_strictly_validates_patient_rows():
    rows = _patient_rows({"A": {"p1": 2}})
    missing_patient = [dict(row) for row in rows]
    del missing_patient[0]["patient_id"]
    with pytest.raises(ValueError, match="patient_id"):
        supervised.target_sampling_weights(
            missing_patient,
            "target_patient_balanced",
        )

    conflicting_patient = [dict(row) for row in rows]
    conflict = dict(conflicting_patient[0])
    conflict["target"] = "B"
    conflict["patient_id"] = "p2"
    conflicting_patient.append(conflict)
    with pytest.raises(ValueError, match="conflicting patient_id"):
        supervised.ExactTargetPatientSampler(conflicting_patient, seed=3)

    with pytest.raises(ValueError, match=r"duplicate \(target, case_id\)"):
        supervised.ExactTargetPatientSampler(rows + [dict(rows[0])], seed=3)


def test_exact_sampler_default_rounds_down_to_equal_target_quota():
    assert supervised.resolve_exact_samples_per_epoch(4431, 6, None) == 4428
    with pytest.raises(ValueError, match="must be divisible"):
        supervised.resolve_exact_samples_per_epoch(4431, 6, 4431)

    rows = _patient_rows(
        {
            "A": {"p1": 3, "p2": 1},
            "B": {"p3": 2, "p4": 1},
            "C": {"p5": 2, "p6": 2},
        }
    )
    sampler = supervised.ExactTargetPatientSampler(rows, seed=2026)
    schedule = list(sampler)
    metadata = sampler.metadata_for_epoch(0)

    assert len(rows) == 11
    assert len(sampler) == len(schedule) == 9
    assert Counter(rows[index]["target"] for index in schedule) == {
        "A": 3,
        "B": 3,
        "C": 3,
    }
    assert metadata["target_quota"] == {"A": 3, "B": 3, "C": 3}
    assert set(metadata["target_patient_exposure"]) == {"A", "B", "C"}
    for exposure in metadata["target_patient_exposure"].values():
        assert exposure == {"patients": 2, "min": 1, "max": 2}


def test_exact_sampler_is_epoch_aware_paired_and_cycles_timepoints():
    rows = _patient_rows(
        {
            "A": {"p1": 3, "p2": 1},
            "B": {"p3": 2, "p4": 2},
        }
    )
    d0 = supervised.ExactTargetPatientSampler(rows, seed=77, samples_per_epoch=8)
    l0 = supervised.ExactTargetPatientSampler(rows, seed=77, samples_per_epoch=8)

    for epoch in (0, 1):
        d0.set_epoch(epoch)
        l0.set_epoch(epoch)
        assert list(d0) == list(l0)
        assert (
            d0.metadata_for_epoch(epoch)["schedule_sha256"]
            == l0.metadata_for_epoch(epoch)["schedule_sha256"]
        )

    epoch0 = d0.indices_for_epoch(0)
    epoch1 = d0.indices_for_epoch(1)
    assert epoch0 != epoch1
    assert (
        d0.metadata_for_epoch(0)["schedule_sha256"]
        != d0.metadata_for_epoch(1)["schedule_sha256"]
    )
    for epoch_schedule in (epoch0, epoch1):
        assert Counter(rows[index]["target"] for index in epoch_schedule) == {
            "A": 4,
            "B": 4,
        }
        assert d0.metadata_for_epoch(
            0 if epoch_schedule is epoch0 else 1
        )["target_patient_exposure"] == {
            "A": {"patients": 2, "min": 2, "max": 2},
            "B": {"patients": 2, "min": 2, "max": 2},
        }

    p1_epoch0 = {
        rows[index]["case_id"]
        for index in epoch0
        if rows[index]["target"] == "A" and rows[index]["patient_id"] == "p1"
    }
    p1_epoch1 = {
        rows[index]["case_id"]
        for index in epoch1
        if rows[index]["target"] == "A" and rows[index]["patient_id"] == "p1"
    }
    assert len(p1_epoch0) == len(p1_epoch1) == 2
    assert len(p1_epoch0 | p1_epoch1) == 3


def test_exact_sampler_different_seeds_change_schedule_and_hash():
    rows = _patient_rows(
        {
            "A": {"p1": 3, "p2": 2, "p3": 1},
            "B": {"p4": 4, "p5": 2, "p6": 1},
            "C": {"p7": 3, "p8": 2, "p9": 2},
        }
    )
    seed2026 = supervised.ExactTargetPatientSampler(
        rows,
        seed=2026,
        samples_per_epoch=18,
    )
    seed2027 = supervised.ExactTargetPatientSampler(
        rows,
        seed=2027,
        samples_per_epoch=18,
    )

    for epoch in (0, 1):
        assert seed2026.indices_for_epoch(epoch) != seed2027.indices_for_epoch(epoch)
        assert (
            seed2026.metadata_for_epoch(epoch)["schedule_sha256"]
            != seed2027.metadata_for_epoch(epoch)["schedule_sha256"]
        )


def test_natural_sampler_default_behavior_is_unchanged():
    assert supervised.make_train_sampler(_rows({"A": 3}), "natural", 7, None) is None


class MetricOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Identity()


def test_validation_metrics_are_sample_weighted_and_report_target_macro(monkeypatch):
    losses = iter(
        [
            torch.tensor([1.0, 3.0]),
            torch.tensor([5.0]),
        ]
    )

    def fake_interaction_forward(**kwargs):
        per_sample = next(losses).to(kwargs["image"])
        logits = torch.full_like(kwargs["gt"], 10.0)
        if len(per_sample) == 2:
            logits[1] = -10.0
        return logits, per_sample.mean(), per_sample

    monkeypatch.setattr(supervised, "interaction_forward", fake_interaction_forward)
    loader = [
        {
            "image": torch.zeros(2, 1, 1, 1, 1),
            "label": torch.ones(2, 1, 1, 1, 1),
            "target": ["A", "B"],
        },
        {
            "image": torch.zeros(1, 1, 1, 1, 1),
            "label": torch.ones(1, 1, 1, 1, 1),
            "target": ["B"],
        },
    ]

    metrics = supervised.run_epoch(
        model=MetricOnlyModel(),
        loader=loader,
        loss_fn=None,
        device=torch.device("cpu"),
        num_clicks=1,
        click_method="random",
        train_image_encoder=False,
    )

    assert metrics["loss"] == pytest.approx(3.0)
    assert metrics["dice"] == pytest.approx(2.0 / 3.0)
    assert metrics["macro_dice"] == pytest.approx(0.75)
    assert metrics["per_target"]["A"] == pytest.approx(
        {"loss": 1.0, "dice": 1.0, "samples": 1, "dice_samples": 1}
    )
    assert metrics["per_target"]["B"] == pytest.approx(
        {"loss": 4.0, "dice": 0.5, "samples": 2, "dice_samples": 2}
    )


def test_macro_dice_ignores_targets_without_valid_dice_samples():
    metrics = {
        "A": {"dice": 0.8, "dice_samples": 2},
        "empty": {"dice": 0.0, "dice_samples": 0},
    }

    assert supervised.macro_dice_from_per_target(metrics) == pytest.approx(0.8)


def test_selection_metric_direction_and_checkpoint_name():
    assert supervised.metric_is_better("val_loss", 0.4, 0.5)
    assert supervised.metric_is_better("val_dice", 0.6, 0.5)
    assert supervised.metric_is_better("val_macro_dice", 0.6, 0.5)
    assert not supervised.metric_is_better("val_loss", 0.6, 0.5)
    assert (
        supervised.selection_checkpoint_name("val_macro_dice")
        == "sam_model_val_macro_dice_best.pth"
    )
    assert (
        supervised.selection_checkpoint_name("val_dice")
        == "sam_model_val_dice_best.pth"
    )


def test_history_csv_writes_one_header_and_macro_metric(tmp_path):
    path = tmp_path / "history.csv"
    row = {
        "epoch": 1,
        "train_loss": 0.8,
        "train_dice": 0.2,
        "train_schedule_sha256": "abc123",
        "train_target_quota": {"A": 4},
        "train_target_patient_exposure": {
            "A": {"patients": 2, "min": 2, "max": 2}
        },
        "val_loss": 0.7,
        "val_dice": 0.4,
        "val_macro_dice": 0.5,
        "val_by_target": {"A": {"dice": 0.5}},
        "elapsed_sec": 1.0,
    }

    supervised.append_history_csv(path, row)
    supervised.append_history_csv(path, {**row, "epoch": 2})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert sum(line.startswith("epoch,") for line in lines) == 1
    with path.open(newline="", encoding="utf-8") as f:
        parsed = list(csv.DictReader(f))
    assert [entry["epoch"] for entry in parsed] == ["1", "2"]
    assert [entry["val_macro_dice"] for entry in parsed] == ["0.5", "0.5"]
    assert [entry["train_schedule_sha256"] for entry in parsed] == [
        "abc123",
        "abc123",
    ]
    assert json.loads(parsed[0]["train_target_quota"]) == {"A": 4}
    assert json.loads(parsed[0]["train_target_patient_exposure"]) == {
        "A": {"patients": 2, "min": 2, "max": 2}
    }


class TinyPromptEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 2
        self.projection = nn.Linear(2, 2)


class TinyAdapterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Linear(2, 2)
        self.prompt_encoder = TinyPromptEncoder()
        self.mask_decoder = nn.Linear(2, 1)


class RecordingScaler:
    def state_dict(self):
        return {"scale": 1024.0}


def test_checkpoint_is_atomic_and_adapter_config_strictly_reloads(tmp_path):
    checkpoint = tmp_path / "sam_model_val_macro_dice_best.pth"
    config = peft.make_adapter_config(
        target_token=True,
        lora_rank=0,
        model_type="tiny",
        lora_blocks=(),
        target_vocab=("A", "B"),
        target_embedding_dim=2,
    )
    model = TinyAdapterModel()
    peft.apply_adapter_config(model, config)
    with torch.no_grad():
        model.target_embed.weight.normal_()
    expected_state = {key: value.clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    supervised.save_checkpoint(
        checkpoint,
        model,
        optimizer,
        RecordingScaler(),
        epoch=2,
        args=Namespace(seed=2026, amp_dtype="fp16"),
        history=[{"epoch": 1}],
        best={"selection": {"metric": "val_macro_dice", "value": 0.5}},
        adapter_config=config,
        checkpoint_metadata={"kind": "selected_best", "metric": "val_macro_dice"},
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["epoch"] == 3
    assert payload["scaler_state_dict"] == {"scale": 1024.0}
    assert payload["adapter_config"] == peft.normalize_adapter_config(config)
    assert payload["checkpoint_metadata"] == {
        "kind": "selected_best",
        "metric": "val_macro_dice",
    }
    assert list(tmp_path.iterdir()) == [checkpoint]

    loaded, effective = peft.load_model_with_adapters(
        checkpoint,
        lambda model_type: TinyAdapterModel() if model_type == "tiny" else None,
        device="cpu",
        requested_config=config,
        default_model_type="tiny",
    )
    assert effective == peft.normalize_adapter_config(config)
    assert loaded.state_dict().keys() == expected_state.keys()
    for key, expected in expected_state.items():
        assert torch.equal(loaded.state_dict()[key], expected)


def test_checkpoint_removes_partial_temp_file_on_failure(tmp_path, monkeypatch):
    checkpoint = tmp_path / "sam_model_latest.pth"
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    def failing_save(payload, path):
        path.write_bytes(b"partial")
        raise OSError("simulated write failure")

    monkeypatch.setattr(supervised.torch, "save", failing_save)

    with pytest.raises(OSError, match="simulated write failure"):
        supervised.save_checkpoint(
            checkpoint,
            model,
            optimizer,
            scaler=None,
            epoch=0,
            args=Namespace(seed=2026, amp_dtype="none"),
            history=[],
            best={},
        )

    assert not checkpoint.exists()
    assert list(tmp_path.iterdir()) == []
