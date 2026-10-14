import json
import os
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from scripts import train_dpdsam_t1ce_a2 as a2_runner
from scripts import train_dpdsam_t1ce_b0 as b0_runner


class _FakeA2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sam_model = nn.Linear(3, 2)
        self.spatial_prompt_generator = nn.Linear(2, 2)


def _manifest() -> dict:
    model = {
        "schema": "aflow.sammed3d.t1ce_binary_a2_spatial_prompt",
        "adapter_config": {
            "schema_version": 1,
            "encoder_qv_lora": {"enabled": False, "rank": 0},
        },
    }
    return {
        "schema": a2_runner.TRAINING_SCHEMA,
        "schema_version": a2_runner.TRAINING_SCHEMA_VERSION,
        "args": {
            "seed": 20260722,
            "encoder_qv_lora_rank": 0,
            "encoder_qv_lora_alpha": 8.0,
        },
        "model": model,
        "optimizer": {
            "learning_rate": 4e-4,
            "weight_decay": 0.1,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "amsgrad": False,
        },
    }


def _clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _payload(
    model: _FakeA2,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    manifest: dict,
) -> dict:
    return {
        "training_schema": a2_runner.TRAINING_SCHEMA,
        "training_schema_version": a2_runner.TRAINING_SCHEMA_VERSION,
        "model_state_dict": _clone_state(model.sam_model),
        "a2_spatial_prompt_state_dict": _clone_state(model.spatial_prompt_generator),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": None,
        "epoch": 1,
        "history": [{"epoch": 1, "elapsed_seconds": 2.0}],
        "best": {
            "dsc": 0.4,
            "epoch": 1,
            "checkpoint": a2_runner.BEST_CHECKPOINT_NAME,
            "grid": "model_grid",
            "metric": "dsc",
        },
        "rng_state": b0_runner.capture_rng_state(generator),
        "implementation_manifest": manifest,
        "implementation_manifest_sha256": b0_runner.canonical_json_sha256(manifest),
        "dpdsam_binary_config": manifest["model"],
        "adapter_config": manifest["model"]["adapter_config"],
    }


def test_a2_parser_exposes_optional_encoder_lora_with_rank_zero_default() -> None:
    parser = a2_runner.build_parser()
    args = parser.parse_args(
        [
            "--train-jsonl",
            "train.jsonl",
            "--validation-jsonl",
            "validation.jsonl",
            "--split-manifest",
            "split.json",
            "--output-dir",
            "out",
        ]
    )
    assert args.encoder_qv_lora_rank == 0
    assert args.encoder_qv_lora_alpha == pytest.approx(8.0)
    overridden = parser.parse_args(
        [
            "--train-jsonl",
            "train.jsonl",
            "--validation-jsonl",
            "validation.jsonl",
            "--split-manifest",
            "split.json",
            "--output-dir",
            "out",
            "--encoder-qv-lora-rank",
            "4",
            "--encoder-qv-lora-alpha",
            "16",
        ]
    )
    assert overridden.encoder_qv_lora_rank == 4
    assert overridden.encoder_qv_lora_alpha == pytest.approx(16.0)


def test_a2_selection_contract_is_model_grid() -> None:
    metrics = {
        "model_grid": {"dsc": 0.62},
        "source_grid": {"dsc": 0.95},
    }
    assert b0_runner.validation_selection_score(metrics) == pytest.approx(0.62)
    assert a2_runner.BEST_CHECKPOINT_NAME == "best_model_grid_dsc.pth"


def test_a2_validation_is_image_first_and_emits_both_grids(monkeypatch) -> None:
    events: list[str] = []

    class FakeSupervisedCase:
        case_id = "BraTS-GLI-00001-100"
        t1ce_path = Path("image.nii.gz")

        @classmethod
        def from_manifest_row(cls, row):
            events.append("parse_row")
            return cls()

    class FakeInferenceCase:
        def __init__(self, *, case_id, t1ce_path) -> None:
            self.case_id = case_id
            self.t1ce_path = t1ce_path

    def load_inference(_case):
        events.append("load_image")
        return SimpleNamespace(image=torch.zeros(1, 2, 2, 2), geometry="same")

    def load_supervised(_case):
        events.append("load_label")
        return SimpleNamespace(geometry="same")

    metric = {
        "dsc": 0.5,
        "iou": 1 / 3,
        "hd95_mm": 2.0,
        "prediction_present": True,
        "target_present": True,
    }

    class Dual:
        def to_dict(self):
            return {
                "model_grid": dict(metric),
                "source_grid": dict(metric),
                "restore_mode": "nearest-exact",
            }

    def score(prediction, view):
        events.append("score")
        assert prediction.shape == (2, 2, 2)
        return Dual()

    class FakeModel(nn.Module):
        def forward(self, image):
            events.append("predict")
            return SimpleNamespace(probabilities=torch.ones(1, 1, 2, 2, 2))

    monkeypatch.setattr(a2_runner, "T1ceSupervisedCase", FakeSupervisedCase)
    monkeypatch.setattr(a2_runner, "T1ceInferenceCase", FakeInferenceCase)
    monkeypatch.setattr(a2_runner, "load_t1ce_inference_view", load_inference)
    monkeypatch.setattr(a2_runner, "load_t1ce_supervised_view", load_supervised)
    monkeypatch.setattr(a2_runner, "score_t1ce_dual_grid", score)

    summary = a2_runner.validate(FakeModel(), [{}], torch.device("cpu"), None)

    assert events == ["parse_row", "load_image", "predict", "load_label", "score"]
    assert summary["headline_metric"] == "model_grid.dsc"
    assert summary["model_grid"]["dsc"] == pytest.approx(0.5)
    assert summary["source_grid"]["dsc"] == pytest.approx(0.5)
    assert summary["case_metrics"][0]["restore_mode"] == "nearest-exact"


def test_a2_resume_restores_sam_spatial_optimizer_and_rng() -> None:
    random.seed(41)
    np.random.seed(42)
    torch.manual_seed(43)
    generator = torch.Generator().manual_seed(44)
    model = _FakeA2()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)
    loss = model.spatial_prompt_generator(model.sam_model(torch.ones(2, 3))).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    manifest = _manifest()
    payload = _payload(model, optimizer, generator, manifest)
    expected_sam = _clone_state(model.sam_model)
    expected_spatial = _clone_state(model.spatial_prompt_generator)
    expected_rng = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
        torch.rand(3, generator=generator),
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    generator.manual_seed(999)

    epoch, history, best = a2_runner.restore_training_checkpoint(
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
    for name, value in model.sam_model.state_dict().items():
        assert torch.equal(value, expected_sam[name])
    for name, value in model.spatial_prompt_generator.state_dict().items():
        assert torch.equal(value, expected_spatial[name])
    assert random.random() == expected_rng[0]
    assert float(np.random.random()) == expected_rng[1]
    assert torch.equal(torch.rand(3), expected_rng[2])
    assert torch.equal(torch.rand(3, generator=generator), expected_rng[3])


def test_a2_resume_rejects_missing_spatial_state_and_lora_drift() -> None:
    model = _FakeA2()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)
    generator = torch.Generator().manual_seed(51)
    manifest = _manifest()
    payload = _payload(model, optimizer, generator, manifest)
    del payload["a2_spatial_prompt_state_dict"]
    with pytest.raises(ValueError, match="a2_spatial_prompt_state_dict"):
        a2_runner.validate_resume_payload(payload, manifest)

    payload = _payload(model, optimizer, generator, manifest)
    changed = json.loads(json.dumps(manifest))
    changed["args"]["encoder_qv_lora_rank"] = 4
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        a2_runner.validate_resume_payload(payload, changed)


def test_worker_mode_dispatches_a2_and_forwards_all_arguments(tmp_path: Path) -> None:
    capture = tmp_path / "worker_args.json"
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "open(os.environ['CAPTURE'], 'w').write(json.dumps(sys.argv[1:]))\n",
        encoding="ascii",
    )
    fake_python.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            "bash",
            str(root / "scripts/run_worker_research.sh"),
            "dpdsam-t1ce-a2",
            "--epochs",
            "20",
            "--encoder-qv-lora-rank",
            "4",
        ],
        check=True,
        env={**os.environ, "AFLOW_PYTHON": str(fake_python), "CAPTURE": str(capture)},
    )
    assert json.loads(capture.read_text(encoding="utf-8")) == [
        "scripts/train_dpdsam_t1ce_a2.py",
        "--epochs",
        "20",
        "--encoder-qv-lora-rank",
        "4",
    ]
