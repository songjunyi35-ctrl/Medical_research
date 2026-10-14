from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts import watch_t1ce_fv_lora_then_test as watcher


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_verify_training_process_rejects_pid_reuse(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        watcher,
        "process_command",
        lambda pid: "python unrelated.py",
    )
    with pytest.raises(RuntimeError, match="reused"):
        watcher.verify_training_process(123, tmp_path / "expected")


def test_completed_training_requires_full_frozen_contract(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path
    run = root / "run"
    run.mkdir()
    trainer = root / "trainer.py"
    preprocessing = root / "preprocessing.py"
    trainer.write_text("trainer", encoding="utf-8")
    preprocessing.write_text("preprocessing", encoding="utf-8")
    prereg = root / "prereg.json"
    write_json(
        prereg,
        {
            "status": "frozen_before_full_training",
            "output_dir": "run",
            "source": {
                "trainer": "trainer.py",
                "trainer_sha256": watcher.sha256_file(trainer),
                "preprocessing_module": "preprocessing.py",
                "preprocessing_module_sha256": watcher.sha256_file(preprocessing),
            },
        },
    )
    write_json(
        run / "t1ce_wt_protocol.json",
        {
            "preprocessing": {
                "geometry": {"name": "whole_resize"},
                "augmentation": {"name": "affine_elastic_noise"},
            },
            "test_access_during_training": False,
        },
    )
    config_args = {
        "epochs": 50,
        "strategy": "decoder_only",
        "encoder_qv_lora_rank": 4,
        "encoder_qv_lora_alpha": 8.0,
        "num_clicks": 3,
        "click_method": "random",
        "max_train_cases": 0,
        "max_val_cases": 0,
        "seed": 20260725,
        "val_seed": 20260723,
        "selection_metric": "val_dice",
    }
    write_json(
        run / "train_config.json",
        {
            "train_rows": 972,
            "val_rows": 108,
            "trainable_parameters": 7731760,
            "args": config_args,
        },
    )
    best = {
        "selection": {
            "metric": "val_dice",
            "checkpoint": "sam_model_val_dice_best.pth",
            "epoch": 7,
            "value": 0.6,
        }
    }
    write_json(run / "best.json", best)
    latest_path = run / "sam_model_latest.pth"
    selected_path = run / "sam_model_val_dice_best.pth"
    latest_path.write_bytes(b"latest")
    selected_path.write_bytes(b"selected")
    log = root / "train.log"
    log.write_text("TRAIN_DONE", encoding="utf-8")
    adapter = {
        "encoder_qv_lora": {"enabled": True, "rank": 4, "alpha": 8.0}
    }
    latest = {
        "epoch": 50,
        "history": [{}] * 50,
        "best": best,
        "adapter_config": adapter,
    }
    selected = {
        "epoch": 7,
        "checkpoint_metadata": {
            "kind": "selected_best",
            "metric": "val_dice",
            "metric_epoch": 7,
        },
        "adapter_config": adapter,
    }
    monkeypatch.setattr(
        watcher.torch,
        "load",
        lambda path, **kwargs: latest
        if Path(path).name == "sam_model_latest.pth"
        else selected,
    )
    args = argparse.Namespace(
        root=root,
        training_output=Path("run"),
        training_log=log,
        prereg=prereg,
    )

    verified = watcher.verify_completed_training(args)

    assert verified["best"] == best
    assert verified["checkpoint"] == selected_path.resolve()
    assert verified["adapter_config"] == adapter


def test_completed_training_rejects_partial_epoch(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    trainer = tmp_path / "trainer.py"
    preprocessing = tmp_path / "preprocessing.py"
    trainer.write_text("trainer", encoding="utf-8")
    preprocessing.write_text("preprocessing", encoding="utf-8")
    prereg = tmp_path / "prereg.json"
    write_json(
        prereg,
        {
            "status": "frozen_before_full_training",
            "output_dir": "run",
            "source": {
                "trainer": "trainer.py",
                "trainer_sha256": watcher.sha256_file(trainer),
                "preprocessing_module": "preprocessing.py",
                "preprocessing_module_sha256": watcher.sha256_file(preprocessing),
            },
        },
    )
    write_json(
        run / "t1ce_wt_protocol.json",
        {
            "preprocessing": {
                "geometry": {"name": "whole_resize"},
                "augmentation": {"name": "affine_elastic_noise"},
            },
            "test_access_during_training": False,
        },
    )
    write_json(
        run / "train_config.json",
        {
            "train_rows": 972,
            "val_rows": 108,
            "trainable_parameters": 7731760,
            "args": {
                "epochs": 50,
                "strategy": "decoder_only",
                "encoder_qv_lora_rank": 4,
                "encoder_qv_lora_alpha": 8.0,
                "num_clicks": 3,
                "click_method": "random",
                "max_train_cases": 0,
                "max_val_cases": 0,
                "seed": 20260725,
                "val_seed": 20260723,
                "selection_metric": "val_dice",
            },
        },
    )
    best = {
        "selection": {
            "metric": "val_dice",
            "checkpoint": "sam_model_val_dice_best.pth",
            "epoch": 7,
        }
    }
    write_json(run / "best.json", best)
    (run / "sam_model_latest.pth").write_bytes(b"latest")
    (run / "sam_model_val_dice_best.pth").write_bytes(b"selected")
    log = tmp_path / "train.log"
    log.write_text("TRAIN_DONE", encoding="utf-8")
    monkeypatch.setattr(
        watcher.torch,
        "load",
        lambda *args, **kwargs: {
            "epoch": 49,
            "history": [{}] * 49,
            "best": best,
            "adapter_config": {
                "encoder_qv_lora": {"enabled": True, "rank": 4, "alpha": 8.0}
            },
        },
    )
    args = argparse.Namespace(
        root=tmp_path,
        training_output=Path("run"),
        training_log=log,
        prereg=prereg,
    )

    with pytest.raises(RuntimeError, match="latest.epoch mismatch"):
        watcher.verify_completed_training(args)
