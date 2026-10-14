from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from scripts.preflight_sammed3d_t1ce_rank0_matched import read_manifest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/preflight_sammed3d_t1ce_rank0_matched_v2.py"
PREREG = (
    ROOT
    / "workspace/BraTS/sammed3d_t1ce_wt/"
    "FV_RANK0_MATCHED_AUG_PREREG_20260724_V2.json"
)
LAUNCHER = ROOT / "scripts/launch_sammed3d_t1ce_rank0_matched_a10_v2.sh"


def load_v2_module():
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec = importlib.util.spec_from_file_location("rank0_preflight_v2", SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def manifest_row(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "subject_id": "BraTS-GLI-00001",
        "image_path": "/development/image.nii.gz",
        "label_path": "/development/label.nii.gz",
        "split": "train",
        "modality": "t1ce",
        "target": "WT",
        "foreground_labels": [1, 2, 3],
        "excluded_labels": [4],
    }


def test_frozen_v2_preflight_fails_closed_after_single_repo_migration() -> None:
    """A historical prereg must not silently authorize migrated source bytes.

    The July launcher is preserved as historical evidence, but its live entrypoint
    was made repo-local in August.  The old prereg SHA therefore must fail closed;
    a new run would require a new preregistration, never a rewritten old record.
    """
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    assert prereg["source"]["launcher"]["sha256"] == (
        "6d9cc84679bed8474d8f427543bf731602a06ca6b9084608429bffc5f4b45bf8"
    )
    with pytest.raises(RuntimeError, match="launcher byte SHA256 mismatch"):
        load_v2_module().run_preflight_v2(PREREG)


def test_manifest_duplicate_case_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.jsonl"
    row = manifest_row("case-1")
    path.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="duplicate case_id"):
        read_manifest(path, "train")


def test_recorded_source_hash_drift_fails_closed(tmp_path: Path) -> None:
    module = load_v2_module()
    path = tmp_path / "bound.txt"
    path.write_text("frozen\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="byte SHA256 mismatch"):
        module.validate_recorded_hash(
            tmp_path,
            {"path": str(path), "sha256": "0" * 64},
            "synthetic source",
        )


def test_launcher_is_exact_rank0_development_training_only() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for required in (
        "--encoder-qv-lora-rank 0",
        "--spatial-preprocessing whole_resize",
        "--train-augmentation affine_elastic_noise",
        "--epochs 50",
        "--lr 1e-4",
        "--weight-decay 0.1",
        "--seed 20260725",
        "--val-seed 20260723",
        "fv_rank0_matched_aug_train972_val108_e50_s20260725",
    ):
        assert required in text
    assert "evaluate_sammed3d" not in text
    assert "watch_t1ce" not in text
