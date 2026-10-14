#!/usr/bin/env python3
"""One-time evaluation of the selector-frozen plain locator on test270.

This evaluator is deliberately test-only. It accepts one frozen manifest and one
selector-chosen checkpoint, and exposes no threshold, TTA, postprocessing, or
checkpoint-selection controls.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_cached import CACHE_SCHEMA, MODEL_SHAPE
from scripts.evaluate_t1ce_wt_plain_locator_confirmation import load_plain_locator
from scripts.train_t1ce_wt_capg3d import (
    read_case_ids,
    sha256_file,
    subject_id,
    validate,
    write_json_atomic,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_locator_test270"
AUTHORIZATION = "USER_AUTHORIZED_PLAIN_LOCATOR_TEST270_2026-08-17"
TEST_MANIFEST_SHA256 = "a5ddc0ed2c0b269431758b14061da3c6163dd0c48133b60a2ef8efd7f23aa8f9"
CHECKPOINT_SHA256 = "0bb7ca52d7ac30068f752a4fc05a859f4dce42dce6ace40ba1f6b8b332889c2f"
TEST_CASES = 270
TEST_SUBJECTS = 119
BOOTSTRAP_SEED = 20260817
BOOTSTRAP_SAMPLES = 10_000


def verified_role_cases(metadata: dict[str, Any] | None, *, role: str) -> list[str]:
    if not isinstance(metadata, dict):
        raise ValueError(f"checkpoint lacks frozen {role} role metadata")
    path = Path(str(metadata["path"]))
    if sha256_file(path) != str(metadata["sha256"]):
        raise RuntimeError(f"checkpoint-bound {role} role manifest SHA-256 mismatch")
    values = read_case_ids(path)
    assert values is not None
    return values


def bootstrap_interval(values: list[float], rng: np.random.Generator) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    draws = rng.integers(0, len(array), size=(BOOTSTRAP_SAMPLES, len(array)))
    means = array[draws].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def intervals(result: dict[str, Any]) -> dict[str, Any]:
    cases = list(result["per_case"])
    scan_values = [float(case["locator_dsc"]) for case in cases]
    by_subject: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        by_subject[str(case["subject_id"])].append(float(case["locator_dsc"]))
    subject_values = [statistics.fmean(values) for values in by_subject.values()]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return {
        "method": "percentile bootstrap",
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "scan_dsc_mean_95ci": bootstrap_interval(scan_values, rng),
        "subject_dsc_mean_95ci": bootstrap_interval(subject_values, rng),
    }


class TestBrainCropDataset(Dataset):
    """Read only a fully audited image-derived test cache."""

    def __init__(self, cache_dir: Path, expected_case_ids: list[str]) -> None:
        self.cache_dir = cache_dir.resolve()
        self.index = json.loads((self.cache_dir / "index.json").read_text())
        if self.index.get("schema") != CACHE_SCHEMA:
            raise ValueError("unexpected input cache schema")
        if self.index.get("split") != "test" or self.index.get("test_access") is not True:
            raise ValueError("test evaluator requires a declared test cache")
        preprocessing = self.index.get("preprocessing", {})
        required = {
            "spatial": "braincrop_resize",
            "uses_gt_for_localization": False,
            "n4_bias_field_correction": True,
            "n4_shrink_factor": 2,
        }
        for key, expected in required.items():
            if preprocessing.get(key) != expected:
                raise ValueError(f"test cache preprocessing mismatch for {key}")
        self.case_ids = [str(value) for value in self.index["case_ids"]]
        if self.case_ids != expected_case_ids:
            raise ValueError("test cache case order differs from frozen manifest")
        self.count = int(self.index["count"])
        if self.count != TEST_CASES or len(self.case_ids) != self.count:
            raise ValueError("test cache count mismatch")
        self._images: np.memmap | None = None
        self._labels: np.memmap | None = None

    def _open(self) -> None:
        if self._images is None:
            self._images = np.memmap(
                self.cache_dir / "images.f32", dtype=np.float32, mode="r",
                shape=(self.count, *MODEL_SHAPE),
            )
            self._labels = np.memmap(
                self.cache_dir / "labels.u8", dtype=np.uint8, mode="r",
                shape=(self.count, *MODEL_SHAPE),
            )

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, slot: int) -> dict[str, torch.Tensor | str]:
        self._open()
        assert self._images is not None and self._labels is not None
        case_id = self.case_ids[slot]
        return {
            "image": torch.from_numpy(np.array(self._images[slot], dtype=np.float32))[None],
            "label": torch.from_numpy(np.array(self._labels[slot], dtype=np.float32))[None],
            "case_id": case_id,
            "subject_id": subject_id(case_id),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--locator-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if args.authorization != AUTHORIZATION:
        raise RuntimeError("explicit test270 authorization is required")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite test report: {output}")
    manifest = args.test_manifest.resolve()
    if sha256_file(manifest) != TEST_MANIFEST_SHA256:
        raise RuntimeError("test manifest SHA-256 mismatch")
    checkpoint = args.locator_checkpoint.resolve()
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("selector-frozen checkpoint SHA-256 mismatch")

    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    case_ids = [str(row["case_id"]) for row in rows]
    row_subjects = [str(row["subject_id"]) for row in rows]
    if len(rows) != TEST_CASES or len(set(case_ids)) != TEST_CASES:
        raise RuntimeError("test manifest case count or uniqueness mismatch")
    if len(set(row_subjects)) != TEST_SUBJECTS:
        raise RuntimeError("test manifest subject count mismatch")
    for row in rows:
        if row.get("split") != "test" or set(row.get("modalities", {})) != {"t1c"}:
            raise RuntimeError("test manifest row violates frozen T1ce/test projection")
        if not row.get("seg_path"):
            raise RuntimeError("test manifest row lacks ground truth")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("test270 evaluation requires CUDA")
    torch.cuda.set_device(device)
    model, payload = load_plain_locator(checkpoint, device)
    if int(payload["epoch"]) != 99:
        raise RuntimeError("unexpected selector-chosen epoch")
    roles = payload.get("data_roles", {})
    train_cases = verified_role_cases(roles.get("train_case_ids"), role="train")
    selector_cases = verified_role_cases(
        roles.get("validation_case_ids"), role="selector"
    )
    development_cases = set(train_cases) | set(selector_cases)
    development_subjects = {subject_id(case_id) for case_id in development_cases}
    test_subjects = set(row_subjects)
    if development_cases & set(case_ids) or development_subjects & test_subjects:
        raise RuntimeError("test overlaps checkpoint train/selector roles")

    data = TestBrainCropDataset(args.cache, case_ids)
    if data.index.get("manifest_sha256") != TEST_MANIFEST_SHA256:
        raise RuntimeError("test cache is bound to the wrong manifest")
    for filename, field in (("images.f32", "images_sha256"), ("labels.u8", "labels_sha256")):
        if sha256_file(args.cache / filename) != str(data.index[field]):
            raise RuntimeError(f"test cache {filename} SHA-256 mismatch")

    started_path = output.with_suffix(output.suffix + ".started.json")
    write_json_atomic(
        started_path,
        {
            "schema": SCHEMA + ".started",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "authorization": AUTHORIZATION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "selected_epoch": 99,
            "test_manifest_sha256": TEST_MANIFEST_SHA256,
            "fixed_logit_threshold": 0.0,
            "checkpoint_selection_uses_test": False,
        },
    )
    loader = DataLoader(
        data, batch_size=min(args.batch_size, len(data)), shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    result = validate(model, loader, device)
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "supplementary evaluation on a previously revealed internal test set",
        "not_pristine_blind_test": True,
        "not_external_confirmation": True,
        "test_access": True,
        "authorization": AUTHORIZATION,
        "protocol": {
            "model": "plain 4.70M-parameter MONAI SegResNet locator",
            "input": "T1ce only",
            "geometry": "image-derived nonzero brain crop resized to 128^3",
            "threshold": "raw segmentation logit > 0, fixed before test",
            "checkpoint_selection": "epoch 99; highest selector scan-macro Dice",
            "test_use": "metrics only; no tuning, threshold choice, or checkpoint selection",
            "tta": False,
            "postprocessing": False,
        },
        "checkpoint": {
            "path": str(checkpoint), "sha256": CHECKPOINT_SHA256,
            "selected_epoch": int(payload["epoch"]),
        },
        "data": {
            "manifest": str(manifest), "manifest_sha256": TEST_MANIFEST_SHA256,
            "cache": str(args.cache.resolve()),
            "cache_index_sha256": sha256_file(args.cache / "index.json"),
            "cache_images_sha256": data.index["images_sha256"],
            "cache_labels_sha256": data.index["labels_sha256"],
            "cases": len(case_ids), "subjects": len(test_subjects),
            "case_overlap_with_train_selector": 0,
            "subject_overlap_with_train_selector": 0,
        },
        "summary": result,
        "bootstrap": intervals(result),
    }
    write_json_atomic(output, report)
    print(json.dumps({
        "scan_dsc_mean": result["locator_dsc_mean"],
        "subject_dsc_mean": result["locator_subject_dsc_mean"],
        "automatic_point_hit_rate": result["automatic_point_hit_rate"],
        "cases": result["cases"], "subjects": result["subjects"],
    }, indent=2))


if __name__ == "__main__":
    main()
