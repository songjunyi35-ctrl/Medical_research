#!/usr/bin/env python3
"""Evaluate the authorized fixed epoch200 true-160^3 locator on test270 once."""
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

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator  # noqa: E402
from scripts.build_t1ce_wt_braincrop_test_cache_variable_resolution import (  # noqa: E402
    AUTHORIZATION,
    TEST_CACHE_SCHEMA,
)
from scripts.train_t1ce_wt_capg3d import (  # noqa: E402
    sha256_file,
    subject_id,
    validate,
    write_json_atomic,
)
from scripts.train_t1ce_wt_plain_resolution_full_cohort import (  # noqa: E402
    CHECKPOINT_SCHEMA,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_true160_epoch200_test270"
CHECKPOINT_SHA256 = "a25e8519b3f9a66ee5c389ad94f0e62ac548dfd6339468637c8f2c3536139b67"
TEST_MANIFEST_SHA256 = "a5ddc0ed2c0b269431758b14061da3c6163dd0c48133b60a2ef8efd7f23aa8f9"
TRAIN_MANIFEST_SHA256 = "a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16"
BOOTSTRAP_SEED = 20260831
BOOTSTRAP_SAMPLES = 10_000


class Test160Dataset(Dataset):
    def __init__(self, cache: Path) -> None:
        self.cache = cache.resolve()
        self.index = json.loads((self.cache / "index.json").read_text(encoding="utf-8"))
        if self.index.get("schema") != TEST_CACHE_SCHEMA:
            raise ValueError("unexpected true-160 test cache schema")
        if self.index.get("test_access") is not True:
            raise ValueError("test cache does not declare authorized test access")
        if self.index.get("outer_selector_access") is not False:
            raise ValueError("test cache accessed outer selector")
        if self.index.get("confirmation_access") is not False:
            raise ValueError("test cache accessed confirmation")
        if tuple(self.index.get("model_shape", ())) != (160, 160, 160):
            raise ValueError("test cache is not true-160^3")
        if self.index.get("preprocessing", {}).get("source_is_existing_128_cache") is not False:
            raise ValueError("test cache was not built from raw NIfTI")
        if self.index.get("raw_reproduction", {}).get("all_exact") is not True:
            raise ValueError("test cache lacks exact raw reproduction proof")
        self.count = int(self.index["count"])
        self.case_ids = [str(value) for value in self.index["case_ids"]]
        self._images: np.memmap | None = None
        self._labels: np.memmap | None = None
        voxels = 160 ** 3
        if (self.cache / "images.f32").stat().st_size != self.count * voxels * 4:
            raise ValueError("test image blob byte size mismatch")
        if (self.cache / "labels.u8").stat().st_size != self.count * voxels:
            raise ValueError("test label blob byte size mismatch")

    def _open(self) -> None:
        if self._images is None:
            self._images = np.memmap(
                self.cache / "images.f32", dtype=np.float32, mode="r",
                shape=(self.count, 160, 160, 160)
            )
            self._labels = np.memmap(
                self.cache / "labels.u8", dtype=np.uint8, mode="r",
                shape=(self.count, 160, 160, 160)
            )

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        self._open()
        assert self._images is not None and self._labels is not None
        case_id = self.case_ids[index]
        return {
            "image": torch.from_numpy(np.array(self._images[index], dtype=np.float32))[None],
            "label": torch.from_numpy(
                np.array(self._labels[index], dtype=np.uint8).astype(np.float32)
            )[None],
            "case_id": case_id,
            "subject_id": subject_id(case_id)
        }


def _bootstrap(values: list[float], rng: np.random.Generator) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    draws = rng.integers(0, len(array), size=(BOOTSTRAP_SAMPLES, len(array)))
    return [float(value) for value in np.quantile(array[draws].mean(axis=1), [0.025, 0.975])]


def confidence_intervals(summary: dict[str, Any]) -> dict[str, Any]:
    scan = [float(item["locator_dsc"]) for item in summary["per_case"]]
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in summary["per_case"]:
        grouped[str(item["subject_id"])].append(float(item["locator_dsc"]))
    subjects = [statistics.fmean(values) for values in grouped.values()]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return {
        "method": "percentile bootstrap",
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "scan_dsc_mean_95ci": _bootstrap(scan, rng),
        "subject_dsc_mean_95ci": _bootstrap(subjects, rng)
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.authorization != AUTHORIZATION:
        raise RuntimeError("explicit true-160 epoch200 test270 authorization is required")
    output = args.output.resolve()
    started = output.with_suffix(output.suffix + ".started.json")
    if output.exists() or started.exists():
        raise FileExistsError("refusing repeated true-160 epoch200 test270 evaluation")
    if sha256_file(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("epoch200 checkpoint SHA-256 mismatch")
    if sha256_file(args.test_manifest) != TEST_MANIFEST_SHA256:
        raise ValueError("test270 manifest SHA-256 mismatch")
    if sha256_file(args.train_manifest) != TRAIN_MANIFEST_SHA256:
        raise ValueError("full train manifest SHA-256 mismatch")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unexpected full-cohort checkpoint schema")
    if checkpoint.get("epoch") != 200 or checkpoint.get("resolution") != 160:
        raise ValueError("checkpoint is not fixed true-160 epoch200")
    if any(
        checkpoint.get(key) is not False
        for key in ("validation_access", "outer_selector_access", "confirmation_access", "test_access")
    ):
        raise ValueError("checkpoint declares prior evaluation access")
    if checkpoint.get("data_role", {}).get("train", {}).get("sha256") != TRAIN_MANIFEST_SHA256:
        raise ValueError("checkpoint is not bound to the full 756-case train cohort")

    test_rows = [
        json.loads(line) for line in args.test_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_rows = [
        json.loads(line) for line in args.train_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    test_ids = {str(row["case_id"]) for row in test_rows}
    train_ids = {str(row["case_id"]) for row in train_rows}
    test_subjects = {str(row["subject_id"]) for row in test_rows}
    train_subjects = {str(row["subject_id"]) for row in train_rows}
    if len(test_ids) != 270 or len(test_subjects) != 119:
        raise ValueError("unexpected test270 case/subject count")
    if test_ids & train_ids or test_subjects & train_subjects:
        raise ValueError("test270 overlaps the full train cohort")

    data = Test160Dataset(args.cache)
    if data.case_ids != [str(row["case_id"]) for row in test_rows]:
        raise ValueError("test cache order differs from the manifest")
    if data.index.get("manifest_sha256") != TEST_MANIFEST_SHA256:
        raise ValueError("test cache manifest binding mismatch")
    for filename, field in (("images.f32", "images_sha256"), ("labels.u8", "labels_sha256")):
        if sha256_file(args.cache / filename) != data.index[field]:
            raise ValueError(f"test cache {filename} SHA-256 mismatch")

    write_json_atomic(started, {
        "schema": SCHEMA + ".started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": AUTHORIZATION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "fixed_epoch": 200,
        "resolution": 160,
        "test_manifest_sha256": TEST_MANIFEST_SHA256,
        "fixed_logit_threshold": 0.0,
        "tta": False,
        "postprocessing": False,
        "checkpoint_selection_uses_test": False
    })
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("test270 evaluation requires CUDA")
    torch.cuda.set_device(device)
    architecture = checkpoint["architecture"]
    model = AutomaticPromptLocator(
        center_aware=False,
        init_filters=int(architecture["init_filters"]),
        dropout=float(architecture["dropout"])
    ).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != 4_700_897:
        raise ValueError("unexpected locator parameter count")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0
    )
    summary = validate(model, loader, device)
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "fixed epoch200 evaluation on a reused internal test set",
        "not_pristine_blind_test": True,
        "not_external_confirmation": True,
        "authorization": AUTHORIZATION,
        "protocol": {
            "model": "plain 4.70M-parameter MONAI SegResNet locator",
            "input": "T1ce only",
            "geometry": "raw-NIfTI image-derived nonzero brain crop resized to true-160^3",
            "threshold": "raw segmentation logit > 0",
            "checkpoint": "fixed epoch200 full-756 training output",
            "tta": False,
            "postprocessing": False,
            "sam": False
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": CHECKPOINT_SHA256,
            "epoch": 200,
            "resolution": 160
        },
        "data": {
            "manifest": str(args.test_manifest.resolve()),
            "manifest_sha256": TEST_MANIFEST_SHA256,
            "cache": str(args.cache.resolve()),
            "cache_index_sha256": sha256_file(args.cache / "index.json"),
            "cache_images_sha256": data.index["images_sha256"],
            "cache_labels_sha256": data.index["labels_sha256"],
            "cases": 270,
            "subjects": 119,
            "case_overlap_with_train": 0,
            "subject_overlap_with_train": 0
        },
        "summary": summary,
        "bootstrap": confidence_intervals(summary),
        "grid_comparability_note": (
            "headline Dice is measured on the true-160 model grid; historical 128 results "
            "used a 128 grid, so cross-resolution deltas include label rasterization differences"
        ),
        "outer_selector_access_during_test": False,
        "confirmation_access": False,
        "test_access": True
    }
    write_json_atomic(output, report)
    print(json.dumps({
        "scan_dsc_mean": summary["locator_dsc_mean"],
        "subject_dsc_mean": summary["locator_subject_dsc_mean"],
        "automatic_point_hit_rate": summary["automatic_point_hit_rate"],
        "cases": summary["cases"],
        "subjects": summary["subjects"]
    }, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
