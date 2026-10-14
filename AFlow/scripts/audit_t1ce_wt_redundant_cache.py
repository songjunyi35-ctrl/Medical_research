#!/usr/bin/env python3
"""Prove that one train-only cache is a byte-exact subset of another; never delete."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np


class CacheAuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_cache(path: Path, expected_index_sha256: str, allowed_splits: set[str]) -> tuple[dict, np.memmap, np.memmap]:
    path = path.resolve()
    lowered = str(path).lower()
    if any(token in lowered for token in ("test270", "sealed_test", "confirmation")):
        raise CacheAuditError("only train-role caches may be audited")
    index_path = path / "index.json"
    if sha256_file(index_path) != expected_index_sha256:
        raise CacheAuditError(f"cache index SHA-256 mismatch: {path}")
    index = json.loads(index_path.read_text())
    if str(index.get("split")) not in allowed_splits or index.get("test_access") is True:
        raise CacheAuditError(f"cache is not an allowed train role: {path}")
    preprocessing = index.get("preprocessing", {})
    required = {"spatial": "braincrop_resize", "uses_gt_for_localization": False, "n4_bias_field_correction": True, "n4_shrink_factor": 2}
    for key, value in required.items():
        if preprocessing.get(key) != value:
            raise CacheAuditError(f"preprocessing mismatch for {key}: {path}")
    shape = tuple(int(value) for value in index["model_shape"])
    count = int(index["count"])
    images = np.memmap(path / "images.f32", dtype=np.float32, mode="r", shape=(count, *shape))
    labels = np.memmap(path / "labels.u8", dtype=np.uint8, mode="r", shape=(count, *shape))
    return index, images, labels


def audit(args: argparse.Namespace) -> dict:
    superset_path = args.superset.resolve()
    subset_path = args.subset.resolve()
    superset, superset_images, superset_labels = _load_cache(superset_path, args.expected_superset_index_sha256, {"train"})
    subset, subset_images, subset_labels = _load_cache(subset_path, args.expected_subset_index_sha256, {"historical_train_cohort_only"})
    if superset["model_shape"] != subset["model_shape"]:
        raise CacheAuditError("cache model shapes differ")
    superset_ids = [str(value) for value in superset["case_ids"]]
    subset_ids = [str(value) for value in subset["case_ids"]]
    if len(superset_ids) != len(set(superset_ids)) or len(subset_ids) != len(set(subset_ids)):
        raise CacheAuditError("cache contains duplicate case IDs")
    missing = sorted(set(subset_ids) - set(superset_ids))
    if missing:
        raise CacheAuditError(f"subset cache has cases absent from superset: {missing[:5]}")
    position = {case_id: slot for slot, case_id in enumerate(superset_ids)}
    image_exact = 0
    label_exact = 0
    max_image_abs_delta = 0.0
    mismatches = []
    started = time.time()
    for subset_slot, case_id in enumerate(subset_ids):
        superset_slot = position[case_id]
        left_image = np.asarray(subset_images[subset_slot])
        right_image = np.asarray(superset_images[superset_slot])
        left_label = np.asarray(subset_labels[subset_slot])
        right_label = np.asarray(superset_labels[superset_slot])
        same_image = bool(np.array_equal(left_image, right_image))
        same_label = bool(np.array_equal(left_label, right_label))
        image_exact += same_image
        label_exact += same_label
        if not same_image:
            delta = float(np.max(np.abs(left_image - right_image)))
            max_image_abs_delta = max(max_image_abs_delta, delta)
        if not same_image or not same_label:
            mismatches.append({"case_id": case_id, "image_exact": same_image, "label_exact": same_label})
        if (subset_slot + 1) % 100 == 0 or subset_slot + 1 == len(subset_ids):
            print(f"[{subset_slot + 1}/{len(subset_ids)}] image_exact={image_exact} label_exact={label_exact}", flush=True)
    reclaim_bytes = sum((subset_path / name).stat().st_size for name in ("images.f32", "labels.u8", "index.json"))
    report = {
        "schema": "aflow.brats.t1ce_wt.redundant_train_cache_audit",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "superset": {"path": str(superset_path), "index_sha256": args.expected_superset_index_sha256, "cases": len(superset_ids)},
        "subset": {"path": str(subset_path), "index_sha256": args.expected_subset_index_sha256, "cases": len(subset_ids)},
        "subset_case_ids_all_in_superset": True,
        "same_case_order": subset_ids == superset_ids[:len(subset_ids)],
        "image_byte_exact_cases": image_exact,
        "label_byte_exact_cases": label_exact,
        "all_tensors_byte_exact": image_exact == len(subset_ids) and label_exact == len(subset_ids),
        "max_image_abs_delta": max_image_abs_delta,
        "mismatches": mismatches,
        "audited_bytes": len(subset_ids) * int(np.prod(subset["model_shape"])) * 5,
        "candidate_reclaim_bytes": reclaim_bytes,
        "rebuildable_from_raw_and_archived_builder": True,
        "deletion_performed": False,
        "test_or_confirmation_cache_access": False,
        "model_inference": False,
        "elapsed_seconds": time.time() - started,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--superset", type=Path, required=True)
    parser.add_argument("--expected-superset-index-sha256", required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--expected-subset-index-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite cache audit: {output}")
    report = audit(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({key: report[key] for key in ("subset_case_ids_all_in_superset", "image_byte_exact_cases", "label_byte_exact_cases", "all_tensors_byte_exact", "candidate_reclaim_bytes", "deletion_performed")}, indent=2))


if __name__ == "__main__":
    main()
