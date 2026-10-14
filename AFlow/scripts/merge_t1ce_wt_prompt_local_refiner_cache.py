#!/usr/bin/env python3
"""Merge deterministic prompt-local cache shards into source-cache order."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_cached import MODEL_SHAPE  # noqa: E402
from scripts.evaluate_sammed3d_t1ce_wt_checkpoint import (  # noqa: E402
    sha256_file,
    write_json_atomic,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_prompt_local_refiner_input_cache"
MERGED_SCHEMA = "aflow.sammed3d.t1ce_wt_prompt_local_refiner_merged_cache"


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    shard_dirs = [path.resolve() for path in args.shard_dirs]
    indexes = [json.loads((path / "index.json").read_text()) for path in shard_dirs]
    if any(index.get("schema") != SCHEMA for index in indexes):
        raise ValueError("unexpected shard schema")
    if any(index.get("test_access") is not False for index in indexes):
        raise ValueError("test cache shard is forbidden")
    split = indexes[0]["split"]
    source_cache = indexes[0]["source_cache"]
    sam = indexes[0]["sam"]
    global_refiner = indexes[0]["global_refiner"]
    for index in indexes[1:]:
        if index["split"] != split or index["source_cache"] != source_cache:
            raise RuntimeError("shards do not share one source cache")
        if index["sam"]["checkpoint_sha256"] != sam["checkpoint_sha256"]:
            raise RuntimeError("SAM checkpoint mismatch across shards")
        if index["global_refiner"]["checkpoint_sha256"] != global_refiner["checkpoint_sha256"]:
            raise RuntimeError("global refiner mismatch across shards")

    count = int(source_cache["count"])
    source_index = json.loads((Path(source_cache["path"]) / "index.json").read_text())
    sam_tmp = output_dir / "sam_logits.f16.tmp"
    refined_tmp = output_dir / "global_refined_logits.f16.tmp"
    sam_out = np.memmap(sam_tmp, dtype=np.float16, mode="w+", shape=(count, *MODEL_SHAPE))
    refined_out = np.memmap(
        refined_tmp, dtype=np.float16, mode="w+", shape=(count, *MODEL_SHAPE)
    )
    points_out = np.full((count, 3, 3), -1, dtype=np.int16)
    labels_out = np.full((count, 3), -1, dtype=np.int8)
    case_records: list[dict[str, Any] | None] = [None] * count
    seen: set[int] = set()

    for shard_dir, index in zip(shard_dirs, indexes, strict=True):
        shard_count = int(index["count"])
        if sha256_file(shard_dir / "sam_logits.f16") != index["sam"]["logits_sha256"]:
            raise RuntimeError(f"SAM logit hash mismatch in {shard_dir}")
        if (
            sha256_file(shard_dir / "global_refined_logits.f16")
            != index["global_refiner"]["logits_sha256"]
        ):
            raise RuntimeError(f"global-refiner logit hash mismatch in {shard_dir}")
        if sha256_file(shard_dir / "click_points.i16.npy") != index["click_points_sha256"]:
            raise RuntimeError(f"click-point hash mismatch in {shard_dir}")
        if sha256_file(shard_dir / "click_labels.i8.npy") != index["click_labels_sha256"]:
            raise RuntimeError(f"click-label hash mismatch in {shard_dir}")
        sam_in = np.memmap(
            shard_dir / "sam_logits.f16", dtype=np.float16, mode="r",
            shape=(shard_count, *MODEL_SHAPE),
        )
        refined_in = np.memmap(
            shard_dir / "global_refined_logits.f16", dtype=np.float16, mode="r",
            shape=(shard_count, *MODEL_SHAPE),
        )
        points_in = np.load(shard_dir / "click_points.i16.npy", allow_pickle=False)
        labels_in = np.load(shard_dir / "click_labels.i8.npy", allow_pickle=False)
        for local_slot, source_slot in enumerate(index["source_slots"]):
            source_slot = int(source_slot)
            if source_slot in seen:
                raise RuntimeError(f"duplicate source slot {source_slot}")
            if index["case_ids"][local_slot] != source_index["case_ids"][source_slot]:
                raise RuntimeError("case/source ordering mismatch")
            seen.add(source_slot)
            sam_out[source_slot] = sam_in[local_slot]
            refined_out[source_slot] = refined_in[local_slot]
            points_out[source_slot] = points_in[local_slot]
            labels_out[source_slot] = labels_in[local_slot]
            case_records[source_slot] = index["cases"][local_slot]
    if seen != set(range(count)) or any(record is None for record in case_records):
        raise RuntimeError(f"incomplete merged cache: {len(seen)}/{count}")

    sam_out.flush()
    refined_out.flush()
    del sam_out, refined_out
    sam_path = output_dir / "sam_logits.f16"
    refined_path = output_dir / "global_refined_logits.f16"
    os.replace(sam_tmp, sam_path)
    os.replace(refined_tmp, refined_path)
    np.save(output_dir / "click_points.i16.npy", points_out, allow_pickle=False)
    np.save(output_dir / "click_labels.i8.npy", labels_out, allow_pickle=False)
    cases = [record for record in case_records if record is not None]
    report = {
        "schema": MERGED_SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "test_access": False,
        "count": count,
        "case_ids": list(source_index["case_ids"]),
        "source_cache": source_cache,
        "sam": sam | {"logits_sha256": sha256_file(sam_path)},
        "global_refiner": global_refiner
        | {"logits_sha256": sha256_file(refined_path)},
        "click_points_sha256": sha256_file(output_dir / "click_points.i16.npy"),
        "click_labels_sha256": sha256_file(output_dir / "click_labels.i8.npy"),
        "sam_dsc_mean": float(np.mean([case["sam_dsc"] for case in cases])),
        "global_refined_dsc_mean": float(
            np.mean([case["global_refined_dsc"] for case in cases])
        ),
        "shards": [
            {"path": str(path), "index_sha256": sha256_file(path / "index.json")}
            for path in shard_dirs
        ],
        "cases": cases,
        "merger_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_atomic(output_dir / "index.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
