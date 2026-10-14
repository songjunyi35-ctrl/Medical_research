#!/usr/bin/env python3
"""Rebuild all 75 external inputs in /tmp and compare with the frozen cache."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.prepare_ucsd_ptgbm_external_confirmation as prepare


def compare_tensors(left: Path, right: Path, count: int, shape: tuple[int, int, int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, dtype in (("images.f32", np.float32), ("labels.u8", np.uint8)):
        a = np.memmap(left / name, dtype=dtype, mode="r", shape=(count, *shape))
        b = np.memmap(right / name, dtype=dtype, mode="r", shape=(count, *shape))
        exact = 0
        max_abs = 0.0
        mismatches = []
        for slot in range(count):
            same = bool(np.array_equal(a[slot], b[slot]))
            exact += same
            if not same:
                max_abs = max(max_abs, float(np.max(np.abs(a[slot].astype(np.float64) - b[slot].astype(np.float64)))))
                mismatches.append(slot)
        result[name] = {"byte_exact_cases": exact, "max_abs_delta": max_abs, "mismatch_slots": mismatches}
    result["all_tensors_byte_exact"] = all(item["byte_exact_cases"] == count for key, item in result.items() if key != "all_tensors_byte_exact")
    return result


def verify(args: argparse.Namespace) -> dict[str, Any]:
    formal = args.formal_cache.resolve()
    formal_index_path = formal / "index.json"
    if prepare.sha256_file(formal_index_path) != args.expected_formal_index_sha256:
        raise RuntimeError("formal external cache index SHA-256 mismatch")
    formal_index = json.loads(formal_index_path.read_text())
    if formal_index.get("split") != "external_confirmation" or formal_index.get("count") != 75 or formal_index.get("subjects") != 53:
        raise RuntimeError("formal external cache contract mismatch")
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="aflow-ucsd75-rebuild-", dir="/tmp") as temporary:
        rebuilt = Path(temporary) / "cache128"
        build_args = argparse.Namespace(
            inventory=args.inventory,
            expected_inventory_sha256=args.expected_inventory_sha256,
            download_dir=args.download_dir,
            expected_download_receipt_sha256=args.expected_download_receipt_sha256,
            output_dir=rebuilt,
            workers=args.workers,
        )
        rebuilt_index = prepare.build_cache(build_args)
        if rebuilt_index["case_ids"] != formal_index["case_ids"] or rebuilt_index["subject_ids"] != formal_index["subject_ids"]:
            raise RuntimeError("rebuilt cache ID order differs from formal cache")
        comparison = compare_tensors(formal, rebuilt, 75, (128, 128, 128))
        rebuilt_hashes = {"images_sha256": rebuilt_index["images_sha256"], "labels_sha256": rebuilt_index["labels_sha256"]}
        temporary_path = str(Path(temporary))
    report = {
        "schema": "aflow.ucsd_ptgbm.external_cache_full_reproduction",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_cache": str(formal),
        "formal_index_sha256": args.expected_formal_index_sha256,
        "cases": 75,
        "subjects": 53,
        "rebuilt_hashes": rebuilt_hashes,
        "formal_hashes": {"images_sha256": formal_index["images_sha256"], "labels_sha256": formal_index["labels_sha256"]},
        "comparison": comparison,
        "temporary_directory": temporary_path,
        "temporary_directory_removed": not Path(temporary_path).exists(),
        "formal_cache_modified": False,
        "model_inference": False,
        "new_external_cohort_access": False,
        "reused_test270_access": False,
        "elapsed_seconds": time.time() - started,
        "runner_sha256": prepare.sha256_file(Path(__file__).resolve()),
    }
    if not comparison["all_tensors_byte_exact"] or rebuilt_hashes != report["formal_hashes"] or not report["temporary_directory_removed"]:
        raise RuntimeError("full external cache reproduction failed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--expected-download-receipt-sha256", required=True)
    parser.add_argument("--formal-cache", type=Path, required=True)
    parser.add_argument("--expected-formal-index-sha256", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite reproduction report: {output}")
    report = verify(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({"cases": report["cases"], "comparison": report["comparison"], "temporary_directory_removed": report["temporary_directory_removed"], "model_inference": False}, indent=2))


if __name__ == "__main__":
    main()
