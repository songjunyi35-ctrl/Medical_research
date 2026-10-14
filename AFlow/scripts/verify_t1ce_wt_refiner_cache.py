#!/usr/bin/env python3
"""Verify the persistent train/validation frozen-logit refiner cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_SHAPE = (128, 128, 128)
EXPECTED = {
    "train": {"count": 972, "coarse_dsc": 0.829776},
    "validation": {"count": 108, "coarse_dsc": 0.636588},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_recorded(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def verify_split(cache_root: Path, split: str, full_hash: bool) -> dict:
    cache_dir = cache_root / split
    index_path = cache_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema") != "aflow.sammed3d.t1ce_wt_refiner_input_cache":
        raise RuntimeError(f"unexpected cache schema: {index_path}")
    expected = EXPECTED[split]
    count = int(index["count"])
    if count != expected["count"] or len(index["case_ids"]) != count:
        raise RuntimeError(f"{split} count/case_ids mismatch")
    if index.get("test_access") is not False or index.get("split") != split:
        raise RuntimeError(f"{split} cache is not development-only")

    source = resolve_recorded(index["source_cache"]["path"])
    source_index = json.loads((source / "index.json").read_text(encoding="utf-8"))
    if list(source_index["case_ids"][:count]) != list(index["case_ids"]):
        raise RuntimeError(f"{split} source/refiner case ordering mismatch")
    for kind in ("images", "labels"):
        if source_index[f"{kind}_sha256"] != index["source_cache"][f"{kind}_sha256"]:
            raise RuntimeError(f"{split} source {kind} SHA binding mismatch")

    logits_path = cache_dir / "sam_logits.f16"
    expected_bytes = count * int(np.prod(MODEL_SHAPE)) * np.dtype(np.float16).itemsize
    if logits_path.stat().st_size != expected_bytes:
        raise RuntimeError(f"{split} logits size mismatch")
    logits = np.memmap(logits_path, dtype=np.float16, mode="r", shape=(count, *MODEL_SHAPE))
    for slot in sorted({0, count // 2, count - 1}):
        if not bool(np.isfinite(np.asarray(logits[slot])).all()):
            raise RuntimeError(f"{split} contains non-finite logits at slot {slot}")
    del logits

    coarse_dsc = float(index["baseline_click3"]["dsc_mean"])
    if abs(coarse_dsc - expected["coarse_dsc"]) > 5e-4:
        raise RuntimeError(
            f"{split} coarse DSC {coarse_dsc:.6f} is outside expected replay tolerance"
        )
    result = {
        "split": split,
        "count": count,
        "coarse_dsc": coarse_dsc,
        "index_sha256": sha256_file(index_path),
        "source": str(source),
        "logits_bytes": expected_bytes,
        "portable_source_path": not Path(index["source_cache"]["path"]).is_absolute(),
    }
    if full_hash:
        result["logits_sha256"] = sha256_file(logits_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "workspace/BraTS/cache/t1ce_wt_refiner_graymean_20260813",
    )
    parser.add_argument("--full-hash", action="store_true")
    args = parser.parse_args()
    result = {
        split: verify_split(args.cache_root.resolve(), split, args.full_hash)
        for split in ("train", "validation")
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
