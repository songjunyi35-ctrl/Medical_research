#!/usr/bin/env python3
"""Build T1ce-only VA-HDFS manifests without changing frozen source manifests."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = "/mnt/bn/liveaim/songjunyi/brats2024"
VA_ROOT = "/mnt/hdfs/sjy/datasets/brats2024_gli_t1ce_seg_v1"
OUTPUT_DIR = ROOT / "data/datasets/brats2024_t1ce_va_hdfs_20260812_v1"

SOURCES = {
    "train_t1ce_wt.jsonl": ROOT
    / "data/datasets/brats2024_sammed3d_t1ce_wt_inner90_10_seed20260723_v1/train_t1ce_wt.jsonl",
    "validation_t1ce_wt.jsonl": ROOT
    / "data/datasets/brats2024_sammed3d_t1ce_wt_inner90_10_seed20260723_v1/validation_t1ce_wt.jsonl",
    "dpdsam_train_t1ce_binary.jsonl": ROOT
    / "data/datasets/brats2024_dpdsam_t1ce_binary_seed20260721_v1/train_t1ce_binary.jsonl",
    "dpdsam_validation_t1ce_binary.jsonl": ROOT
    / "data/datasets/brats2024_dpdsam_t1ce_binary_seed20260721_v1/validation_t1ce_binary.jsonl",
    "outer_train_t1ce.jsonl": ROOT
    / "data/datasets/brats2024_random_subject_80_20_seed20260722_v1/train_cases.jsonl",
    "sealed_test/test_cases_t1ce.jsonl": ROOT
    / "data/datasets/brats2024_random_subject_80_20_seed20260722_v1/sealed_test/test_cases.jsonl",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relocate_path(value: Any) -> str:
    if not isinstance(value, str) or not value.startswith(f"{LEGACY_ROOT}/"):
        raise ValueError(f"expected a path below {LEGACY_ROOT}: {value!r}")
    return f"{VA_ROOT}/{value[len(LEGACY_ROOT) + 1:]}"


def relocate_row(row: dict[str, Any]) -> dict[str, Any]:
    relocated = dict(row)
    if "image_path" in relocated:
        relocated["image_path"] = relocate_path(relocated["image_path"])
    if "label_path" in relocated:
        relocated["label_path"] = relocate_path(relocated["label_path"])
    if "seg_path" in relocated:
        relocated["seg_path"] = relocate_path(relocated["seg_path"])
    if "source_case_dir" in relocated:
        relocated["source_case_dir"] = relocate_path(relocated["source_case_dir"])
    if "modalities" in relocated:
        modalities = relocated["modalities"]
        if not isinstance(modalities, dict) or "t1c" not in modalities:
            raise ValueError("outer manifest row is missing modalities.t1c")
        relocated["modalities"] = {"t1c": relocate_path(modalities["t1c"])}
    return relocated


def build() -> dict[str, Any]:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)
    outputs: dict[str, Any] = {}
    for relative, source in SOURCES.items():
        rows = [
            relocate_row(json.loads(line))
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        output = OUTPUT_DIR / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(", ", ": ")) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        outputs[relative] = {
            "count": len(rows),
            "sha256": sha256_file(output),
            "source": str(source.relative_to(ROOT)),
            "source_sha256": sha256_file(source),
        }

    manifest = {
        "schema": "aflow.brats2024.t1ce_storage_relocation",
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "active_t1ce_manifests",
        "operation": "path-only relocation plus removal of unused non-T1ce modality fields",
        "legacy_root": LEGACY_ROOT,
        "active_root": VA_ROOT,
        "frozen_sources_modified": False,
        "sealed_test_nifti_opened": False,
        "outputs": outputs,
    }
    manifest_path = OUTPUT_DIR / "storage_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
