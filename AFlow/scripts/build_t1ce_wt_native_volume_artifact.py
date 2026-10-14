#!/usr/bin/env python3
"""Build frozen train-only native-grid WT volumes for 972-case sampling."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_variable_resolution import VariableResolutionBrainCropDataset  # noqa: E402
from scripts.train_t1ce_wt_capg3d import read_case_ids, sha256_file, write_json_atomic  # noqa: E402


SCHEMA = "aflow.brats.t1ce_wt_native_volume_artifact"
CANDIDATE_MANIFEST_SHA256 = "6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65"
CONTROL_MANIFEST_SHA256 = "a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16"
HISTORICAL_INDEX_SHA256 = "d26b823a4715d9f3676d75cddabcf630180d48807352c23042be7faf240b622d"
CANDIDATE_CACHE_INDEX_SHA256 = "7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd"
IDEA_SHA256 = "b90b5b340d58b5589859695b0c7515e6a256e9ae4bd8e6a9824cae07a1539856"
PREREG_SHA256 = "1a0d4cd398b1233ceac4571b900783f34e0a84e74a4b836b3ef1502050281d0f"
AMENDMENT_SHA256 = "5a84420082b4165f8db2cc4e8861d95c5a2b3191fb9b51f8948b3d150fb11b72"
AMENDMENT_002_SHA256 = "9d59b490b533244fba01dba1554a13d80ffdd67cc672659743a116a879154b38"


def tensor_sha256(value: torch.Tensor) -> str:
    import hashlib

    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def native_sampling_weights(case_ids: list[str], volumes: dict[str, int]) -> torch.Tensor:
    values = np.asarray([float(volumes[case_id]) for case_id in case_ids], dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any(values <= 0):
        raise ValueError("native volume vector must be one-dimensional and positive")
    median = float(np.median(values))
    weights = np.sqrt(median / values)
    return torch.as_tensor(np.clip(weights, 0.5, 4.0), dtype=torch.double)


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--control-manifest", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--historical-formal-cache", type=Path, required=True)
    parser.add_argument("--idea", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing output: {args.output}")
    expected = {
        args.candidate_manifest: CANDIDATE_MANIFEST_SHA256,
        args.control_manifest: CONTROL_MANIFEST_SHA256,
        args.candidate_cache / "index.json": CANDIDATE_CACHE_INDEX_SHA256,
        args.historical_formal_cache / "index.json": HISTORICAL_INDEX_SHA256,
        args.idea: IDEA_SHA256,
        args.prereg: PREREG_SHA256,
        args.amendment: AMENDMENT_SHA256,
        args.amendment2: AMENDMENT_002_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"frozen input SHA mismatch: {path}")

    rows = read_manifest(args.candidate_manifest)
    if len(rows) != 972:
        raise ValueError("candidate manifest must contain972 rows")
    ids = [str(row["case_id"]) for row in rows]
    if len(set(ids)) != 972:
        raise ValueError("candidate manifest has duplicate case IDs")
    control_ids = read_case_ids(args.control_manifest)
    if control_ids is None or len(control_ids) != 756 or not set(control_ids) < set(ids):
        raise ValueError("control IDs are not a strict 756-case subset")
    candidate_index = json.loads(
        (args.candidate_cache / "index.json").read_text(encoding="utf-8")
    )
    crop_geometry = candidate_index.get("crop_geometry", {})
    if set(crop_geometry) != set(ids):
        raise ValueError("candidate cache crop geometry does not cover exactly972 cases")
    native_volumes: dict[str, int] = {}
    full_native_volumes: dict[str, int] = {}
    native_shapes: dict[str, list[int]] = {}
    source_labels: dict[str, str] = {}
    source_orientations: dict[str, dict[str, list[str]]] = {}
    def process(row: dict) -> tuple[str, int, int, list[int], str, list[str], list[str]]:
        case_id = str(row["case_id"])
        if row.get("split") != "train" or row.get("target") != "WT":
            raise ValueError(f"unexpected train role/target for {case_id}")
        label_path = Path(str(row["label_path"]))
        if "sealed_test" in str(label_path).lower() or not label_path.is_file():
            raise ValueError(f"invalid or missing train label: {label_path}")
        stored_image = nib.load(str(label_path))
        stored_orientation = [str(value) for value in nib.aff2axcodes(stored_image.affine)]
        image = nib.as_closest_canonical(stored_image)
        canonical_orientation = [str(value) for value in nib.aff2axcodes(image.affine)]
        if canonical_orientation != ["R", "A", "S"]:
            raise ValueError(f"canonical orientation is not RAS for {case_id}")
        array = np.asanyarray(image.dataobj)
        if array.ndim != 3 or not np.isfinite(array).all():
            raise ValueError(f"invalid label array for {case_id}")
        rounded = np.rint(array)
        if not np.array_equal(array, rounded):
            raise ValueError(f"non-integer label values for {case_id}")
        labels = rounded.astype(np.int16, copy=False)
        unique = set(int(value) for value in np.unique(labels))
        if not unique <= {0, 1, 2, 3, 4}:
            raise ValueError(f"unexpected label values for {case_id}: {unique}")
        full_volume = int(np.count_nonzero((labels == 1) | (labels == 2) | (labels == 3)))
        geometry = crop_geometry[case_id]
        native_shape = [int(value) for value in labels.shape]
        if [int(value) for value in geometry.get("native_shape", [])] != native_shape:
            raise ValueError(f"crop native_shape mismatch for {case_id}")
        origin = [int(value) for value in geometry["origin"]]
        extent = [int(value) for value in geometry["extent"]]
        if len(origin) != 3 or len(extent) != 3 or any(value <= 0 for value in extent):
            raise ValueError(f"invalid crop geometry for {case_id}")
        stop = [origin[axis] + extent[axis] for axis in range(3)]
        if any(origin[axis] < 0 or stop[axis] > native_shape[axis] for axis in range(3)):
            raise ValueError(f"out-of-bounds crop geometry for {case_id}")
        box = tuple(slice(origin[axis], stop[axis]) for axis in range(3))
        cropped = labels[box]
        volume = int(np.count_nonzero((cropped == 1) | (cropped == 2) | (cropped == 3)))
        if full_volume <= 0 or volume <= 0:
            raise ValueError(f"empty native WT for {case_id}")
        return (
            case_id, volume, full_volume, native_shape, str(label_path),
            stored_orientation, canonical_orientation,
        )

    if args.workers <= 0:
        raise ValueError("workers must be positive")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        processed = executor.map(process, rows)
        for index, item in enumerate(processed):
            (
                case_id, volume, full_volume, native_shape, label_path,
                stored_orientation, canonical_orientation,
            ) = item
            native_volumes[case_id] = volume
            full_native_volumes[case_id] = full_volume
            native_shapes[case_id] = native_shape
            source_labels[case_id] = label_path
            source_orientations[case_id] = {
                "stored": stored_orientation,
                "canonical": canonical_orientation,
            }
            if index % 100 == 0:
                print(
                    f"processed={index}/{len(rows)} case={case_id} "
                    f"crop_native_wt={volume} full_native_wt={full_volume}",
                    flush=True,
                )

    historical_index = json.loads(
        (args.historical_formal_cache / "index.json").read_text(encoding="utf-8")
    )
    historical = historical_index["per_case_source_positive_voxels"]
    mismatches = {
        case_id: {"generated": native_volumes[case_id], "historical": int(historical[case_id])}
        for case_id in control_ids
        if native_volumes[case_id] != int(historical[case_id])
    }
    generated_control_weights = native_sampling_weights(control_ids, native_volumes)
    historical_dataset = VariableResolutionBrainCropDataset(
        args.historical_formal_cache,
        expected_resolution=128,
        expected_manifest_sha256=CONTROL_MANIFEST_SHA256,
        case_ids=control_ids,
    )
    historical_weights = historical_dataset.sampling_weights()
    weight_max_abs = float((generated_control_weights - historical_weights).abs().max())
    weights_exact = torch.equal(generated_control_weights, historical_weights)
    if mismatches or not weights_exact:
        raise RuntimeError(
            f"native-volume parity failed: mismatches={len(mismatches)} weight_max_abs={weight_max_abs}"
        )
    candidate_weights = native_sampling_weights(ids, native_volumes)
    result = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "candidate_manifest": str(args.candidate_manifest),
        "candidate_manifest_sha256": CANDIDATE_MANIFEST_SHA256,
        "control_manifest": str(args.control_manifest),
        "control_manifest_sha256": CONTROL_MANIFEST_SHA256,
        "candidate_cache_index_sha256": CANDIDATE_CACHE_INDEX_SHA256,
        "historical_formal_index_sha256": HISTORICAL_INDEX_SHA256,
        "definition": {
            "target": "WT",
            "included_labels": [1, 2, 3],
            "excluded_labels": [0, 4],
            "volume_unit": "source NIfTI voxels inside frozen image-only brain crop",
            "crop_geometry_source": "candidate cache crop_geometry origin/extent",
            "weight_formula": "clip(sqrt(median(selected_native_volumes)/native_volume_i),0.5,4.0)",
            "weight_dtype": "torch.float64_cpu",
        },
        "counts": {
            "candidate_cases": len(ids),
            "control_cases": len(control_ids),
            "all_native_volumes_positive": all(value > 0 for value in native_volumes.values()),
        },
        "native_wt_voxels": native_volumes,
        "full_fov_native_wt_voxels_audit_only": full_native_volumes,
        "native_shapes": native_shapes,
        "source_label_paths": source_labels,
        "source_orientations": source_orientations,
        "audit": {
            "overlap_cases": len(control_ids),
            "native_volume_mismatch_count": len(mismatches),
            "native_volume_mismatches": mismatches,
            "control_weight_values_exact": weights_exact,
            "control_weight_max_abs_difference": weight_max_abs,
            "control_weight_sha256": tensor_sha256(generated_control_weights),
            "historical_control_weight_sha256": tensor_sha256(historical_weights),
            "candidate_weight_sha256": tensor_sha256(candidate_weights),
            "control_weight_min": float(generated_control_weights.min()),
            "control_weight_max": float(generated_control_weights.max()),
            "candidate_weight_min": float(candidate_weights.min()),
            "candidate_weight_max": float(candidate_weights.max()),
            "all_gates_passed": not mismatches and weights_exact,
        },
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    write_json_atomic(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "cases": result["counts"],
        "audit": result["audit"],
        "output": str(args.output),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
