"""RHUH input-only no-N4 bridge QC; never load labels or model checkpoints."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from sammed3d.background_zero_decode import decode_background_zero

RECORD = ROOT / "archive/research_records/2026-09-05_rhuh_no_n4_input_bridge"
SOURCE_RECORD = ROOT / "archive/reviews/2026-09-05_30h_chain_audit"
IDEA_SHA = "5f5100b218ed8770106c712f93a3fabae994f31ffb7956e4acf2cffcbf2bf47c"
PREREG_SHA = "a94a46f278326655f5db18552e90ec533053cd3b6470d9ead58ec264faa8a36c"
PREACCESS_SHA = "65892e4d90a577ff2623cd2786d5142aeed86be809a5b273faf245321a4481c9"
DOWNLOAD_SHA = "c6f17decbc81e8a2bc8ec385438b7e83547644aed26009fb376bd27a218d725f"
OLD_FORMAT_SHA = "de0689c4008316b948f58bf4c07f439ca459deafc9d13468727c32ca7ca2849c"
DECODER_SHA = "de562395a0d733bc8d9c47bbd0bab6d9bf260af5d2c38913df7250a90397b476"
MODEL_SHAPE = (128, 128, 128)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def bbox(mask: np.ndarray) -> tuple[list[int], list[int]]:
    indices = np.nonzero(mask)
    if any(index.size == 0 for index in indices):
        raise ValueError("empty decoded image support")
    return [int(index.min()) for index in indices], [int(index.max()) + 1 for index in indices]


def bridge(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    decoded, background = decode_background_zero(array)
    support = decoded > 0
    origin, stop = bbox(support)
    values = decoded[support].astype(np.float64)
    mean = float(values.mean())
    std = float(values.std())
    if not np.isfinite(std) or std <= 1e-8:
        raise ValueError("invalid decoded foreground variance")
    normalized = np.zeros(decoded.shape, dtype=np.float32)
    normalized[support] = ((decoded[support].astype(np.float64) - mean) / std).astype(np.float32)
    normalized_values = normalized[support].astype(np.float64)
    slices = tuple(slice(low, high) for low, high in zip(origin, stop))
    image_tensor = torch.from_numpy(np.ascontiguousarray(normalized[slices])).reshape(1, 1, *normalized[slices].shape)
    mask_tensor = torch.from_numpy(np.ascontiguousarray(support[slices])).reshape(1, 1, *support[slices].shape)
    resized_image = F.interpolate(
        image_tensor, size=MODEL_SHAPE, mode="trilinear", align_corners=False
    )[0, 0]
    resized_mask = F.interpolate(mask_tensor.float(), size=MODEL_SHAPE, mode="nearest-exact")[0, 0].bool()
    resized_image.masked_fill_(~resized_mask, 0.0)
    output = np.ascontiguousarray(resized_image.numpy(), dtype=np.float32)
    output_mask = np.ascontiguousarray(resized_mask.numpy(), dtype=np.uint8)
    metadata = {
        "background": background,
        "decoded_background_exact_zero": bool(np.all(decoded[~support] == 0)),
        "source_positive_voxels": int(support.sum()),
        "bbox_origin": origin,
        "bbox_stop": stop,
        "bbox_extent": [high - low for low, high in zip(origin, stop)],
        "normalization_mean": mean,
        "normalization_population_std": std,
        "normalized_positive_mean": float(normalized_values.mean()),
        "normalized_positive_population_std": float(normalized_values.std()),
        "output_shape": [1, *MODEL_SHAPE],
        "output_finite": bool(np.isfinite(output).all()),
        "output_mask_voxels": int(output_mask.sum()),
        "output_image_sha256": array_sha(output),
        "output_mask_sha256": array_sha(output_mask),
    }
    return output, output_mask, metadata


def case_passed(metadata: dict, spacing: np.ndarray) -> bool:
    return bool(
        metadata["background"]["border_mode_fraction"] >= 0.95
        and metadata["decoded_background_exact_zero"]
        and metadata["source_positive_voxels"] > 1
        and all(np.isfinite(spacing))
        and all(value > 0 for value in spacing)
        and all(value > 0 for value in metadata["bbox_extent"])
        and abs(metadata["normalized_positive_mean"]) <= 1e-6
        and abs(metadata["normalized_positive_population_std"] - 1.0) <= 1e-6
        and metadata["output_shape"] == [1, 128, 128, 128]
        and metadata["output_finite"]
        and metadata["output_mask_voxels"] > 0
    )


def main() -> None:
    from scripts.freeze_rhuh_no_n4_input_bridge import check_implementation

    check_implementation()
    frozen = {
        RECORD / "IDEA.md": IDEA_SHA,
        RECORD / "PREREG.md": PREREG_SHA,
        SOURCE_RECORD / "RHUH_IMAGE_QC_PREACCESS.json": PREACCESS_SHA,
        SOURCE_RECORD / "RHUH_IMAGE_DOWNLOAD.json": DOWNLOAD_SHA,
        SOURCE_RECORD / "RHUH_IMAGE_FORMAT_QC.json": OLD_FORMAT_SHA,
        ROOT / "sammed3d/background_zero_decode.py": DECODER_SHA,
    }
    for path, expected in frozen.items():
        if sha(path) != expected:
            raise ValueError(f"frozen RHUH bridge input changed: {path}")
    preaccess = json.loads((SOURCE_RECORD / "RHUH_IMAGE_QC_PREACCESS.json").read_text())
    download = json.loads((SOURCE_RECORD / "RHUH_IMAGE_DOWNLOAD.json").read_text())
    if preaccess.get("cases") != 80 or download.get("all_official_match") is not True:
        raise ValueError("RHUH80 source identity did not pass")
    if len(download.get("files", [])) != 80:
        raise ValueError("RHUH download receipt is not 80 cases")

    torch.set_num_threads(1)
    start = time.monotonic()
    rows = []
    rebuild_inputs = []
    for index, item in enumerate(download["files"]):
        path = Path(item["path"])
        if sha(path) != item["sha256"]:
            raise ValueError(f"official RHUH image hash changed: {item['case_id']}")
        nii = nib.as_closest_canonical(nib.load(path))
        array = np.ascontiguousarray(np.asanyarray(nii.dataobj), dtype=np.float32)
        spacing = nib.affines.voxel_sizes(nii.affine)
        output, output_mask, metadata = bridge(array)
        row = {
            "case_id": item["case_id"],
            "raw_image_sha256": item["sha256"],
            "canonical_axis_codes": list(nib.aff2axcodes(nii.affine)),
            "canonical_shape": list(array.shape),
            "canonical_spacing": [float(value) for value in spacing],
            **metadata,
        }
        row["passed"] = case_passed(row, spacing)
        rows.append(row)
        if index < 3:
            rebuild_inputs.append((item["case_id"], array, row["output_image_sha256"], row["output_mask_sha256"]))
        print(f"checked {index + 1}/80 {item['case_id']} passed={row['passed']}", flush=True)
        if time.monotonic() - start > 20 * 60:
            raise RuntimeError("20-minute RHUH input-only gate exceeded")

    rebuilds = []
    for case_id, array, expected_image, expected_mask in rebuild_inputs:
        output, output_mask, _ = bridge(array)
        result = {
            "case_id": case_id,
            "image_sha256": array_sha(output),
            "mask_sha256": array_sha(output_mask),
        }
        result["exact"] = (
            result["image_sha256"] == expected_image and result["mask_sha256"] == expected_mask
        )
        rebuilds.append(result)
    all_passed = all(row["passed"] for row in rows) and all(row["exact"] for row in rebuilds)
    report = {
        "schema": "aflow.rhuh_no_n4_input_bridge.qc",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed_input_bridge" if all_passed else "failed_input_bridge",
        "cases": len(rows),
        "subjects": len({row["case_id"].rsplit("_", 1)[0] for row in rows}),
        "rows": rows,
        "independent_rebuilds": rebuilds,
        "all_gates_passed": all_passed,
        "elapsed_seconds": time.monotonic() - start,
        "persistent_cache_created": False,
        "rhuh_labels_downloaded": False,
        "rhuh_label_voxels_accessed": False,
        "model_checkpoint_loaded": False,
        "model_forwards": 0,
        "new_metrics_accessed": False,
        "fixed_validation_accessed": False,
        "outer_accessed": False,
        "confirmation_accessed": False,
        "test270_accessed": False,
        "ucsd75_model_accessed": False,
        "idea_sha256": IDEA_SHA,
        "prereg_sha256": PREREG_SHA,
        "preaccess_sha256": PREACCESS_SHA,
        "download_receipt_sha256": DOWNLOAD_SHA,
        "runner_sha256": sha(Path(__file__)),
        "decoder_sha256": DECODER_SHA,
        "official_preprocessing_commit": "46cf3825278243036c08131b62dcfcfb1d3bba39",
    }
    path = RECORD / "INPUT_QC.json"
    with path.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    if not all_passed:
        raise RuntimeError("RHUH no-N4 input bridge gates failed")


if __name__ == "__main__":
    main()
