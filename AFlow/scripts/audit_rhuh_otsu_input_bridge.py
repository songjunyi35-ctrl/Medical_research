"""RHUH input-only Otsu-N4 bridge QC; no segmentation or checkpoint access."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from sammed3d.background_zero_decode import decode_background_zero
from sammed3d.n4_otsu_input import n4_otsu_correct

RECORD = ROOT / "archive/research_records/2026-09-05_rhuh_otsu_input_bridge"
SOURCE_RECORD = ROOT / "archive/reviews/2026-09-05_30h_chain_audit"
IDEA_SHA = "517c07b2d5046993d53ea501e79d982b104898880e60b0e011044d5b5f9d1906"
PREREG_SHA = "28874d2c2f5795fbd4bec1994a13dafb0c8c94616120c3d83ce868920fa25673"
PREACCESS_SHA = "65892e4d90a577ff2623cd2786d5142aeed86be809a5b273faf245321a4481c9"
DOWNLOAD_SHA = "c6f17decbc81e8a2bc8ec385438b7e83547644aed26009fb376bd27a218d725f"
MECHANISM_SHA = "2b0f6e4b2b2b008d2669b79423203a2f348fee981bc3e50176721d9047d720da"
DECODER_SHA = "de562395a0d733bc8d9c47bbd0bab6d9bf260af5d2c38913df7250a90397b476"
N4_OTSU_SHA = "61e36493565944fbd73289c1b9921f704640b4c8c58793ef9cbdabcaf451e466"
MODEL_SHAPE = (128, 128, 128)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def positive_bbox(array: np.ndarray) -> tuple[list[int], list[int]]:
    indices = np.nonzero(array > 0)
    if any(index.size == 0 for index in indices):
        raise ValueError("empty Otsu-N4 support")
    return [int(index.min()) for index in indices], [int(index.max()) + 1 for index in indices]


def otsu_bridge(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    decoded, background = decode_background_zero(array)
    corrected, fit = n4_otsu_correct(decoded)
    support = corrected > 0
    if not np.array_equal(support, decoded > 0):
        raise ValueError("Otsu-N4 changed full support")
    origin, stop = positive_bbox(corrected)
    values = corrected[support].astype(np.float64)
    mean, std = float(values.mean()), float(values.std())
    if not np.isfinite(std) or std <= 1e-8:
        raise ValueError("invalid Otsu-N4 foreground variance")
    normalized = np.zeros(corrected.shape, dtype=np.float32)
    normalized[support] = ((corrected[support].astype(np.float64) - mean) / std).astype(np.float32)
    normalized_values = normalized[support].astype(np.float64)
    slices = tuple(slice(low, high) for low, high in zip(origin, stop))
    image_crop = np.ascontiguousarray(normalized[slices])
    mask_crop = np.ascontiguousarray(support[slices])
    image_tensor = torch.from_numpy(image_crop).reshape(1, 1, *image_crop.shape)
    mask_tensor = torch.from_numpy(mask_crop).reshape(1, 1, *mask_crop.shape)
    image = F.interpolate(image_tensor, size=MODEL_SHAPE, mode="trilinear", align_corners=False)[0, 0]
    mask = F.interpolate(mask_tensor.float(), size=MODEL_SHAPE, mode="nearest-exact")[0, 0].bool()
    image.masked_fill_(~mask, 0.0)
    output = np.ascontiguousarray(image.numpy(), dtype=np.float32)
    output_mask = np.ascontiguousarray(mask.numpy(), dtype=np.uint8)
    metadata = {
        "background": background,
        "fit_mask": fit,
        "decoded_background_exact_zero": bool(np.all(decoded[decoded <= 0] == 0)),
        "support_exact_after_n4": bool(np.array_equal(corrected > 0, decoded > 0)),
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


def passed(metadata: dict, spacing: np.ndarray) -> bool:
    return bool(
        metadata["background"]["border_mode_fraction"] >= 0.95
        and metadata["decoded_background_exact_zero"]
        and metadata["support_exact_after_n4"]
        and metadata["fit_mask"]["fit_voxels"] > 0
        and metadata["fit_mask"]["fit_voxels"] <= metadata["source_positive_voxels"]
        and metadata["fit_mask"]["mask_used_only_for_fitting"] is True
        and all(np.isfinite(spacing))
        and all(value > 0 for value in spacing)
        and all(value > 0 for value in metadata["bbox_extent"])
        and abs(metadata["normalized_positive_mean"]) <= 1e-6
        and abs(metadata["normalized_positive_population_std"] - 1.0) <= 1e-6
        and metadata["output_shape"] == [1, 128, 128, 128]
        and metadata["output_finite"]
        and metadata["output_mask_voxels"] > 0
    )


def initialize_worker() -> None:
    torch.set_num_threads(1)
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)


def worker(item: dict) -> dict:
    path = Path(item["path"])
    if sha(path) != item["sha256"]:
        raise ValueError(f"official RHUH image hash changed: {item['case_id']}")
    nii = nib.as_closest_canonical(nib.load(path))
    array = np.ascontiguousarray(np.asanyarray(nii.dataobj), dtype=np.float32)
    spacing = nib.affines.voxel_sizes(nii.affine)
    _, _, metadata = otsu_bridge(array)
    row = {
        "case_id": item["case_id"],
        "raw_image_sha256": item["sha256"],
        "canonical_axis_codes": list(nib.aff2axcodes(nii.affine)),
        "canonical_shape": list(array.shape),
        "canonical_spacing": [float(value) for value in spacing],
        **metadata,
    }
    row["passed"] = passed(row, spacing)
    return row


def main(workers: int) -> None:
    from scripts.freeze_rhuh_otsu_input_bridge import check_implementation

    check_implementation()
    frozen = {
        RECORD / "IDEA.md": IDEA_SHA,
        RECORD / "PREREG.md": PREREG_SHA,
        SOURCE_RECORD / "RHUH_IMAGE_QC_PREACCESS.json": PREACCESS_SHA,
        SOURCE_RECORD / "RHUH_IMAGE_DOWNLOAD.json": DOWNLOAD_SHA,
        ROOT / "archive/research_records/2026-09-05_n4_otsu_clean_replication/MECHANISM_GATE.json": MECHANISM_SHA,
        ROOT / "sammed3d/background_zero_decode.py": DECODER_SHA,
        ROOT / "sammed3d/n4_otsu_input.py": N4_OTSU_SHA,
    }
    for path, expected in frozen.items():
        if sha(path) != expected:
            raise ValueError(f"frozen RHUH Otsu bridge input changed: {path}")
    download = json.loads((SOURCE_RECORD / "RHUH_IMAGE_DOWNLOAD.json").read_text())
    if download.get("all_official_match") is not True or len(download.get("files", [])) != 80:
        raise ValueError("RHUH80 source identity failed")
    if workers < 1 or workers > 8:
        raise ValueError("workers must be 1..8")
    start = time.monotonic()
    with mp.get_context("spawn").Pool(workers, initializer=initialize_worker) as pool:
        rows = list(pool.imap(worker, download["files"], chunksize=1))
    for index, row in enumerate(rows):
        print(f"checked Otsu {index + 1}/80 {row['case_id']} passed={row['passed']}", flush=True)
    rebuilds = []
    for item, original in zip(download["files"][:3], rows[:3]):
        rebuilt = worker(item)
        record = {
            "case_id": item["case_id"],
            "image_sha256": rebuilt["output_image_sha256"],
            "mask_sha256": rebuilt["output_mask_sha256"],
            "fit_mask_sha256": rebuilt["fit_mask"]["fit_mask_sha256"],
        }
        record["exact"] = (
            record["image_sha256"] == original["output_image_sha256"]
            and record["mask_sha256"] == original["output_mask_sha256"]
            and record["fit_mask_sha256"] == original["fit_mask"]["fit_mask_sha256"]
        )
        rebuilds.append(record)
    elapsed = time.monotonic() - start
    all_passed = all(row["passed"] for row in rows) and all(row["exact"] for row in rebuilds) and elapsed <= 20 * 60
    report = {
        "schema": "aflow.rhuh_otsu_input_bridge.qc",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed_input_bridge" if all_passed else "failed_input_bridge",
        "cases": len(rows),
        "subjects": len({row["case_id"].rsplit("_", 1)[0] for row in rows}),
        "rows": rows,
        "independent_rebuilds": rebuilds,
        "workers": workers,
        "elapsed_seconds": elapsed,
        "all_gates_passed": all_passed,
        "persistent_cache_created": False,
        "rhuh_labels_downloaded": False,
        "rhuh_label_voxels_accessed": False,
        "model_checkpoint_loaded": False,
        "model_forwards": 0,
        "new_metrics_accessed": False,
        "old_cohort_accessed": False,
        "idea_sha256": IDEA_SHA,
        "prereg_sha256": PREREG_SHA,
        "preaccess_sha256": PREACCESS_SHA,
        "download_receipt_sha256": DOWNLOAD_SHA,
        "mechanism_sha256": MECHANISM_SHA,
        "runner_sha256": sha(Path(__file__)),
        "decoder_sha256": DECODER_SHA,
        "n4_otsu_sha256": N4_OTSU_SHA,
    }
    with (RECORD / "INPUT_QC.json").open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    if not all_passed:
        raise RuntimeError("RHUH Otsu input bridge gates failed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    main(parser.parse_args().workers)
