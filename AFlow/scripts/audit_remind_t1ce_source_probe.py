"""Run frozen header/geometry/label QC on the downloaded ReMIND-002 probe."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import SimpleITK as sitk

from scripts.download_remind_t1ce_source_probe import RECORD, WORK, sha

DOWNLOAD_SHA = "TO_BE_FROZEN_AFTER_DOWNLOAD"


def image_geometry(image: sitk.Image) -> dict:
    return {
        "size_xyz": [int(value) for value in image.GetSize()],
        "spacing_xyz": [float(value) for value in image.GetSpacing()],
        "origin_xyz": [float(value) for value in image.GetOrigin()],
        "direction": [float(value) for value in image.GetDirection()],
    }


def geometry_delta(left: sitk.Image, right: sitk.Image) -> dict:
    a, b = image_geometry(left), image_geometry(right)
    corners = []
    size = left.GetSize()
    if tuple(size) == tuple(right.GetSize()):
        for x in (0, size[0] - 1):
            for y in (0, size[1] - 1):
                for z in (0, size[2] - 1):
                    p = left.TransformIndexToPhysicalPoint((x, y, z))
                    q = right.TransformIndexToPhysicalPoint((x, y, z))
                    corners.append(max(abs(u - v) for u, v in zip(p, q)))
    return {
        "left": a,
        "right": b,
        "size_exact": a["size_xyz"] == b["size_xyz"],
        "spacing_max_abs": max(abs(u - v) for u, v in zip(a["spacing_xyz"], b["spacing_xyz"])),
        "origin_max_abs": max(abs(u - v) for u, v in zip(a["origin_xyz"], b["origin_xyz"])),
        "direction_max_abs": max(abs(u - v) for u, v in zip(a["direction"], b["direction"])),
        "corner_max_abs_mm": max(corners) if corners else math.inf,
    }


def binary(array: np.ndarray) -> tuple[bool, list[float]]:
    values = sorted(float(value) for value in np.unique(array))
    return set(values) <= {0.0, 1.0} and 1.0 in values, values


def border_mode(array: np.ndarray) -> tuple[float, float]:
    border = np.concatenate((array[0].ravel(), array[-1].ravel(), array[:, 0].ravel(),
                             array[:, -1].ravel(), array[:, :, 0].ravel(), array[:, :, -1].ravel()))
    values, counts = np.unique(border, return_counts=True)
    slot = int(np.argmax(counts))
    return float(values[slot]), float(counts[slot] / counts.sum())


def main() -> None:
    from scripts.freeze_remind_t1ce_source_probe import check_implementation

    check_implementation()
    receipt_path = RECORD / "DOWNLOAD_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("all_download_gates_passed") is not True:
        raise ValueError("ReMIND probe download gates failed")
    if DOWNLOAD_SHA != "TO_BE_FROZEN_AFTER_DOWNLOAD" and sha(receipt_path) != DOWNLOAD_SHA:
        raise ValueError("ReMIND probe download receipt changed")
    dicom_root = Path(receipt["dicom"]["extract_dir"])
    all_files = [path for path in dicom_root.rglob("*") if path.is_file()]
    parents = {path.parent for path in all_files}
    series_uid = json.loads((RECORD / "PROBE_PREACCESS.json").read_text())["probe"]["t1ce"]["SeriesInstanceUID"]
    candidates = []
    for parent in parents:
        names = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(parent), series_uid))
        if names:
            candidates.append(names)
    if len(candidates) != 1 or len(candidates[0]) != 192:
        raise ValueError("ReMIND DICOM extraction did not expose one 192-instance series")
    filenames = candidates[0]
    patient_ids, study_uids, series_uids, sop_uids = set(), set(), set(), set()
    instance_hashes = []
    for filename in filenames:
        reader = sitk.ImageFileReader()
        reader.SetFileName(filename)
        reader.ReadImageInformation()
        patient_ids.add(reader.GetMetaData("0010|0020").strip())
        study_uids.add(reader.GetMetaData("0020|000d").strip())
        series_uids.add(reader.GetMetaData("0020|000e").strip())
        sop_uids.add(reader.GetMetaData("0008|0018").strip())
        instance_hashes.append({"name": Path(filename).name, "sha256": sha(Path(filename))})
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(filenames)
    image = series_reader.Execute()
    transfer = receipt["nrrd_transfer_a"]["files"]
    manual_path = Path(next(row["path"] for row in transfer if "-SEG-tumor-MR-" in row["path"]))
    cerebrum_path = Path(next(row["path"] for row in transfer if "-SEG-cerebrum-MR-" in row["path"]))
    manual = sitk.ReadImage(str(manual_path))
    cerebrum = sitk.ReadImage(str(cerebrum_path))
    image_array = sitk.GetArrayFromImage(image)
    manual_array = sitk.GetArrayFromImage(manual)
    cerebrum_array = sitk.GetArrayFromImage(cerebrum)
    manual_binary, manual_values = binary(manual_array)
    cerebrum_binary, cerebrum_values = binary(cerebrum_array)
    image_manual = geometry_delta(image, manual)
    image_cerebrum = geometry_delta(image, cerebrum)
    mode, mode_fraction = border_mode(image_array)
    nonzero = image_array != 0
    indices = np.nonzero(nonzero)
    bbox = None if any(index.size == 0 for index in indices) else {
        "origin_zyx": [int(index.min()) for index in indices],
        "stop_zyx": [int(index.max()) + 1 for index in indices],
    }
    tumor = manual_array > 0
    brain = cerebrum_array > 0
    outside_nonzero = int(np.count_nonzero(nonzero & ~brain))
    geometry_ok = lambda value: value["size_exact"] and all(value[key] <= 1e-5 for key in (
        "spacing_max_abs", "origin_max_abs", "direction_max_abs", "corner_max_abs_mm"
    ))
    gates = {
        "dicom_instances_192": len(filenames) == 192 and len(sop_uids) == 192,
        "patient_exact": patient_ids == {"ReMIND-002"},
        "study_uid_exact": study_uids == {json.loads((RECORD / "PROBE_PREACCESS.json").read_text())["probe"]["t1ce"]["StudyInstanceUID"]},
        "series_uid_exact": series_uids == {series_uid},
        "image_finite_3d": image_array.ndim == 3 and bool(np.isfinite(image_array).all()),
        "manual_binary_nonempty": manual_binary,
        "cerebrum_binary_nonempty": cerebrum_binary,
        "image_manual_geometry": geometry_ok(image_manual),
        "image_cerebrum_geometry": geometry_ok(image_cerebrum),
        "tumor_inside_cerebrum": bool(np.all(~tumor | brain)),
    }
    report = {
        "schema": "aflow.remind_t1ce_source_probe.qc",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed_source_probe" if all(gates.values()) else "failed_source_probe",
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "dicom": {"files": len(filenames), "patient_ids": sorted(patient_ids), "study_uids": sorted(study_uids),
                  "series_uids": sorted(series_uids), "unique_sop_uids": len(sop_uids),
                  "instance_manifest_sha256": hashlib.sha256(json.dumps(instance_hashes, sort_keys=True).encode()).hexdigest()},
        "image": {"geometry": image_geometry(image), "array_shape_zyx": list(image_array.shape),
                  "dtype": str(image_array.dtype), "min": float(image_array.min()), "max": float(image_array.max()),
                  "negative_voxels": int(np.count_nonzero(image_array < 0)), "zero_voxels": int(np.count_nonzero(image_array == 0)),
                  "positive_voxels": int(np.count_nonzero(image_array > 0)), "nonzero_voxels": int(nonzero.sum()),
                  "border_mode": mode, "border_mode_fraction": mode_fraction, "nonzero_bbox": bbox},
        "manual_tumor": {"path": str(manual_path), "sha256": sha(manual_path), "values": manual_values,
                         "positive_voxels": int(tumor.sum()), "geometry": image_manual},
        "automatic_cerebrum": {"path": str(cerebrum_path), "sha256": sha(cerebrum_path), "values": cerebrum_values,
                               "positive_voxels": int(brain.sum()), "geometry": image_cerebrum,
                               "image_nonzero_outside_mask_voxels": outside_nonzero,
                               "image_nonzero_outside_mask_fraction": outside_nonzero / int(nonzero.sum())},
        "download_receipt_sha256": sha(receipt_path),
        "other_candidate_pixels": False,
        "checkpoint_loaded": False,
        "model_forwards": 0,
        "metrics_accessed": False,
        "runner_sha256": sha(Path(__file__)),
    }
    with (RECORD / "PROBE_QC.json").open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_gates_passed"]:
        raise RuntimeError("ReMIND source probe gates failed")


if __name__ == "__main__":
    main()
