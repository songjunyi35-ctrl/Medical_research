"""Map ReMIND-002 cropped DICOM SEG grids to T1ce physical space once."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import SimpleITK as sitk

from scripts import remind_dicomseg_probe_protocol as base

RECORD = ROOT / "archive/research_records/2026-09-05_remind_physical_grid_probe"
DOWNLOAD = ROOT / "archive/research_records/2026-09-05_remind_dicomseg_source_probe/DOWNLOAD_RECEIPT.json"
DOWNLOAD_SHA = "270ab28554f76f11142f6956efbfcb8cd54c659dbe5318e29c082c335880c4f9"


def original(value) -> str:
    return str(getattr(value, "original_string", value)).strip()


def ds_interval(value) -> tuple[Decimal, Decimal, Decimal, str]:
    text = original(value)
    if not re.fullmatch(r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+))(?:[eE][+-]?\d+)?", text):
        raise ValueError(f"unsupported DICOM DS: {text!r}")
    try:
        center = Decimal(text)
    except InvalidOperation as error:
        raise ValueError("invalid DICOM Decimal") from error
    if not center.is_finite():
        raise ValueError("nonfinite DICOM Decimal")
    half = Decimal(5).scaleb(center.as_tuple().exponent - 1)
    return center - half, center + half, half, text


def contains(interval: tuple[Decimal, Decimal], value) -> bool:
    center = Decimal(original(value))
    return interval[0] <= center <= interval[1]


def intersects(left, right) -> bool:
    a, b = ds_interval(left), ds_interval(right)
    return max(a[0], b[0]) <= min(a[1], b[1])


def gap_interval(first, second) -> tuple[float, float, float, float]:
    centers_a, centers_b, uncertainty = [], [], []
    for left, right in zip(first, second):
        a, b = ds_interval(left), ds_interval(right)
        centers_a.append(float((a[0] + a[1]) / 2))
        centers_b.append(float((b[0] + b[1]) / 2))
        uncertainty.append(float(a[2] + b[2]))
    nominal = float(np.linalg.norm(np.asarray(centers_b) - np.asarray(centers_a)))
    bound = float(np.linalg.norm(uncertainty))
    return max(0.0, nominal - bound), nominal + bound, nominal, bound


def orientation(values) -> tuple[np.ndarray, tuple[str, ...]]:
    strings = tuple(original(value) for value in values)
    vector = np.asarray([float(value) for value in values], dtype=np.float64)
    first, second = vector[:3], vector[3:]
    normal = np.cross(first, second)
    if not np.isfinite(normal).all() or np.linalg.norm(normal) <= 0:
        raise ValueError("invalid DICOM orientation")
    normal /= np.linalg.norm(normal)
    matrix = np.column_stack((first / np.linalg.norm(first), second / np.linalg.norm(second), normal))
    return matrix, strings


def position_key(position, normal: np.ndarray) -> float:
    return float(np.dot(np.asarray([float(value) for value in position]), normal))


def regular_positions(positions: list, declared, normal: np.ndarray) -> tuple[list[int], list[tuple[float, float, float, float]]]:
    order = sorted(range(len(positions)), key=lambda slot: position_key(positions[slot], normal))
    intervals = [gap_interval(positions[left], positions[right]) for left, right in zip(order, order[1:])]
    declared_center = float(Decimal(original(declared)))
    if not intervals or not all(low > 0 and low <= declared_center <= high for low, high, _, _ in intervals):
        raise ValueError("frame positions incompatible with declared spacing interval")
    return order, intervals


def make_image(array: np.ndarray, position, iop, pixel_spacing, z_spacing) -> sitk.Image:
    matrix, _ = orientation(iop)
    image = sitk.GetImageFromArray(np.ascontiguousarray(array))
    image.SetOrigin(tuple(float(value) for value in position))
    image.SetSpacing((float(pixel_spacing[1]), float(pixel_spacing[0]), float(z_spacing)))
    image.SetDirection(tuple(matrix.reshape(-1)))
    return image


def positive_bbox_inside(segmentation: sitk.Image, mask: np.ndarray, reference: sitk.Image) -> tuple[bool, dict]:
    indices = np.nonzero(mask > 0)
    if any(values.size == 0 for values in indices):
        return False, {"positive": 0}
    z_bounds, y_bounds, x_bounds = [(int(values.min()), int(values.max())) for values in indices]
    continuous = []
    for z in z_bounds:
        for y in y_bounds:
            for x in x_bounds:
                point = segmentation.TransformIndexToPhysicalPoint((x, y, z))
                continuous.append(reference.TransformPhysicalPointToContinuousIndex(point))
    size = reference.GetSize()
    inside = all(all(-0.5 <= point[axis] <= size[axis] - 0.5 for axis in range(3)) for point in continuous)
    return inside, {"positive": int((mask > 0).sum()), "bbox_zyx": [list(z_bounds), list(y_bounds), list(x_bounds)],
                    "source_continuous_corner_min": np.min(continuous, axis=0).tolist(),
                    "source_continuous_corner_max": np.max(continuous, axis=0).tolist()}


def main() -> None:
    from scripts.freeze_remind_physical_grid_probe import check_implementation

    check_implementation()
    if base.sha(DOWNLOAD) != DOWNLOAD_SHA:
        raise ValueError("DICOM SEG download receipt changed")
    receipt = json.loads(DOWNLOAD.read_text())
    sys.path.insert(0, str(base.DEPENDENCY))
    import pydicom

    t1_paths = [Path(receipt["t1ce"]["dicom_dir"]) / row["name"] for row in receipt["t1ce"]["dicom_manifest"]]
    sources = [pydicom.dcmread(path) for path in t1_paths]
    source_matrix, source_orientation = orientation(sources[0].ImageOrientationPatient)
    if any(tuple(original(value) for value in item.ImageOrientationPatient) != source_orientation for item in sources):
        raise ValueError("source orientation strings differ")
    source_positions = [item.ImagePositionPatient for item in sources]
    source_order, source_intervals = regular_positions(source_positions, sources[0].SpacingBetweenSlices, source_matrix[:, 2])
    sources = [sources[index] for index in source_order]
    image_array = np.stack([item.pixel_array.astype(np.float32) * float(getattr(item, "RescaleSlope", 1.0))
                            + float(getattr(item, "RescaleIntercept", 0.0)) for item in sources])
    source_image = make_image(image_array, sources[0].ImagePositionPatient, sources[0].ImageOrientationPatient,
                              sources[0].PixelSpacing, sources[0].SpacingBetweenSlices)
    masks, records = {}, {}
    for role in ("tumor", "cerebrum"):
        item = receipt[role + "_a"]["dicom_manifest"][0]
        path = Path(receipt[role + "_a"]["dicom_dir"]) / item["name"]
        dataset = pydicom.dcmread(path)
        if str(dataset.SeriesInstanceUID) != base.ASSETS[role]["series_uid"] or str(dataset.SegmentationType) != "BINARY":
            raise ValueError(f"wrong {role} SEG identity/type")
        shared = dataset.SharedFunctionalGroupsSequence[0]
        iop = shared.PlaneOrientationSequence[0].ImageOrientationPatient
        matrix, orientation_strings = orientation(iop)
        measures = shared.PixelMeasuresSequence[0]
        frames = dataset.PerFrameFunctionalGroupsSequence
        positions = [frame.PlanePositionSequence[0].ImagePositionPatient for frame in frames]
        order, intervals = regular_positions(positions, measures.SliceThickness, matrix[:, 2])
        pixels = np.asarray(dataset.pixel_array)
        if pixels.ndim == 2:
            pixels = pixels[None]
        if pixels.shape != (len(frames), int(dataset.Rows), int(dataset.Columns)) or not set(np.unique(pixels)) <= {0, 1}:
            raise ValueError(f"{role} SEG pixel contract failed")
        pixels = pixels[order].astype(np.uint8)
        seg_image = make_image(pixels, positions[order[0]], iop, measures.PixelSpacing, measures.SliceThickness)
        inside, bbox = positive_bbox_inside(seg_image, pixels, source_image)
        mapped_image = sitk.Resample(seg_image, source_image, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        mapped = sitk.GetArrayFromImage(mapped_image).astype(np.uint8)
        original_volume = float(pixels.sum() * np.prod(seg_image.GetSpacing()))
        mapped_volume = float(mapped.sum() * np.prod(source_image.GetSpacing()))
        relative = abs(mapped_volume - original_volume) / original_volume if original_volume > 0 else math.inf
        masks[role] = mapped
        records[role] = {
            "source_grid": {"frames": len(frames), "rows": int(dataset.Rows), "columns": int(dataset.Columns),
                            "spacing_xyz": list(seg_image.GetSpacing()), "origin_xyz": list(seg_image.GetOrigin()),
                            "direction": list(seg_image.GetDirection())},
            "position_intervals": {"count": len(intervals), "all_contain_declared": True,
                                   "lower_min": min(value[0] for value in intervals),
                                   "upper_max": max(value[1] for value in intervals)},
            "positive_bbox_inside_source": inside,
            "bbox": bbox,
            "original_positive_voxels": int(pixels.sum()),
            "mapped_positive_voxels": int(mapped.sum()),
            "original_physical_volume_mm3": original_volume,
            "mapped_physical_volume_mm3": mapped_volume,
            "physical_volume_relative_error": relative,
            "mapped_mask_sha256": hashlib.sha256(mapped.tobytes()).hexdigest(),
            "segment_label": str(dataset.SegmentSequence[0].SegmentLabel),
        }
    tumor, cerebrum = masks["tumor"].astype(bool), masks["cerebrum"].astype(bool)
    outside = int(np.count_nonzero((image_array != 0) & ~cerebrum))
    gates = {
        "source_intervals": len(source_intervals) == 191,
        "tumor_regular_grid": records["tumor"]["position_intervals"]["all_contain_declared"],
        "cerebrum_regular_grid": records["cerebrum"]["position_intervals"]["all_contain_declared"],
        "tumor_bbox_inside": records["tumor"]["positive_bbox_inside_source"],
        "cerebrum_bbox_inside": records["cerebrum"]["positive_bbox_inside_source"],
        "tumor_nonempty_binary": bool(tumor.any()),
        "cerebrum_nonempty_binary": bool(cerebrum.any()),
        "tumor_volume_within_5pct": records["tumor"]["physical_volume_relative_error"] <= 0.05,
        "cerebrum_volume_within_5pct": records["cerebrum"]["physical_volume_relative_error"] <= 0.05,
        "tumor_inside_cerebrum": bool(np.all(~tumor | cerebrum)),
    }
    report = {
        "schema": "aflow.remind_physical_grid_probe.result",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed_source_probe" if all(gates.values()) else "failed_source_probe",
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "source": {"shape_zyx": list(image_array.shape), "spacing_xyz": list(source_image.GetSpacing()),
                   "origin_xyz": list(source_image.GetOrigin()), "direction": list(source_image.GetDirection()),
                   "finite": bool(np.isfinite(image_array).all()), "min": float(image_array.min()),
                   "max": float(image_array.max()), "nonzero_voxels": int(np.count_nonzero(image_array))},
        "tumor": records["tumor"],
        "cerebrum": {**records["cerebrum"], "image_nonzero_outside_mask_voxels": outside,
                      "image_nonzero_outside_mask_fraction": outside / int(np.count_nonzero(image_array))},
        "access": {"t1ce_pixel_array": True, "seg_pixel_array": True, "old_nrrd_array": False,
                   "other_candidate_pixels": False, "checkpoint": False, "model_forwards": 0, "metrics": False},
        "download_receipt_sha256": DOWNLOAD_SHA,
        "runner_sha256": base.sha(Path(__file__)),
    }
    base.write_new(RECORD / "PROBE_QC.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_gates_passed"]:
        raise RuntimeError("ReMIND physical-grid probe failed")


if __name__ == "__main__":
    main()
