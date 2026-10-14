"""Quantization-aware ReMIND-002 DICOM/SEG geometry and pixel-array probe."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import remind_dicomseg_probe_protocol as base

RECORD = ROOT / "archive/research_records/2026-09-05_remind_quantized_geometry_probe"
DOWNLOAD = ROOT / "archive/research_records/2026-09-05_remind_dicomseg_source_probe/DOWNLOAD_RECEIPT.json"
DOWNLOAD_SHA = "270ab28554f76f11142f6956efbfcb8cd54c659dbe5318e29c082c335880c4f9"
IDEA_SHA = "TO_FREEZE"
PREREG_SHA = "TO_FREEZE"


def original(value) -> str:
    return str(getattr(value, "original_string", value)).strip()


def ds_interval(value) -> tuple[Decimal, Decimal, Decimal, str]:
    text = original(value)
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        raise ValueError(f"unsupported DICOM DS representation: {text!r}")
    center = Decimal(text)
    decimals = len(text.split(".", 1)[1]) if "." in text else 0
    half = Decimal(5).scaleb(-decimals - 1)
    return center - half, center + half, half, text


def intersects(left, right) -> bool:
    lo_a, hi_a, _, _ = ds_interval(left)
    lo_b, hi_b, _, _ = ds_interval(right)
    return max(lo_a, lo_b) <= min(hi_a, hi_b)


def gap_interval(first, second) -> tuple[float, float, float, float]:
    centers_a, centers_b, uncertainty = [], [], []
    for left, right in zip(first, second):
        lo_a, hi_a, half_a, _ = ds_interval(left)
        lo_b, hi_b, half_b, _ = ds_interval(right)
        centers_a.append(float((lo_a + hi_a) / 2))
        centers_b.append(float((lo_b + hi_b) / 2))
        uncertainty.append(float(half_a + half_b))
    nominal = float(np.linalg.norm(np.asarray(centers_b) - np.asarray(centers_a)))
    bound = float(np.linalg.norm(uncertainty))
    return max(0.0, nominal - bound), nominal + bound, nominal, bound


def sort_key(dataset) -> float:
    orientation = np.asarray([float(value) for value in dataset.ImageOrientationPatient])
    normal = np.cross(orientation[:3], orientation[3:])
    return float(np.dot(np.asarray([float(value) for value in dataset.ImagePositionPatient]), normal))


def source_sop(frame) -> str:
    derivation = frame.DerivationImageSequence
    if len(derivation) != 1 or len(derivation[0].SourceImageSequence) != 1:
        raise ValueError("SEG frame must reference one source image")
    return str(derivation[0].SourceImageSequence[0].ReferencedSOPInstanceUID)


def decode_seg(dataset, sources: list, sop_to_slot: dict[str, int]) -> tuple[np.ndarray, dict]:
    if str(dataset.Modality) != "SEG" or str(dataset.SegmentationType) != "BINARY" or int(dataset.BitsAllocated) != 1:
        raise ValueError("unsupported DICOM SEG encoding")
    if len(dataset.SegmentSequence) != 1:
        raise ValueError("expected one segment")
    rows, columns = int(sources[0].Rows), int(sources[0].Columns)
    if int(dataset.Rows) != rows or int(dataset.Columns) != columns:
        raise ValueError("SEG/source rows or columns differ")
    frames = dataset.PerFrameFunctionalGroupsSequence
    pixels = np.asarray(dataset.pixel_array)
    if pixels.ndim == 2:
        pixels = pixels[None]
    if pixels.shape != (len(frames), rows, columns) or not set(np.unique(pixels)) <= {0, 1}:
        raise ValueError("SEG pixel frame/binary contract failed")
    output = np.zeros((len(sources), rows, columns), dtype=np.uint8)
    seen = set()
    position_intersections = []
    for index, frame in enumerate(frames):
        if len(frame.SegmentIdentificationSequence) != 1 or int(frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber) != 1:
            raise ValueError("unexpected segment number")
        sop = source_sop(frame)
        if sop not in sop_to_slot or sop in seen:
            raise ValueError("SEG source SOP is missing/duplicate")
        seen.add(sop)
        positions = frame.PlanePositionSequence
        if len(positions) != 1:
            raise ValueError("SEG frame lacks unique position")
        source = sources[sop_to_slot[sop]]
        axis_ok = [intersects(left, right) for left, right in zip(
            positions[0].ImagePositionPatient, source.ImagePositionPatient
        )]
        position_intersections.append(all(axis_ok))
        output[sop_to_slot[sop]] = pixels[index].astype(np.uint8)
    measures = dataset.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence
    if len(measures) != 1:
        raise ValueError("SEG lacks shared pixel measures")
    spacing_intersections = [intersects(left, right) for left, right in zip(
        measures[0].PixelSpacing, sources[0].PixelSpacing
    )]
    return output, {
        "frames": len(frames),
        "referenced_sops": len(seen),
        "unreferenced_zero_slices": len(sources) - len(seen),
        "all_position_intervals_intersect": all(position_intersections),
        "all_pixel_spacing_intervals_intersect": all(spacing_intersections),
        "segment_label": str(dataset.SegmentSequence[0].SegmentLabel),
        "positive_voxels": int(output.sum()),
        "mask_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
        "transfer_syntax": str(dataset.file_meta.TransferSyntaxUID),
    }


def main() -> None:
    from scripts.freeze_remind_quantized_geometry_probe import check_implementation

    check_implementation()
    if base.sha(DOWNLOAD) != DOWNLOAD_SHA:
        raise ValueError("frozen DICOM SEG download receipt changed")
    receipt = json.loads(DOWNLOAD.read_text())
    if receipt.get("all_download_gates_passed") is not True:
        raise ValueError("DICOM SEG byte gates failed")
    sys.path.insert(0, str(base.DEPENDENCY))
    import pydicom

    t1_paths = [Path(receipt["t1ce"]["dicom_dir"]) / row["name"] for row in receipt["t1ce"]["dicom_manifest"]]
    sources = [pydicom.dcmread(path) for path in t1_paths]
    orientation_strings = [tuple(original(value) for value in dataset.ImageOrientationPatient) for dataset in sources]
    if len(set(orientation_strings)) != 1:
        raise ValueError("source orientation strings differ")
    sources.sort(key=sort_key)
    projections = [sort_key(dataset) for dataset in sources]
    if not all(right > left for left, right in zip(projections, projections[1:])):
        raise ValueError("source positions are not strictly monotonic")
    spacing_strings = [original(dataset.SpacingBetweenSlices) for dataset in sources]
    if set(spacing_strings) != {"1"}:
        raise ValueError("source declared spacing is not exact string1")
    intervals = [gap_interval(left.ImagePositionPatient, right.ImagePositionPatient)
                 for left, right in zip(sources, sources[1:])]
    gap_gates = [low > 0 and low <= 1.0 <= high for low, high, _, _ in intervals]
    if not all(gap_gates):
        raise ValueError("source gap quantization intervals do not contain declared1mm")
    sop_to_slot = {str(dataset.SOPInstanceUID): slot for slot, dataset in enumerate(sources)}
    if len(sop_to_slot) != 192 or {int(dataset.InstanceNumber) for dataset in sources} != set(range(1, 193)):
        raise ValueError("source SOP/Instance identity failed")
    # Pixel arrays are first touched only after every source interval gate above passes.
    image = np.stack([
        dataset.pixel_array.astype(np.float32) * float(getattr(dataset, "RescaleSlope", 1.0))
        + float(getattr(dataset, "RescaleIntercept", 0.0))
        for dataset in sources
    ])
    masks, metadata = {}, {}
    for role in ("tumor", "cerebrum"):
        item = receipt[role + "_a"]["dicom_manifest"][0]
        path = Path(receipt[role + "_a"]["dicom_dir"]) / item["name"]
        dataset = pydicom.dcmread(path)
        if str(dataset.PatientID) != "ReMIND-002" or str(dataset.SeriesInstanceUID) != base.ASSETS[role]["series_uid"]:
            raise ValueError(f"wrong {role} DICOM SEG identity")
        mask, info = decode_seg(dataset, sources, sop_to_slot)
        masks[role], metadata[role] = mask, info
    tumor, cerebrum = masks["tumor"].astype(bool), masks["cerebrum"].astype(bool)
    nonzero = image != 0
    outside = int(np.count_nonzero(nonzero & ~cerebrum))
    gates = {
        "source_gap_intervals_191": len(intervals) == 191 and all(gap_gates),
        "source_image_finite": image.shape[0] == 192 and bool(np.isfinite(image).all()),
        "tumor_position_intervals": metadata["tumor"]["all_position_intervals_intersect"],
        "cerebrum_position_intervals": metadata["cerebrum"]["all_position_intervals_intersect"],
        "pixel_spacing_intervals": metadata["tumor"]["all_pixel_spacing_intervals_intersect"]
                                   and metadata["cerebrum"]["all_pixel_spacing_intervals_intersect"],
        "tumor_nonempty": metadata["tumor"]["positive_voxels"] > 0,
        "cerebrum_nonempty": metadata["cerebrum"]["positive_voxels"] > 0,
        "tumor_inside_cerebrum": bool(np.all(~tumor | cerebrum)),
    }
    report = {
        "schema": "aflow.remind_quantized_geometry_probe.result",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed_source_probe" if all(gates.values()) else "failed_source_probe",
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "source_intervals": {
            "count": len(intervals),
            "lower_min": min(row[0] for row in intervals), "lower_max": max(row[0] for row in intervals),
            "upper_min": min(row[1] for row in intervals), "upper_max": max(row[1] for row in intervals),
            "nominal_min": min(row[2] for row in intervals), "nominal_max": max(row[2] for row in intervals),
            "uncertainty_bound_min": min(row[3] for row in intervals),
            "uncertainty_bound_max": max(row[3] for row in intervals),
            "all_contain_declared_1mm": all(gap_gates),
        },
        "image": {"shape_zyx": list(image.shape), "dtype": str(image.dtype),
                  "min": float(image.min()), "max": float(image.max()),
                  "negative_voxels": int(np.count_nonzero(image < 0)),
                  "zero_voxels": int(np.count_nonzero(image == 0)),
                  "positive_voxels": int(np.count_nonzero(image > 0)),
                  "nonzero_voxels": int(nonzero.sum())},
        "tumor": metadata["tumor"],
        "cerebrum": {**metadata["cerebrum"],
                      "image_nonzero_outside_mask_voxels": outside,
                      "image_nonzero_outside_mask_fraction": outside / int(nonzero.sum())},
        "access": {"t1ce_pixel_array": True, "dicomseg_pixel_array": True,
                   "old_nrrd_array": False, "other_candidate_pixels": False,
                   "checkpoint": False, "model_forwards": 0, "metrics": False},
        "download_receipt_sha256": DOWNLOAD_SHA,
        "dependency_manifest_sha256": base.DEPENDENCY_MANIFEST_SHA,
        "runner_sha256": base.sha(Path(__file__)),
    }
    base.write_new(RECORD / "PROBE_QC.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_gates_passed"]:
        raise RuntimeError("quantized geometry ReMIND probe failed")


if __name__ == "__main__":
    main()
