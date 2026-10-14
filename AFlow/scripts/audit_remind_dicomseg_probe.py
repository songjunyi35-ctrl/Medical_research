"""Decode frozen ReMIND-002 DICOM SEG frames by referenced source SOP UID."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from scripts import remind_dicomseg_probe_protocol as p


def source_sort_key(dataset) -> float:
    orientation = np.asarray([float(value) for value in dataset.ImageOrientationPatient], dtype=np.float64)
    row, column = orientation[:3], orientation[3:]
    normal = np.cross(row, column)
    return float(np.dot(np.asarray(dataset.ImagePositionPatient, dtype=np.float64), normal))


def source_sop(frame) -> str:
    derivation = frame.DerivationImageSequence
    if len(derivation) != 1 or len(derivation[0].SourceImageSequence) != 1:
        raise ValueError("SEG frame must reference exactly one source image")
    return str(derivation[0].SourceImageSequence[0].ReferencedSOPInstanceUID)


def frame_position(frame) -> np.ndarray:
    if len(frame.PlanePositionSequence) != 1:
        raise ValueError("SEG frame must have one PlanePositionSequence")
    return np.asarray([float(value) for value in frame.PlanePositionSequence[0].ImagePositionPatient], dtype=np.float64)


def decode_seg(dataset, sop_to_slot: dict[str, int], source_positions: dict[str, np.ndarray], rows: int, columns: int) -> tuple[np.ndarray, dict]:
    if str(dataset.Modality) != "SEG" or str(dataset.SegmentationType) != "BINARY" or int(dataset.BitsAllocated) != 1:
        raise ValueError("unsupported non-binary DICOM SEG")
    if len(dataset.SegmentSequence) != 1:
        raise ValueError("expected exactly one segment")
    pixels = np.asarray(dataset.pixel_array)
    if pixels.ndim == 2:
        pixels = pixels[None]
    frames = dataset.PerFrameFunctionalGroupsSequence
    if pixels.shape != (len(frames), rows, columns):
        raise ValueError("DICOM SEG pixel/frame shape mismatch")
    if not set(np.unique(pixels)) <= {0, 1}:
        raise ValueError("DICOM SEG values are not binary")
    output = np.zeros((len(sop_to_slot), rows, columns), dtype=np.uint8)
    seen = set()
    position_errors = []
    for slot, frame in enumerate(frames):
        if len(frame.SegmentIdentificationSequence) != 1 or int(frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber) != 1:
            raise ValueError("unexpected DICOM SEG segment number")
        sop = source_sop(frame)
        if sop not in sop_to_slot or sop in seen:
            raise ValueError("missing/duplicate/non-source SOP reference")
        seen.add(sop)
        position_errors.append(float(np.max(np.abs(frame_position(frame) - source_positions[sop]))))
        output[sop_to_slot[sop]] = pixels[slot].astype(np.uint8)
    return output, {
        "frames": len(frames),
        "referenced_source_sops": len(seen),
        "unreferenced_zero_slices": len(sop_to_slot) - len(seen),
        "position_max_abs_mm": max(position_errors) if position_errors else math.inf,
        "segment_label": str(dataset.SegmentSequence[0].SegmentLabel),
        "mask_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
        "positive_voxels": int(output.sum()),
    }


def main() -> None:
    from scripts.freeze_remind_dicomseg_probe import check_implementation

    check_implementation()
    receipt_path = p.RECORD / "DOWNLOAD_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("all_download_gates_passed") is not True:
        raise ValueError("ReMIND DICOM SEG downloads did not pass")
    sys.path.insert(0, str(p.DEPENDENCY))
    import pydicom

    if pydicom.__version__ != "3.0.1" or not str(pydicom.__file__).startswith(str(p.DEPENDENCY)):
        raise ValueError("wrong pydicom dependency")
    t1_paths = [Path(receipt["t1ce"]["dicom_dir"]) / row["name"] for row in receipt["t1ce"]["dicom_manifest"]]
    sources = [pydicom.dcmread(path) for path in t1_paths]
    sources.sort(key=source_sort_key)
    positions = [source_sort_key(dataset) for dataset in sources]
    gaps = np.diff(positions)
    if not np.all(gaps > 0) or float(np.max(np.abs(gaps - np.median(gaps)))) > 1e-5:
        raise ValueError("source slice positions are not regular/strict")
    sop_to_slot = {str(dataset.SOPInstanceUID): slot for slot, dataset in enumerate(sources)}
    if len(sop_to_slot) != 192:
        raise ValueError("source SOP UIDs not unique")
    source_positions = {str(dataset.SOPInstanceUID): np.asarray(dataset.ImagePositionPatient, dtype=np.float64) for dataset in sources}
    patient_ids = {str(dataset.PatientID) for dataset in sources}
    study_uids = {str(dataset.StudyInstanceUID) for dataset in sources}
    series_uids = {str(dataset.SeriesInstanceUID) for dataset in sources}
    rows, columns = int(sources[0].Rows), int(sources[0].Columns)
    pixel_spacing = np.asarray(sources[0].PixelSpacing, dtype=np.float64)
    image_slices = []
    for dataset in sources:
        pixel = dataset.pixel_array.astype(np.float32)
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        image_slices.append(pixel * slope + intercept)
    image = np.stack(image_slices)
    decoded = {}
    masks = {}
    for role in ("tumor", "cerebrum"):
        path = Path(receipt[role + "_a"]["dicom_dir"]) / receipt[role + "_a"]["dicom_manifest"][0]["name"]
        dataset = pydicom.dcmread(path)
        if str(dataset.PatientID) != "ReMIND-002" or str(dataset.StudyInstanceUID) != p.ASSETS[role]["study_uid"]:
            raise ValueError(f"wrong {role} patient/study")
        if str(dataset.SeriesInstanceUID) != p.ASSETS[role]["series_uid"] or str(dataset.SeriesDescription) != p.ASSETS[role]["description"]:
            raise ValueError(f"wrong {role} series identity")
        mask, metadata = decode_seg(dataset, sop_to_slot, source_positions, rows, columns)
        shared = dataset.SharedFunctionalGroupsSequence[0]
        measures = shared.PixelMeasuresSequence[0]
        metadata["pixel_spacing_max_abs"] = float(np.max(np.abs(np.asarray(measures.PixelSpacing, dtype=np.float64) - pixel_spacing)))
        metadata["slice_thickness"] = float(measures.SliceThickness)
        metadata["transfer_syntax"] = str(dataset.file_meta.TransferSyntaxUID)
        metadata["series_description"] = str(dataset.SeriesDescription)
        metadata["segmentation_type"] = str(dataset.SegmentationType)
        decoded[role] = metadata
        masks[role] = mask
    tumor, cerebrum = masks["tumor"].astype(bool), masks["cerebrum"].astype(bool)
    nonzero = image != 0
    outside = int(np.count_nonzero(nonzero & ~cerebrum))
    gates = {
        "source_identity": patient_ids == {"ReMIND-002"} and study_uids == {p.ASSETS["t1ce"]["study_uid"]}
                           and series_uids == {p.ASSETS["t1ce"]["series_uid"]},
        "source_shape_finite": image.shape == (192, rows, columns) and bool(np.isfinite(image).all()),
        "tumor_binary_nonempty": decoded["tumor"]["positive_voxels"] > 0,
        "cerebrum_binary_nonempty": decoded["cerebrum"]["positive_voxels"] > 0,
        "all_frame_positions_exact": decoded["tumor"]["position_max_abs_mm"] <= 1e-5
                                     and decoded["cerebrum"]["position_max_abs_mm"] <= 1e-5,
        "pixel_spacing_exact": decoded["tumor"]["pixel_spacing_max_abs"] <= 1e-5
                               and decoded["cerebrum"]["pixel_spacing_max_abs"] <= 1e-5,
        "tumor_inside_cerebrum": bool(np.all(~tumor | cerebrum)),
    }
    report = {
        "schema": "aflow.remind_dicomseg_source_probe.qc",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed_source_probe" if all(gates.values()) else "failed_source_probe",
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "source": {"shape_zyx": list(image.shape), "rows": rows, "columns": columns,
                   "slice_gap": float(np.median(gaps)), "pixel_spacing": pixel_spacing.tolist(),
                   "dtype": str(image.dtype), "min": float(image.min()), "max": float(image.max()),
                   "negative_voxels": int(np.count_nonzero(image < 0)), "zero_voxels": int(np.count_nonzero(image == 0)),
                   "positive_voxels": int(np.count_nonzero(image > 0)), "nonzero_voxels": int(nonzero.sum())},
        "tumor": decoded["tumor"],
        "cerebrum": {**decoded["cerebrum"], "image_nonzero_outside_mask_voxels": outside,
                      "image_nonzero_outside_mask_fraction": outside / int(nonzero.sum())},
        "dependency": {"pydicom": pydicom.__version__, "wheel_sha256": p.WHEEL_SHA,
                       "manifest_sha256": p.DEPENDENCY_MANIFEST_SHA},
        "download_receipt_sha256": p.sha(receipt_path),
        "old_nrrd_arrays_opened": False,
        "other_candidate_pixels": False,
        "checkpoint_loaded": False,
        "model_forwards": 0,
        "metrics": False,
        "runner_sha256": p.sha(Path(__file__)),
    }
    p.write_new(p.RECORD / "PROBE_QC.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_gates_passed"]:
        raise RuntimeError("ReMIND DICOM SEG source-probe gates failed")


if __name__ == "__main__":
    main()
