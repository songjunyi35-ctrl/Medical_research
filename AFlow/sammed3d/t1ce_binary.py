"""T1ce-only binary data and metric contract for the DPD-SAM track.

The foreground is the BraTS post-treatment residual-tumor union of raw labels
1/2/3.  Label 4 (resection cavity) is deliberately background.  All spatial
preprocessing is whole-volume resizing; this module has no crop or patch API.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from surface_distance import metrics as surface_metrics

from sammed3d.brats_data import assert_no_supervision_fields


T1CE_MODALITY_KEY = "t1c"
RESIDUAL_TUMOR_LABELS = frozenset({1, 2, 3})
ALLOWED_RAW_LABELS = frozenset({0, 1, 2, 3, 4})
DEFAULT_MODEL_SHAPE = (128, 128, 128)
CASE_ID_RE = re.compile(r"^BraTS-GLI-\d{5}-\d{3}$")

SpatialShape = tuple[int, int, int]
Spacing = tuple[float, float, float]


class T1ceBinaryDataError(ValueError):
    pass


class T1ceBinaryGeometryError(T1ceBinaryDataError):
    pass


def _path(value: object, field: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise T1ceBinaryDataError(f"{field} must be a non-empty path")
    return Path(value)


def _case_id(value: object) -> str:
    if not isinstance(value, str) or CASE_ID_RE.fullmatch(value) is None:
        raise T1ceBinaryDataError(f"invalid BraTS2024 case_id: {value!r}")
    return value


def _t1ce_path_from_row(row: Mapping[str, object]) -> Path:
    if row.get("schema") == "aflow.brats2024.dpdsam_t1ce_binary_split":
        if row.get("modality") != "t1ce":
            raise T1ceBinaryDataError("DPD split row must declare modality=t1ce")
        return _path(row.get("image_path"), "image_path")
    modalities = row.get("modalities")
    if not isinstance(modalities, Mapping):
        raise T1ceBinaryDataError("manifest row must contain a modalities mapping")
    if T1CE_MODALITY_KEY not in modalities:
        raise T1ceBinaryDataError("manifest row is missing modalities.t1c")
    # Do not validate, resolve, stat, or load any other modality entry.
    return _path(modalities[T1CE_MODALITY_KEY], "modalities.t1c")


@dataclass(frozen=True)
class T1ceInferenceCase:
    case_id: str
    t1ce_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _case_id(self.case_id))
        object.__setattr__(self, "t1ce_path", _path(self.t1ce_path, "t1ce_path"))

    @classmethod
    def from_manifest_row(cls, row: Mapping[str, object]) -> "T1ceInferenceCase":
        try:
            assert_no_supervision_fields(row, "T1ce inference manifest row")
        except ValueError as exc:
            raise T1ceBinaryDataError(str(exc)) from exc
        return cls(_case_id(row.get("case_id")), _t1ce_path_from_row(row))


@dataclass(frozen=True)
class T1ceSupervisedCase:
    case_id: str
    t1ce_path: Path
    seg_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _case_id(self.case_id))
        object.__setattr__(self, "t1ce_path", _path(self.t1ce_path, "t1ce_path"))
        object.__setattr__(self, "seg_path", _path(self.seg_path, "seg_path"))

    @classmethod
    def from_manifest_row(cls, row: Mapping[str, object]) -> "T1ceSupervisedCase":
        seg_path = row.get("seg_path", row.get("label_path"))
        if seg_path is None:
            raise T1ceBinaryDataError("supervised manifest row is missing segmentation path")
        return cls(
            _case_id(row.get("case_id")),
            _t1ce_path_from_row(row),
            _path(seg_path, "segmentation path"),
        )


def _shape(value: Sequence[int], field: str) -> SpatialShape:
    resolved = tuple(int(item) for item in value)
    if len(resolved) != 3 or any(item <= 0 for item in resolved):
        raise T1ceBinaryDataError(f"{field} must contain three positive integers")
    return resolved  # type: ignore[return-value]


def _spacing(value: Sequence[float], field: str = "spacing_mm") -> Spacing:
    resolved = tuple(float(item) for item in value)
    if len(resolved) != 3 or not all(
        np.isfinite(item) and item > 0.0 for item in resolved
    ):
        raise T1ceBinaryDataError(f"{field} must contain three finite positive values")
    return resolved  # type: ignore[return-value]


@dataclass(frozen=True)
class T1ceNormalization:
    nonzero_voxels: int
    mean: float
    std: float
    convention: str = "source_t1ce_nonzero_population_zscore"


@dataclass(frozen=True)
class WholeVolumeResizeGeometry:
    original_shape: SpatialShape
    model_shape: SpatialShape
    original_spacing_mm: Spacing
    model_spacing_mm: Spacing
    convention: str = "spacing_out=spacing_in*shape_in/shape_out"


@dataclass(frozen=True)
class T1ceInferenceView:
    case_id: str
    image: torch.Tensor
    brain_mask: torch.Tensor
    normalization: T1ceNormalization
    geometry: WholeVolumeResizeGeometry
    modality: str = T1CE_MODALITY_KEY


@dataclass(frozen=True)
class T1ceSupervisedView:
    case_id: str
    image: torch.Tensor
    target: torch.Tensor
    source_target: torch.Tensor
    brain_mask: torch.Tensor
    normalization: T1ceNormalization
    geometry: WholeVolumeResizeGeometry
    modality: str = T1CE_MODALITY_KEY
    foreground_raw_labels: tuple[int, int, int] = (1, 2, 3)


def model_spacing_mm(
    original_shape: Sequence[int],
    original_spacing_mm: Sequence[float],
    model_shape: Sequence[int] = DEFAULT_MODEL_SHAPE,
) -> Spacing:
    """Preserve the physical field of view when changing the voxel grid."""
    source_shape = _shape(original_shape, "original_shape")
    target_shape = _shape(model_shape, "model_shape")
    spacing = _spacing(original_spacing_mm)
    return tuple(
        spacing[axis] * source_shape[axis] / target_shape[axis]
        for axis in range(3)
    )  # type: ignore[return-value]


def residual_tumor_target(segmentation: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert raw BraTS labels to {1,2,3} foreground, excluding RC label 4."""
    values = torch.as_tensor(segmentation)
    if values.ndim != 3:
        raise T1ceBinaryDataError(
            f"segmentation must be 3D, got shape {tuple(values.shape)}"
        )
    if values.is_floating_point():
        if not torch.isfinite(values).all():
            raise T1ceBinaryDataError("segmentation contains non-finite values")
        rounded = values.round()
        if not torch.allclose(values, rounded, rtol=0.0, atol=1e-5):
            raise T1ceBinaryDataError("segmentation contains non-integer values")
        values = rounded
    else:
        # PyTorch does not implement every CPU set operation for UInt16, a
        # dtype used by some BraTS NIfTI segmentations.
        values = values.to(torch.int64)
    labels = {int(item) for item in torch.unique(values).cpu().tolist()}
    if not labels <= ALLOWED_RAW_LABELS:
        raise T1ceBinaryDataError(
            f"segmentation contains unexpected raw labels: {sorted(labels)}"
        )
    return (values == 1) | (values == 2) | (values == 3)


def _normalize_nonzero_t1ce(
    image: np.ndarray | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, T1ceNormalization]:
    tensor = torch.as_tensor(image, dtype=torch.float32)
    if tensor.ndim != 3:
        raise T1ceBinaryDataError(f"T1ce image must be 3D, got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise T1ceBinaryDataError("T1ce image contains non-finite values")
    brain_mask = tensor != 0
    count = int(brain_mask.sum())
    if count < 2:
        raise T1ceBinaryDataError("T1ce image has fewer than two nonzero brain voxels")
    selected = tensor[brain_mask].double()
    mean = float(selected.mean())
    std = float(selected.std(unbiased=False))
    if not np.isfinite(std) or std <= 1e-8:
        raise T1ceBinaryDataError(f"T1ce nonzero standard deviation is invalid: {std}")
    normalized = torch.zeros_like(tensor)
    normalized[brain_mask] = (tensor[brain_mask] - mean) / std
    return normalized, brain_mask, T1ceNormalization(count, mean, std)


def _resize(volume: torch.Tensor, output_shape: SpatialShape, mode: str) -> torch.Tensor:
    batch = volume.float().reshape(1, 1, *volume.shape)
    if mode == "trilinear":
        resized = F.interpolate(batch, size=output_shape, mode=mode, align_corners=False)
    elif mode == "nearest-exact":
        resized = F.interpolate(batch, size=output_shape, mode=mode)
    else:
        raise ValueError("mode must be trilinear or nearest-exact")
    return resized[0, 0]


def prepare_t1ce_inference_array(
    case_id: str,
    t1ce: np.ndarray | torch.Tensor,
    spacing_mm: Sequence[float],
    *,
    output_shape: Sequence[int] = DEFAULT_MODEL_SHAPE,
) -> T1ceInferenceView:
    """Normalize and resize a complete T1ce volume without spatial cropping."""
    case_id = _case_id(case_id)
    target_shape = _shape(output_shape, "output_shape")
    normalized, brain_mask, stats = _normalize_nonzero_t1ce(t1ce)
    source_shape = _shape(normalized.shape, "T1ce shape")
    resized_mask = _resize(brain_mask, target_shape, "nearest-exact").bool()
    resized_image = _resize(normalized, target_shape, "trilinear")
    resized_image.masked_fill_(~resized_mask, 0.0)
    geometry = WholeVolumeResizeGeometry(
        original_shape=source_shape,
        model_shape=target_shape,
        original_spacing_mm=_spacing(spacing_mm),
        model_spacing_mm=model_spacing_mm(source_shape, spacing_mm, target_shape),
    )
    return T1ceInferenceView(
        case_id=case_id,
        image=resized_image.unsqueeze(0).contiguous(),
        brain_mask=resized_mask.contiguous(),
        normalization=stats,
        geometry=geometry,
    )


def prepare_t1ce_supervised_arrays(
    case_id: str,
    t1ce: np.ndarray | torch.Tensor,
    segmentation: np.ndarray | torch.Tensor,
    spacing_mm: Sequence[float],
    *,
    output_shape: Sequence[int] = DEFAULT_MODEL_SHAPE,
) -> T1ceSupervisedView:
    inference = prepare_t1ce_inference_array(
        case_id, t1ce, spacing_mm, output_shape=output_shape
    )
    target = residual_tumor_target(segmentation)
    if tuple(target.shape) != inference.geometry.original_shape:
        raise T1ceBinaryGeometryError(
            f"{case_id} segmentation shape {tuple(target.shape)} != "
            f"T1ce shape {inference.geometry.original_shape}"
        )
    resized_target = _resize(
        target, inference.geometry.model_shape, "nearest-exact"
    ).bool()
    return T1ceSupervisedView(
        case_id=inference.case_id,
        image=inference.image,
        target=resized_target.unsqueeze(0).contiguous(),
        source_target=target.unsqueeze(0).contiguous(),
        brain_mask=inference.brain_mask,
        normalization=inference.normalization,
        geometry=inference.geometry,
    )


def _load_nifti(path: Path, description: str) -> tuple[np.ndarray, np.ndarray, Spacing]:
    if not path.is_file():
        raise T1ceBinaryDataError(f"missing {description}: {path}")
    try:
        image = nib.load(str(path), mmap=True)
        data = np.asanyarray(image.dataobj)
    except (OSError, ValueError) as exc:
        raise T1ceBinaryDataError(f"failed to read {description} {path}: {exc}") from exc
    if data.ndim != 3:
        raise T1ceBinaryDataError(f"{description} must be 3D, got {data.shape}")
    affine = np.asarray(image.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise T1ceBinaryDataError(f"{description} affine must be a finite 4x4 matrix")
    spacing = _spacing(nib.affines.voxel_sizes(affine), f"{description} spacing")
    return data, affine, spacing


def load_t1ce_inference_view(
    case: T1ceInferenceCase,
    *,
    output_shape: Sequence[int] = DEFAULT_MODEL_SHAPE,
) -> T1ceInferenceView:
    if not isinstance(case, T1ceInferenceCase):
        raise TypeError("load_t1ce_inference_view requires T1ceInferenceCase")
    image, _, spacing = _load_nifti(case.t1ce_path, "T1ce image")
    return prepare_t1ce_inference_array(
        case.case_id, image, spacing, output_shape=output_shape
    )


def load_t1ce_supervised_view(
    case: T1ceSupervisedCase,
    *,
    output_shape: Sequence[int] = DEFAULT_MODEL_SHAPE,
) -> T1ceSupervisedView:
    if not isinstance(case, T1ceSupervisedCase):
        raise TypeError("load_t1ce_supervised_view requires T1ceSupervisedCase")
    image, image_affine, spacing = _load_nifti(case.t1ce_path, "T1ce image")
    segmentation, seg_affine, _ = _load_nifti(case.seg_path, "segmentation")
    if image.shape != segmentation.shape:
        raise T1ceBinaryGeometryError(
            f"{case.case_id} segmentation shape {segmentation.shape} != T1ce shape {image.shape}"
        )
    if not np.allclose(image_affine, seg_affine, rtol=0.0, atol=1e-5):
        raise T1ceBinaryGeometryError(f"{case.case_id} segmentation affine mismatch")
    return prepare_t1ce_supervised_arrays(
        case.case_id, image, segmentation, spacing, output_shape=output_shape
    )


@dataclass(frozen=True)
class BinaryCaseMetrics:
    dsc: float
    iou: float
    hd95_mm: float
    prediction_present: bool
    target_present: bool
    spacing_mm: Spacing
    empty_policy: str

    def to_dict(self) -> dict[str, float | bool | list[float] | str]:
        result = asdict(self)
        result["spacing_mm"] = list(self.spacing_mm)
        return result


@dataclass(frozen=True)
class DualGridBinaryMetrics:
    model_grid: BinaryCaseMetrics
    source_grid: BinaryCaseMetrics
    restore_mode: str = "nearest-exact"
    headline_metric: str = "unresolved_reconstruction_assumption"

    def to_dict(self) -> dict[str, object]:
        return {
            "model_grid": self.model_grid.to_dict(),
            "source_grid": self.source_grid.to_dict(),
            "restore_mode": self.restore_mode,
            "headline_metric": self.headline_metric,
        }


def _binary_array(value: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    array = np.asarray(value.detach().cpu() if isinstance(value, torch.Tensor) else value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"{name} must be [D,H,W] or [1,D,H,W], got {array.shape}")
    if not np.isin(array, (False, True, 0, 1)).all():
        raise ValueError(f"{name} must be binary; threshold probabilities before scoring")
    return array.astype(bool, copy=False)


def binary_case_metrics(
    prediction: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    spacing_mm: Sequence[float],
    *,
    empty_hd95_penalty_mm: float | None = None,
) -> BinaryCaseMetrics:
    """Conventional voxel DSC/IoU and symmetric surface HD95 in millimetres."""
    prediction_array = _binary_array(prediction, "prediction")
    target_array = _binary_array(target, "target")
    if prediction_array.shape != target_array.shape:
        raise ValueError(
            f"prediction/target shape mismatch: {prediction_array.shape} != {target_array.shape}"
        )
    spacing = _spacing(spacing_mm)
    penalty = (
        float(empty_hd95_penalty_mm)
        if empty_hd95_penalty_mm is not None
        else float(np.linalg.norm(np.asarray(prediction_array.shape) * np.asarray(spacing)))
    )
    if not np.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("empty_hd95_penalty_mm must be finite and positive")
    prediction_present = bool(prediction_array.any())
    target_present = bool(target_array.any())
    if not prediction_present and not target_present:
        dsc, iou, hd95 = 1.0, 1.0, 0.0
    elif not prediction_present or not target_present:
        dsc, iou, hd95 = 0.0, 0.0, penalty
    else:
        intersection = int(np.logical_and(prediction_array, target_array).sum())
        prediction_count = int(prediction_array.sum())
        target_count = int(target_array.sum())
        union = prediction_count + target_count - intersection
        dsc = 2.0 * intersection / (prediction_count + target_count)
        iou = intersection / union
        distances = surface_metrics.compute_surface_distances(
            target_array, prediction_array, spacing
        )
        hd95 = float(surface_metrics.compute_robust_hausdorff(distances, 95.0))
        if not np.isfinite(hd95):
            hd95 = penalty
    return BinaryCaseMetrics(
        dsc=float(dsc),
        iou=float(iou),
        hd95_mm=float(hd95),
        prediction_present=prediction_present,
        target_present=target_present,
        spacing_mm=spacing,
        empty_policy=(
            "both_empty:dsc=1,iou=1,hd95=0;"
            f"one_empty:dsc=0,iou=0,hd95={penalty:g}mm"
        ),
    )


def score_t1ce_supervised_view(
    prediction: np.ndarray | torch.Tensor,
    view: T1ceSupervisedView,
    *,
    empty_hd95_penalty_mm: float | None = None,
) -> BinaryCaseMetrics:
    """Restore a thresholded prediction and score on the source physical grid."""
    if not isinstance(view, T1ceSupervisedView):
        raise TypeError("score_t1ce_supervised_view requires T1ceSupervisedView")
    prediction_array = _binary_array(prediction, "prediction")
    if prediction_array.shape != view.geometry.model_shape:
        raise ValueError(
            "prediction shape must match model grid: "
            f"{prediction_array.shape} != {view.geometry.model_shape}"
        )
    restored = _resize(
        torch.as_tensor(prediction_array),
        view.geometry.original_shape,
        "nearest-exact",
    ).bool()
    return binary_case_metrics(
        restored,
        view.source_target,
        view.geometry.original_spacing_mm,
        empty_hd95_penalty_mm=empty_hd95_penalty_mm,
    )


def score_t1ce_dual_grid(
    prediction: np.ndarray | torch.Tensor,
    view: T1ceSupervisedView,
    *,
    empty_hd95_penalty_mm: float | None = None,
) -> DualGridBinaryMetrics:
    """Report model-grid and restored source-grid metrics without picking a headline."""
    if not isinstance(view, T1ceSupervisedView):
        raise TypeError("score_t1ce_dual_grid requires T1ceSupervisedView")
    prediction_array = _binary_array(prediction, "prediction")
    if prediction_array.shape != view.geometry.model_shape:
        raise ValueError(
            "prediction shape must match model grid: "
            f"{prediction_array.shape} != {view.geometry.model_shape}"
        )
    model_grid = binary_case_metrics(
        prediction_array,
        view.target,
        view.geometry.model_spacing_mm,
        empty_hd95_penalty_mm=empty_hd95_penalty_mm,
    )
    source_grid = score_t1ce_supervised_view(
        prediction_array,
        view,
        empty_hd95_penalty_mm=empty_hd95_penalty_mm,
    )
    return DualGridBinaryMetrics(model_grid=model_grid, source_grid=source_grid)
