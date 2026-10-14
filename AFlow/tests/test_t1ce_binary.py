from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sammed3d.t1ce_binary import (
    DEFAULT_MODEL_SHAPE,
    RESIDUAL_TUMOR_LABELS,
    T1ceBinaryDataError,
    T1ceBinaryGeometryError,
    T1ceInferenceCase,
    T1ceSupervisedCase,
    binary_case_metrics,
    model_spacing_mm,
    prepare_t1ce_inference_array,
    prepare_t1ce_supervised_arrays,
    residual_tumor_target,
    score_t1ce_dual_grid,
    score_t1ce_supervised_view,
)


CASE_ID = "BraTS-GLI-00005-100"


def test_manifest_contract_selects_only_t1c_and_inference_rejects_gt() -> None:
    row = {
        "case_id": CASE_ID,
        "modalities": {
            "t1n": object(),
            "t1c": "/only/path-that-may-be-read.nii.gz",
            "t2w": object(),
            "t2f": object(),
        },
    }
    case = T1ceInferenceCase.from_manifest_row(row)

    assert str(case.t1ce_path) == "/only/path-that-may-be-read.nii.gz"
    assert {field.name for field in fields(case)} == {"case_id", "t1ce_path"}
    with pytest.raises(T1ceBinaryDataError, match="supervision field"):
        T1ceInferenceCase.from_manifest_row({**row, "seg_path": "/forbidden.nii.gz"})

    supervised = T1ceSupervisedCase.from_manifest_row(
        {**row, "seg_path": "/target.nii.gz"}
    )
    assert supervised.t1ce_path == case.t1ce_path


def test_residual_target_is_exactly_raw_labels_1_2_3() -> None:
    raw = np.arange(5, dtype=np.uint8).reshape(5, 1, 1)
    target = residual_tumor_target(raw)

    assert RESIDUAL_TUMOR_LABELS == frozenset({1, 2, 3})
    assert target.reshape(-1).tolist() == [False, True, True, True, False]
    invalid = raw.copy()
    invalid[0] = 5
    with pytest.raises(T1ceBinaryDataError, match="unexpected raw labels"):
        residual_tumor_target(invalid)


def test_residual_target_accepts_uint16_nifti_storage() -> None:
    raw = np.array([0, 1, 2, 3, 4], dtype=np.uint16).reshape(5, 1, 1)
    assert residual_tumor_target(raw).reshape(-1).tolist() == [
        False,
        True,
        True,
        True,
        False,
    ]


def test_whole_volume_resize_zscore_single_channel_and_physical_spacing() -> None:
    image = np.zeros((4, 6, 8), dtype=np.float32)
    image[1:4, 1:5, 2:7] = np.arange(1, 61, dtype=np.float32).reshape(3, 4, 5)
    view = prepare_t1ce_inference_array(
        CASE_ID, image, (2.0, 1.5, 1.0), output_shape=(8, 6, 4)
    )

    assert view.modality == "t1c"
    assert tuple(view.image.shape) == (1, 8, 6, 4)
    assert tuple(view.brain_mask.shape) == (8, 6, 4)
    assert view.geometry.original_shape == (4, 6, 8)
    assert view.geometry.model_shape == (8, 6, 4)
    assert view.geometry.model_spacing_mm == pytest.approx((1.0, 1.5, 2.0))
    assert view.normalization.nonzero_voxels == 60
    assert view.normalization.mean == pytest.approx(30.5)
    assert view.normalization.std == pytest.approx(np.arange(1, 61).std())
    assert torch.count_nonzero(view.image[:, ~view.brain_mask]) == 0


def test_default_grid_is_128_cubed_and_mask_uses_nearest_neighbor() -> None:
    image = np.zeros((2, 2, 2), dtype=np.float32)
    image[0] = 1.0
    image[1] = 3.0
    view = prepare_t1ce_inference_array(CASE_ID, image, (1.0, 1.0, 1.0))

    assert view.geometry.model_shape == DEFAULT_MODEL_SHAPE
    assert tuple(view.image.shape) == (1, *DEFAULT_MODEL_SHAPE)
    assert view.brain_mask.dtype is torch.bool


def test_mask_resize_uses_center_aligned_nearest_exact_not_legacy_nearest() -> None:
    image = np.arange(1, 5 * 6 * 7 + 1, dtype=np.float32).reshape(5, 6, 7)
    segmentation = np.zeros_like(image, dtype=np.uint8)
    segmentation[1:4, 2:5, 3:6] = 2
    view = prepare_t1ce_supervised_arrays(
        CASE_ID, image, segmentation, (1.0, 1.0, 1.0), output_shape=(4, 4, 4)
    )
    source = torch.from_numpy(segmentation == 2).float()[None, None]
    expected = F.interpolate(source, size=(4, 4, 4), mode="nearest-exact")[0]
    legacy = F.interpolate(source, size=(4, 4, 4), mode="nearest")[0]

    assert torch.equal(view.target, expected.bool())
    assert not torch.equal(view.target, legacy.bool())

    dual = score_t1ce_dual_grid(view.target, view)
    assert dual.model_grid.dsc == 1.0
    assert dual.model_grid.spacing_mm == pytest.approx((1.25, 1.5, 1.75))
    assert dual.source_grid.dsc < 1.0
    assert dual.restore_mode == "nearest-exact"
    assert dual.headline_metric == "unresolved_reconstruction_assumption"


def test_supervised_resize_is_nearest_and_excludes_rc() -> None:
    image = np.arange(1, 4 * 4 * 4 + 1, dtype=np.float32).reshape(4, 4, 4)
    segmentation = np.zeros_like(image, dtype=np.uint8)
    segmentation[0, 0, 0] = 1
    segmentation[1, 1, 1] = 2
    segmentation[2, 2, 2] = 3
    segmentation[3, 3, 3] = 4
    view = prepare_t1ce_supervised_arrays(
        CASE_ID, image, segmentation, (1.0, 1.0, 1.0), output_shape=(8, 8, 8)
    )

    assert tuple(view.target.shape) == (1, 8, 8, 8)
    assert tuple(view.source_target.shape) == (1, 4, 4, 4)
    assert view.target.dtype is torch.bool
    assert int(view.target.sum()) == 3 * 8
    assert view.foreground_raw_labels == (1, 2, 3)
    assert not bool(view.target[0, 7, 7, 7])


def test_supervised_shape_mismatch_fails_closed() -> None:
    image = np.arange(1, 28, dtype=np.float32).reshape(3, 3, 3)
    with pytest.raises(T1ceBinaryGeometryError, match="segmentation shape"):
        prepare_t1ce_supervised_arrays(
            CASE_ID,
            image,
            np.zeros((2, 3, 3), dtype=np.uint8),
            (1.0, 1.0, 1.0),
            output_shape=(4, 4, 4),
        )


def test_full_grid_target_is_not_clipped_to_the_t1ce_brain_mask() -> None:
    image = np.zeros((4, 4, 4), dtype=np.float32)
    image[1:3, 1:3, 1:3] = np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2)
    segmentation = np.zeros_like(image, dtype=np.uint8)
    segmentation[0, 0, 0] = 3

    view = prepare_t1ce_supervised_arrays(
        CASE_ID, image, segmentation, (2.0, 1.0, 1.0), output_shape=(4, 4, 4)
    )

    assert not bool(view.brain_mask[0, 0, 0])
    assert bool(view.target[0, 0, 0, 0])
    metrics = score_t1ce_supervised_view(view.target, view)
    assert metrics.dsc == 1.0
    assert metrics.spacing_mm == (2.0, 1.0, 1.0)


def test_metric_overlap_and_anisotropic_hd95_are_in_mm() -> None:
    target = np.zeros((8, 8, 8), dtype=bool)
    prediction = np.zeros_like(target)
    target[2:5, 2:5, 2:5] = True
    prediction[3:6, 2:5, 2:5] = True

    result = binary_case_metrics(prediction, target, (2.0, 1.0, 1.0))

    assert result.dsc == pytest.approx(2.0 / 3.0)
    assert result.iou == pytest.approx(0.5)
    assert result.hd95_mm == pytest.approx(2.0)
    assert result.spacing_mm == (2.0, 1.0, 1.0)


def test_metric_empty_policy_is_frozen() -> None:
    empty = np.zeros((4, 4, 4), dtype=bool)
    foreground = empty.copy()
    foreground[1, 1, 1] = True

    both_empty = binary_case_metrics(empty, empty, (1.0, 1.0, 1.0))
    one_empty = binary_case_metrics(foreground, empty, (1.0, 1.0, 1.0))

    assert (both_empty.dsc, both_empty.iou, both_empty.hd95_mm) == (1.0, 1.0, 0.0)
    assert (one_empty.dsc, one_empty.iou) == (0.0, 0.0)
    assert one_empty.hd95_mm == pytest.approx(np.sqrt(3.0 * 4.0**2))
    assert "one_empty" in one_empty.empty_policy


def test_score_restores_prediction_to_source_grid() -> None:
    image = np.arange(1, 4 * 6 * 8 + 1, dtype=np.float32).reshape(4, 6, 8)
    segmentation = np.full_like(image, 3, dtype=np.uint8)
    view = prepare_t1ce_supervised_arrays(
        CASE_ID, image, segmentation, (2.0, 1.5, 1.0), output_shape=(8, 6, 4)
    )

    result = score_t1ce_supervised_view(view.target, view)

    assert result.spacing_mm == (2.0, 1.5, 1.0)
    assert result.dsc == 1.0


def test_asymmetric_nearest_roundtrip_preserves_axes_and_direction() -> None:
    image = np.arange(1, 4 * 6 * 8 + 1, dtype=np.float32).reshape(4, 6, 8)
    segmentation = np.zeros_like(image, dtype=np.uint8)
    # The z extent uses source pairs because 8 -> 4 -> 8 nearest is exact for pairs.
    segmentation[1:3, 2:5, 2:4] = 3
    view = prepare_t1ce_supervised_arrays(
        CASE_ID, image, segmentation, (2.0, 1.5, 1.0), output_shape=(8, 6, 4)
    )

    result = score_t1ce_supervised_view(view.target, view)

    assert result.dsc == 1.0
    assert result.iou == 1.0
    assert result.hd95_mm == 0.0
    assert result.spacing_mm == (2.0, 1.5, 1.0)


def test_metric_rejects_logits_and_spacing_conversion_is_explicit() -> None:
    with pytest.raises(ValueError, match="must be binary"):
        binary_case_metrics(
            np.full((2, 2, 2), 0.6),
            np.zeros((2, 2, 2)),
            (1.0, 1.0, 1.0),
        )
    assert model_spacing_mm((256, 128, 64), (1.0, 2.0, 3.0)) == (
        2.0,
        2.0,
        1.5,
    )
