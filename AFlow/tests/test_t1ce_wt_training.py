from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
import torch

from sammed3d.t1ce_binary import prepare_t1ce_supervised_arrays
from sammed3d import t1ce_wt_training as training


def _row(split: str, image_path: str = "/image.nii") -> dict:
    return {
        "schema": "aflow.brats2024.sammed3d_t1ce_wt_inner_split",
        "case_id": "BraTS-GLI-00005-100",
        "subject_id": "BraTS-GLI-00005",
        "modality": "t1ce",
        "target": "WT",
        "image_path": image_path,
        "label_path": "/label.nii",
        "foreground_labels": [1, 2, 3],
        "excluded_labels": [4],
        "split": split,
    }


def test_whole_resize_canonicalizes_and_matches_geometry_reference(
    tmp_path,
) -> None:
    image = np.arange(1, 5 * 6 * 7 + 1, dtype=np.float32).reshape(5, 6, 7)
    segmentation = np.zeros_like(image, dtype=np.uint8)
    segmentation[1:4, 2:5, 3:6] = 2
    segmentation[0, 0, 0] = 4
    affine = np.diag([-2.0, 1.5, 1.0, 1.0])
    image_path = tmp_path / "t1c.nii.gz"
    label_path = tmp_path / "seg.nii.gz"
    nib.save(nib.Nifti1Image(image, affine), image_path)
    nib.save(nib.Nifti1Image(segmentation, affine), label_path)
    row = {
        **_row("train", str(image_path)),
        "label_path": str(label_path),
    }

    actual_image, actual_target = training.load_whole_volume_resize(
        row, model_shape=(8, 6, 4)
    )

    canonical_image = nib.as_closest_canonical(nib.load(image_path))
    canonical_label = nib.as_closest_canonical(nib.load(label_path))
    expected = prepare_t1ce_supervised_arrays(
        row["case_id"],
        np.ascontiguousarray(np.asanyarray(canonical_image.dataobj)),
        np.ascontiguousarray(np.asanyarray(canonical_label.dataobj)),
        nib.affines.voxel_sizes(canonical_image.affine),
        output_shape=(8, 6, 4),
    )
    assert actual_image.shape == (1, 8, 6, 4)
    assert actual_target.shape == (1, 8, 6, 4)
    assert torch.equal(actual_image, expected.image)
    assert torch.equal(actual_target.bool(), expected.target)
    assert not bool(actual_target[0, -1, 0, 0])


def test_whole_resize_is_not_normalized_twice() -> None:
    preprocessed = torch.tensor([[[[[-1.0, 0.0, 1.0]]]]])
    forwarded = training.identity_preprocessed_batch(preprocessed)

    assert forwarded is preprocessed
    assert torch.equal(forwarded, preprocessed)


def test_augmentation_preset_contains_affine_elastic_and_noise() -> None:
    config = training.T1ceWtPreprocessConfig(
        spatial_preprocessing=training.WHOLE_RESIZE,
        train_augmentation=training.AFFINE_ELASTIC_NOISE,
    )
    transform = training.build_training_augmentation(config)

    assert transform is not None
    assert [type(item).__name__ for item in transform.transforms] == [
        "RandomAffine",
        "RandomElasticDeformation",
        "RandomNoise",
    ]


def test_training_augmentation_never_applies_to_validation(
    monkeypatch,
) -> None:
    config = training.T1ceWtPreprocessConfig(
        spatial_preprocessing=training.WHOLE_RESIZE,
        train_augmentation=training.AFFINE_ELASTIC_NOISE,
        model_shape=(4, 4, 4),
    )
    image = torch.zeros(1, 4, 4, 4)
    target = torch.zeros_like(image)
    target[0, 1, 1, 1] = 1
    monkeypatch.setattr(
        training,
        "load_preprocessed_case",
        lambda row, config: (image.clone(), target.clone()),
    )
    sentinel_transform = object()
    monkeypatch.setattr(
        training, "build_training_augmentation", lambda config: sentinel_transform
    )
    calls = []

    def fake_apply(input_image, input_target, transform):
        calls.append(transform)
        return input_image + 1, input_target

    monkeypatch.setattr(training, "apply_training_augmentation", fake_apply)
    train_dataset = training.T1ceWtTrainingDataset(
        [_row("train")], config=config
    )
    validation_dataset = training.T1ceWtTrainingDataset(
        [_row("validation")], config=config
    )

    assert torch.all(train_dataset[0]["image"] == 1)
    assert torch.all(validation_dataset[0]["image"] == 0)
    assert calls == [sentinel_transform]
    assert validation_dataset.augmentation is None


def test_dataset_rejects_mixed_or_unknown_splits() -> None:
    config = training.T1ceWtPreprocessConfig()
    with pytest.raises(ValueError, match="one common"):
        training.T1ceWtTrainingDataset(
            [_row("train"), _row("validation")], config=config
        )
    with pytest.raises(ValueError, match="one common"):
        training.T1ceWtTrainingDataset([_row("test")], config=config)
