from __future__ import annotations

from argparse import Namespace

import nibabel as nib
import numpy as np
import pytest
import torch
import torchio as tio

from scripts.build_t1ce_wt_official15_cache import preprocess_row
from scripts import train_sammed3d_t1ce_wt_official15_va as official


def _row(image_path, label_path) -> dict:
    return {
        "schema": "aflow.brats2024.sammed3d_t1ce_wt_inner_split",
        "case_id": "BraTS-GLI-SYNTH-000",
        "subject_id": "BraTS-GLI-SYNTH",
        "modality": "t1ce",
        "target": "WT",
        "image_path": str(image_path),
        "label_path": str(label_path),
        "foreground_labels": [1, 2, 3],
        "excluded_labels": [4],
        "split": "train",
    }


def test_official15_preprocess_matches_public_torchio_recipe(tmp_path) -> None:
    image = np.zeros((20, 24, 28), dtype=np.float32)
    grid = np.arange(12 * 14 * 16, dtype=np.float32).reshape(12, 14, 16)
    image[4:16, 5:19, 6:22] = 1 + grid / 100
    label = np.zeros_like(image, dtype=np.uint8)
    label[8:12, 10:14, 12:16] = 2
    label[9, 11, 13] = 3
    label[7, 9, 11] = 1
    label[2:4, 2:4, 2:4] = 4  # RC is explicitly excluded from WT.
    affine = np.diag([-2.0, 1.2, 2.5, 1.0])
    image_path = tmp_path / "t1c.nii.gz"
    label_path = tmp_path / "seg.nii.gz"
    nib.save(nib.Nifti1Image(image, affine), image_path)
    nib.save(nib.Nifti1Image(label, affine), label_path)

    actual_image, actual_label, qc = preprocess_row(_row(image_path, label_path))

    # Use the serialized NIfTI as the independent reference too. NIfTI stores
    # affine elements as float32, and TorchIO output grids are sensitive to it.
    stored_image = nib.load(image_path)
    stored_label = nib.load(label_path)
    image_on_disk = np.ascontiguousarray(
        np.asanyarray(stored_image.dataobj), dtype=np.float32
    )
    label_on_disk = np.asanyarray(stored_label.dataobj)
    subject = tio.Subject(
        image=tio.ScalarImage(
            tensor=torch.from_numpy(image_on_disk)[None], affine=stored_image.affine
        ),
        label=tio.LabelMap(
            tensor=torch.from_numpy(
                np.isin(label_on_disk, [1, 2, 3]).astype(np.uint8)
            )[None],
            affine=stored_label.affine,
        ),
    )
    deterministic = tio.CropOrPad(mask_name="label", target_shape=(128, 128, 128))(
        tio.ToCanonical()(
            tio.Resample((1.5, 1.5, 1.5), image_interpolation="linear")(subject)
        )
    )
    positive_before_zscore = deterministic.image.data > 0
    expected_image = tio.ZNormalization(masking_method=lambda x: x > 0)(
        deterministic.image
    ).data
    expected_label = (deterministic.label.data > 0).to(torch.uint8)

    assert actual_image.shape == (1, 128, 128, 128)
    assert actual_label.shape == (1, 128, 128, 128)
    assert np.allclose(actual_image, expected_image.numpy(), rtol=0, atol=1e-6)
    assert np.array_equal(actual_label, expected_label.numpy())
    normalized_source_positive = torch.from_numpy(actual_image)[positive_before_zscore]
    assert float(normalized_source_positive.mean()) == pytest.approx(0.0, abs=2e-6)
    assert float(normalized_source_positive.std(unbiased=True)) == pytest.approx(1.0, abs=2e-6)
    assert qc["resampled_spacing"] == [1.5, 1.5, 1.5]


def _args(**overrides) -> Namespace:
    values = {
        "mode": "formal",
        "spatial_preprocessing": "gt_crop",
        "train_augmentation": "random_flip",
        "unfreeze": "full",
        "encoder_lr": 1e-4,
        "head_lr": 1e-5,
        "weight_decay": 0.1,
        "lr_milestones": [120, 180],
        "lr_gamma": 0.1,
        "epochs": 200,
        "batch_size": 6,
        "accumulation_steps": 1,
        "min_epochs_before_stop": 201,
        "patience": 201,
    }
    values.update(overrides)
    return Namespace(**values)


def test_official15_two_gpu_contract_preserves_effective_batch_12() -> None:
    args = _args()
    official.validate_official_args(args, world_size=2)
    assert args.batch_size * 2 * args.accumulation_steps == 12


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("train_augmentation", "none"),
        ("batch_size", 12),
        ("epochs", 199),
        ("head_lr", 1e-4),
    ],
)
def test_official15_contract_fails_closed(override, value) -> None:
    with pytest.raises(ValueError, match="official15"):
        official.validate_official_args(_args(**{override: value}), world_size=2)


def test_official15_protocol_explicitly_records_no_n4_and_train_flip() -> None:
    protocol = official.official_protocol_dict()
    assert protocol["geometry"]["target_spacing_mm"] == [1.5, 1.5, 1.5]
    assert protocol["geometry"]["uses_gt_for_localization"] is True
    assert protocol["n4_bias_field_correction"] is False
    assert protocol["augmentation"]["name"] == "random_flip"
    assert protocol["augmentation"]["validation_augmentation"] is False
