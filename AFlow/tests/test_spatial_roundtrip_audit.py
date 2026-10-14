import multiprocessing as mp
import nibabel as nib
import numpy as np
import pytest
from sammed3d import spatial_roundtrip_audit as m
from scripts.run_fit_spatial_roundtrip import canonical_header_geometry


def test_identity_preserves_shape_components_and_surfaces():
    truth = np.zeros((8, 8, 8), dtype=bool)
    truth[2:5, 2:5, 2:5] = True
    result = m.analyze(truth, [0, 0, 0], [8, 8, 8], truth.astype(np.uint8), (1., 2., 3.))
    assert result['cache_byte_exact'] and result['crop_lost_WT_voxels'] == 0
    for metric in result['restorations'].values():
        assert metric['dice'] == 1 and metric['hd95_mm'] == 0 and metric['surface_dice_1mm'] == 1
    assert result['components']['represented_native_components'] == 1


def test_crop_loss_is_reported_not_excluded():
    truth = np.zeros((8, 8, 8), dtype=bool)
    truth[1, 1, 1] = truth[4, 4, 4] = True
    cache = truth[2:6, 2:6, 2:6].astype(np.uint8)
    result = m.analyze(truth, [2, 2, 2], [4, 4, 4], cache, with_surfaces=False)
    assert result['crop_lost_WT_voxels'] == 1
    assert result['crop_only_dice'] == pytest.approx(2 / 3)
    assert result['components']['fully_crop_lost_components'] == 1
    assert result['components']['resampling_lost_components'] == 0


def test_single_voxel_component_lost_during_downsampling():
    truth = np.zeros((8, 8, 8), dtype=bool)
    truth[0, 0, 0] = True
    truth[4:8, 4:8, 4:8] = True
    cache = m.resize(truth, (4, 4, 4), 'nearest-exact').astype(np.uint8)
    result = m.analyze(truth, [0, 0, 0], [8, 8, 8], cache, with_surfaces=False)
    assert result['components']['native_count_26'] == 2
    assert result['components']['resampling_lost_components'] == 1
    assert result['components']['bins']['1']['resampling_lost'] == 1
    assert result['components']['fully_crop_lost_components'] == 0


def test_two_native_components_merge_in_coarse_grid():
    truth = np.zeros((4, 4, 4), dtype=bool)
    truth[1, 1, 1] = truth[3, 1, 1] = True
    cache = m.resize(truth, (2, 2, 2), 'nearest-exact').astype(np.uint8)
    result = m.analyze(truth, [0, 0, 0], [4, 4, 4], cache, with_surfaces=False)
    assert result['components']['represented_native_components'] == 2
    assert result['components']['cached_count_26'] == 1
    assert result['components']['coarse_components_merging_multiple_native'] == 1


def test_native_bridge_loss_splits_a_component():
    truth = np.zeros((8, 8, 8), dtype=bool)
    truth[1:8, 0, 1] = True
    truth[1, 1, 1] = truth[7, 1, 1] = True
    cache = m.resize(truth, (4, 4, 4), 'nearest-exact').astype(np.uint8)
    result = m.analyze(truth, [0, 0, 0], [8, 8, 8], cache, with_surfaces=False)
    assert result['components']['native_count_26'] == 1
    assert result['components']['native_components_split_in_coarse_grid'] == 1


def test_cache_mismatch_rejected():
    truth = np.ones((4, 4, 4), dtype=bool)
    cached = np.ones((2, 2, 2), dtype=np.uint8)
    cached[0, 0, 0] = 0
    with pytest.raises(ValueError, match='byte parity'):
        m.analyze(truth, [0, 0, 0], [4, 4, 4], cached)


def test_linear_half_is_excluded_at_strict_threshold():
    source = np.asarray([1, 0], dtype=np.uint8).reshape(2, 1, 1)
    restored = m.resize(source, (3, 1, 1), 'trilinear').ravel()
    assert restored.tolist() == [1., .5, 0.]
    assert (restored > .5).tolist() == [True, False, False]


@pytest.mark.parametrize('origin,extent', [([-1, 0, 0], [4, 4, 4]), ([0, 0, 0], [0, 4, 4]), ([1, 0, 0], [4, 4, 4])])
def test_invalid_crop_rejected(origin, extent):
    with pytest.raises(ValueError, match='crop'):
        m.analyze(np.ones((4, 4, 4), dtype=bool), origin, extent, np.ones((2, 2, 2), dtype=np.uint8))


def test_empty_native_rejected():
    with pytest.raises(ValueError, match='empty native'):
        m.analyze(np.zeros((4, 4, 4), dtype=bool), [0]*3, [4]*3, np.zeros((2, 2, 2), dtype=np.uint8))


def test_empty_reconstruction_has_null_distance_not_zero():
    truth = np.ones((4, 4, 4), dtype=bool)
    result = m.measure(truth, np.zeros_like(truth), (1, 1, 1))
    assert result['dice'] == 0 and result['surface_dice_1mm'] == 0 and result['hd95_mm'] is None


def test_header_only_permutation_matches_nibabel_canonical():
    affine = np.asarray([[0, -2, 0, 10], [0, 0, -3, 20], [4, 0, 0, 30], [0, 0, 0, 1]], dtype=float)
    image = nib.Nifti1Image(np.ones((2, 3, 4), dtype=np.uint8), affine)
    expected = nib.as_closest_canonical(image)
    shape, actual = canonical_header_geometry(image.shape, image.affine)
    assert shape == list(expected.shape)
    assert np.array_equal(actual, expected.affine)


def test_spawn_process_runs_same_synthetic_operator():
    truth = np.zeros((8, 8, 8), dtype=bool)
    truth[2:5, 2:5, 2:5] = True
    args = (truth, [0]*3, [8]*3, truth.astype(np.uint8))
    expected = m.analyze(*args)
    with mp.get_context('spawn').Pool(1) as pool:
        actual = pool.apply_async(m.analyze, args).get(timeout=60)
    assert actual == expected
