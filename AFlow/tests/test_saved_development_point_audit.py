import numpy as np
import pytest

from scripts.audit_saved_development_points import check_point


def test_asymmetric_c_order_coordinate():
    mask = np.zeros((2, 3, 4), dtype=np.uint8)
    mask[1, 0, 2] = 1
    row = {'target_voxels': 1, 'point_flat_index': 14, 'point_hit': True}
    result = check_point(row, mask)
    assert result['target_count_exact'] and result['point_membership_exact']
    row['point_flat_index'] = 2
    assert not check_point(row, mask)['point_membership_exact']


def test_incorrect_saved_count_detected():
    row = {'target_voxels': 2, 'point_flat_index': 0, 'point_hit': True}
    assert not check_point(row, np.ones((1, 1, 1), dtype=np.uint8))['target_count_exact']


@pytest.mark.parametrize('point', [-1, 8, True, .5])
def test_invalid_coordinate_rejected(point):
    row = {'target_voxels': 8, 'point_flat_index': point, 'point_hit': True}
    with pytest.raises(ValueError, match='point index'):
        check_point(row, np.ones((2, 2, 2), dtype=np.uint8))


@pytest.mark.parametrize('mask', [np.zeros((2, 2, 2), dtype=np.uint8), np.full((2, 2, 2), 2, dtype=np.uint8)])
def test_invalid_target_rejected(mask):
    with pytest.raises(ValueError):
        check_point({'target_voxels': 8, 'point_flat_index': 0, 'point_hit': True}, mask)
