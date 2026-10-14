import copy
import numpy as np
import pytest
from sammed3d import spatial_roundtrip_audit as core
from scripts.run_fit_spatial_roundtrip_v2 import voxel_units
from scripts.finalize_fit_spatial_roundtrip import verify_row


def row():
    truth = np.zeros((8, 8, 8), dtype=bool)
    truth[0, 0, 0] = True
    truth[4:, 4:, 4:] = True
    cache = core.resize(truth, (4, 4, 4), 'nearest-exact').astype(np.uint8)
    return voxel_units(core.analyze(truth, [0]*3, [8]*3, cache))


def test_saved_counts_and_lost_component_consistent():
    verify_row(row())


@pytest.mark.parametrize('field', ['intersection', 'dice', 'component_size'])
def test_tampered_counts_rejected(field):
    value = copy.deepcopy(row())
    if field == 'component_size':
        value['components']['details'][0]['native_voxels'] += 1
    else:
        value['restorations']['nearest-exact'][field] += 1
    with pytest.raises(AssertionError):
        verify_row(value)
