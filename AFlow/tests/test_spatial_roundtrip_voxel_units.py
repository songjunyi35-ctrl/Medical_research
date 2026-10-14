import numpy as np
from sammed3d import spatial_roundtrip_audit as core
from scripts.run_fit_spatial_roundtrip_v2 import voxel_units, summarize


def test_voxel_unit_adapter_keeps_overlap_and_relabels_all_distances():
    truth = np.ones((4, 4, 4), dtype=bool)
    original = core.analyze(truth, [0]*3, [4]*3, truth.astype(np.uint8))
    result = voxel_units(original)
    assert result['metric_spacing_native_voxels'] == [1., 1., 1.]
    assert 'spacing_mm' not in result
    for metric in result['restorations'].values():
        assert metric['dice'] == 1
        assert metric['hd95_native_voxels'] == 0
        assert metric['surface_dice_one_native_voxel'] == 1
        assert not any(k.endswith('_mm') or '1mm' in k for k in metric)


def test_summary_reports_voxel_units_for_every_case():
    truth = np.ones((4, 4, 4), dtype=bool)
    row = voxel_units(core.analyze(truth, [0]*3, [4]*3, truth.astype(np.uint8)))
    row['subject_id'] = 'synthetic'
    result = summarize([row])
    assert result['restorations']['nearest-exact']['hd95_native_voxels']['scan_mean'] == 0
    assert result['metric_spacing_native_voxels_counts'] == {'(1.0, 1.0, 1.0)': 1}
