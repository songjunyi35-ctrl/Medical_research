"""Fit-only metadata audit for the segmentation data-processing literature review."""
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from scripts import structured_background_protocol as p


def main():
    record = p.ROOT / 'archive/reviews/2026-09-08_segmentation_data_processing'
    output = record / 'FIT_GEOMETRY.json'
    if output.exists():
        raise FileExistsError(output)
    p.checked(p.FIT, p.FIT_SHA)
    p.checked(p.CACHE / 'index.json', p.INDEX_SHA)
    fit = p.rows(p.FIT)
    index = json.loads((p.CACHE / 'index.json').read_text())
    extents = np.asarray([index['crop_geometry'][r['case_id']]['extent'] for r in fit], dtype=float)
    assert len(fit) == 684 and extents.shape == (684, 3) and (extents > 0).all()
    ratio = extents.max(axis=1) / extents.min(axis=1)
    counts = np.asarray([index['per_case_positive_voxels'][r['case_id']] for r in fit])
    data = {'created_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'fit_metadata_only_no_method_or_metric_change',
            'fit_scans': 684, 'fit_subjects': len({r['subject_id'] for r in fit}),
            'fit_sha256': p.FIT_SHA, 'cache_index_sha256': p.INDEX_SHA,
            'native_shapes': sorted({tuple(index['crop_geometry'][r['case_id']]['native_shape']) for r in fit}),
            'extent_min_median_max_by_axis': np.quantile(extents, [0, .5, 1], axis=0).tolist(),
            'axis_scale_ratio_min_median_p95_max': np.quantile(ratio, [0, .5, .95, 1]).tolist(),
            'native_voxel_step_per_output_voxel_min_median_max': np.quantile(extents / 128, [0, .5, 1], axis=0).tolist(),
            'cached_WT_voxels_min_q25_median_max': np.quantile(counts, [0, .25, .5, 1]).tolist(),
            'empty_cached_whole_tumor_scans': int((counts == 0).sum()),
            'limits': ['These are native voxel-index scale ratios, not measured millimetre spacings.',
                       'Shape distortion is observed; its effect on segmentation accuracy is not measured.',
                       'Nonempty whole-tumor masks do not prove all small connected lesions survive resizing.',
                       'No label round-trip experiment or model accuracy ceiling was estimated.'],
            'access': {'fit_manifest_and_cache_metadata': True, 'image_or_label_arrays': False,
                       'development_metrics': False, 'new_model_forwards': 0},
            'script_sha256': p.sha(Path(__file__))}
    p.write_new(output, data)
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
