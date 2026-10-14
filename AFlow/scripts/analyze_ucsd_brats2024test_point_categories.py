#!/usr/bin/env python3
"""Post-hoc argmax-point category audit for the closed UCSD BraTS-2024-test panel (no inference).

Rebuilds the 128^3 multi-class label offline from raw masks, verifies WT equals the
frozen cache exactly, then classifies each saved automatic point as WT / resection
cavity(4) / other background. Descriptive only.
"""
from __future__ import annotations
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from scripts import ucsd_brats2024test_protocol as p


def resize_nearest(a):
    return F.interpolate(torch.from_numpy(np.ascontiguousarray(a)).float()[None, None], size=(128, 128, 128), mode='nearest-exact')[0, 0].numpy()


def main():
    out = p.RECORD / 'POSTHOC_POINT_CATEGORIES.json'
    if out.exists():
        raise FileExistsError(out)
    p.docs()
    raw = json.loads((p.RECORD / 'RAW_UCSD_BRATS2024TEST56.json').read_text())
    cache_receipt = json.loads(p.CACHE_RECEIPT.read_text())
    p.checked(p.CACHE / 'index.json', cache_receipt['index_sha256'])
    p.checked(p.CACHE / 'labels.u8', cache_receipt['labels_sha256'])
    idx = json.loads((p.CACHE / 'index.json').read_text())
    sources = {x['case_id']: x for x in json.loads(p.SOURCE_RECEIPT.read_text())['selected']['files']}
    cache = np.memmap(p.CACHE / 'labels.u8', dtype=np.uint8, mode='r', shape=(p.CASES, 128, 128, 128))
    rebuilt = {}
    for i, case in enumerate(idx['case_ids']):
        s = sources[case]
        p.checked(Path(s['mask_path']), s['mask_sha256'])
        arr = np.rint(np.asanyarray(nib.as_closest_canonical(nib.load(s['mask_path'], mmap=True)).dataobj))
        g = idx['details'][case]['geometry']
        crop = tuple(slice(o, o + e) for o, e in zip(g['origin'], g['extent']))
        view = arr[crop]
        wt = resize_nearest(np.isin(view, [1, 2, 3])).astype(bool)
        rc = resize_nearest(view == 4).astype(bool)
        if not np.array_equal(wt, cache[i].astype(bool)) or np.any(wt & rc):
            raise ValueError('label reconstruction ' + case)
        rebuilt[case] = {'wt': wt, 'rc': rc, 'rc_voxels': int(rc.sum())}
    models = {'full_identity': raw['full972_kd168']['identity']['per_case'],
              'full_flip8': raw['full972_kd168']['flip8']['per_case'],
              'gcsam_identity': raw['gcsam184']['identity']['per_case']}
    summary = {}
    for name, rows in models.items():
        categories = []
        misses = []
        for row in rows:
            r = rebuilt[row['case_id']]
            cat = 'wt' if r['wt'].flat[row['automatic_point_flat_index']] else (
                'resection_cavity' if r['rc'].flat[row['automatic_point_flat_index']] else 'other_background')
            categories.append(cat)
            if cat != 'wt':
                misses.append({'case_id': row['case_id'], 'category': cat, 'dice': row['locator_dsc'],
                               'rc_voxels': r['rc_voxels'], 'target_wt': row['target_wt_voxels']})
        summary[name] = {'point_categories': {x: categories.count(x) for x in ('wt', 'resection_cavity', 'other_background')},
                         'misses': misses}
    rc_present = sum(1 for r in rebuilt.values() if r['rc_voxels'] > 0)
    result = {'schema': 'aflow.ucsd_brats2024test.posthoc_point_categories', 'schema_version': 1,
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'posthoc_label_reconstruction_no_inference_or_selection',
              'cases': p.CASES, 'cache_wt_reconstruction_exact': p.CASES,
              'cases_with_rc4_present': rc_present,
              'rc_voxels_median': float(statistics.median(r['rc_voxels'] for r in rebuilt.values())),
              'models': summary,
              'interpretation': 'Descriptive only. GT resection cavity is unavailable at deployment; no avoidance postprocessing, routing, threshold or retest may follow.',
              'new_model_inference': False,
              'raw_report_sha256': p.sha(p.RECORD / 'RAW_UCSD_BRATS2024TEST56.json'),
              'analyzer_sha256': p.sha(Path(__file__))}
    p.write_new(out, result)
    print(json.dumps({'cases_with_rc4_present': rc_present,
                      'point_categories': {k: v['point_categories'] for k, v in summary.items()},
                      'gcsam_misses': summary['gcsam_identity']['misses']}, indent=2))


if __name__ == '__main__':
    main()
