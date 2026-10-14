#!/usr/bin/env python3
"""Terminal verification for the UCSD BraTS-2024-test panel: re-hash every input and independently recompute all statistics from saved per-case rows."""
from __future__ import annotations
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts import ucsd_brats2024test_protocol as p
from scripts.evaluate_ucsd_brats2024test_panel import interval, subject_means


def main():
    out = p.RECORD / 'VERIFICATION_RECEIPT.json'
    if out.exists():
        raise FileExistsError(out)
    p.docs()
    raw = json.loads((p.RECORD / 'RAW_UCSD_BRATS2024TEST56.json').read_text())
    source = json.loads(p.SOURCE_RECEIPT.read_text())
    cache = json.loads(p.CACHE_RECEIPT.read_text())
    checks = {}
    mismatch = []
    for entry in source['selected']['files']:
        for kind in ('image', 'mask'):
            path = Path(entry[f'{kind}_path'])
            if p.sha(path) != entry[f'{kind}_sha256']:
                mismatch.append(str(path))
    checks['copy_a_rehash_112_exact'] = not mismatch
    checks['full_checkpoint_sha_exact'] = p.sha(p.FULL) == p.FULL_SHA == raw['full972_kd168']['checkpoint_sha256']
    checks['gcsam_checkpoint_sha_exact'] = p.sha(p.GCSAM) == p.GCSAM_SHA == raw['gcsam184']['checkpoint_sha256']
    checks['cache_files_rehash_exact'] = (p.sha(p.CACHE / 'images.f32') == cache['images_sha256']
                                          and p.sha(p.CACHE / 'labels.u8') == cache['labels_sha256']
                                          and p.sha(p.CACHE / 'index.json') == cache['index_sha256'])
    recomputed = {}
    summaries = {'full_identity': raw['full972_kd168']['identity'], 'full_flip8': raw['full972_kd168']['flip8'],
                 'gcsam_identity': raw['gcsam184']['identity']}
    for name, summary in summaries.items():
        rows = summary['per_case']
        means = subject_means(rows)
        recomputed[name] = {
            'scan_mean_exact': abs(statistics.fmean(x['locator_dsc'] for x in rows) - summary['locator_dsc_mean']) < 1e-12,
            'subject_mean_exact': abs(statistics.fmean(means.values()) - summary['locator_subject_dsc_mean']) < 1e-12,
            'point_hits_exact': sum(x['automatic_point_hits_gt'] for x in rows) == summary['automatic_point_hits'],
            'dice_identity_exact': all(abs(x['locator_dsc'] - 2 * x['intersection_voxels'] / (x['predicted_wt_voxels'] + x['target_wt_voxels'])) < 1e-12 for x in rows),
        }
    seeds = {'full_identity': p.SEED_ABS_FULL_IDENTITY, 'full_flip8': p.SEED_ABS_FULL_FLIP8, 'gcsam_identity': p.SEED_ABS_GCSAM}
    stored = {'full_identity': raw['full972_kd168']['identity_bootstrap'], 'full_flip8': raw['full972_kd168']['flip8_bootstrap'],
              'gcsam_identity': raw['gcsam184']['identity_bootstrap']}
    for name, summary in summaries.items():
        rows = summary['per_case']
        scan_ci = interval([x['locator_dsc'] for x in rows], seeds[name])
        subj_ci = interval(list(subject_means(rows).values()), seeds[name] + 1)
        recomputed[name]['absolute_ci_exact'] = (scan_ci == stored[name]['scan_dsc_mean_95ci']
                                                 and subj_ci == stored[name]['subject_dsc_mean_95ci'])
    for comp_name, left, right, seed in (
            ('full_flip8_minus_identity', 'full_identity', 'full_flip8', p.SEED_PAIRED_FLIP8),
            ('gcsam_identity_minus_full_identity', 'full_identity', 'gcsam_identity', p.SEED_PAIRED_GCSAM)):
        stored_cmp = raw['comparisons'][comp_name]
        lrows, rrows = summaries[left]['per_case'], summaries[right]['per_case']
        delta_rows = [{'case_id': l['case_id'], 'subject_id': l['subject_id'], 'locator_dsc': r['locator_dsc'] - l['locator_dsc']}
                      for l, r in zip(lrows, rrows, strict=True)]
        scan_delta = [x['locator_dsc'] for x in delta_rows]
        subj_delta = list(subject_means(delta_rows).values())
        key = [k for k in stored_cmp if k.endswith('minus_identity') or k.endswith('minus_full')][0]
        recomputed[comp_name] = {
            'scan_delta_exact': abs(statistics.fmean(scan_delta) - stored_cmp[key]['scan_dice']) < 1e-12,
            'subject_delta_exact': abs(statistics.fmean(subj_delta) - stored_cmp[key]['subject_dice']) < 1e-12,
            'paired_ci_exact': (interval(scan_delta, seed) == stored_cmp['paired_scan_dice_95ci']
                                and interval(subj_delta, seed + 1) == stored_cmp['paired_subject_dice_95ci']),
            'counts_exact': (sum(x > 0 for x in scan_delta) == stored_cmp['cases_improved']
                             and sum(x < 0 for x in scan_delta) == stored_cmp['cases_worse']),
        }
    flat = all(v for group in recomputed.values() for v in group.values()) and all(checks.values())
    receipt = {'schema': 'aflow.ucsd_brats2024test.verification_receipt', 'schema_version': 1,
               'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'status': 'passed' if flat else 'failed',
               'input_rehash': checks, 'recomputation': recomputed,
               'raw_report_sha256': p.sha(p.RECORD / 'RAW_UCSD_BRATS2024TEST56.json'),
               'source_receipt_sha256': p.sha(p.SOURCE_RECEIPT),
               'cache_receipt_sha256': p.sha(p.CACHE_RECEIPT),
               'model_preaccess_sha256': p.sha(p.MODEL_PREACCESS),
               'model_inference': False}
    p.write_new(out, receipt)
    print(json.dumps({'status': receipt['status'], 'input_rehash': checks,
                      'recomputation_flat_pass': flat}, indent=2))
    if not flat:
        raise RuntimeError('verification failed')


if __name__ == '__main__':
    main()
