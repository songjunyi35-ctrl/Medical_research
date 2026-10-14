#!/usr/bin/env python3
"""Post-hoc, zero-forward error decomposition for the closed UCSD BraTS-2024-test panel.

Reads only saved per-case counts from the archived raw report plus the archived
train-native-volume artifact. Descriptive only: no model access, no re-evaluation,
no threshold/router authorization.
"""
from __future__ import annotations
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts import ucsd_brats2024test_protocol as p

TRAIN_ARTIFACT = p.ROOT / 'archive/research_records/2026-08-21_voxel_kd_data972_native_sampling/NATIVE_WT_VOLUMES.json'
SEED = 20261901


def interval(values, seed):
    a = np.asarray(values, dtype=np.float64)
    draw = np.random.default_rng(seed).integers(0, len(a), size=(p.BOOTSTRAP_SAMPLES, len(a)))
    return [float(x) for x in np.quantile(a[draw].mean(1), (.025, .975))]


def precision(row):
    return 1.0 if row['predicted_wt_voxels'] == 0 else row['intersection_voxels'] / row['predicted_wt_voxels']


def recall(row):
    return row['intersection_voxels'] / row['target_wt_voxels']


def main():
    out = p.RECORD / 'POSTHOC_ERROR_AUDIT.json'
    if out.exists():
        raise FileExistsError(out)
    p.docs()
    raw = json.loads((p.RECORD / 'RAW_UCSD_BRATS2024TEST56.json').read_text())
    rows = {'full_identity': raw['full972_kd168']['identity']['per_case'],
            'full_flip8': raw['full972_kd168']['flip8']['per_case'],
            'gcsam': raw['gcsam184']['identity']['per_case']}
    summary = {}
    for name, rs in rows.items():
        summary[name] = {'precision_mean': statistics.fmean(precision(x) for x in rs),
                         'recall_mean': statistics.fmean(recall(x) for x in rs),
                         'predicted_voxels_mean': statistics.fmean(x['predicted_wt_voxels'] for x in rs),
                         'target_voxels_mean': statistics.fmean(x['target_wt_voxels'] for x in rs),
                         'pred_over_target_mean_ratio': statistics.fmean(x['predicted_wt_voxels'] for x in rs) / statistics.fmean(x['target_wt_voxels'] for x in rs)}
    deltas = {}
    for name, left, right, seed in (('gcsam_minus_full', 'full_identity', 'gcsam', SEED),
                                    ('flip8_minus_identity', 'full_identity', 'full_flip8', SEED + 50)):
        lr, rr = rows[left], rows[right]
        assert [x['case_id'] for x in lr] == [x['case_id'] for x in rr]
        dp = [precision(r) - precision(l) for l, r in zip(lr, rr)]
        dr = [recall(r) - recall(l) for l, r in zip(lr, rr)]
        dv = [r['predicted_wt_voxels'] - l['predicted_wt_voxels'] for l, r in zip(lr, rr)]
        deltas[name] = {'precision_delta_mean': statistics.fmean(dp), 'precision_delta_posthoc_95ci': interval(dp, seed),
                        'recall_delta_mean': statistics.fmean(dr), 'recall_delta_posthoc_95ci': interval(dr, seed + 1),
                        'predicted_voxels_delta_mean': statistics.fmean(dv), 'predicted_voxels_delta_posthoc_95ci': interval(dv, seed + 2)}
    train = json.loads(TRAIN_ARTIFACT.read_text())['native_wt_voxels']
    train_sorted = np.sort(np.asarray(list(train.values()), dtype=np.float64))
    targets = [x['target_wt_voxels'] for x in rows['full_identity']]
    # raw native WT for the cohort comes from the cache index details (1mm voxels, same physical unit as train)
    cache_index = json.loads((p.CACHE / 'index.json').read_text())
    raw_wt = [cache_index['details'][x['case_id']]['raw_wt'] for x in rows['full_identity']]
    percentiles = [float((train_sorted <= v).mean()) for v in raw_wt]
    support = {'cohort_raw_wt_median': float(np.median(raw_wt)),
               'train972_native_wt_median': float(np.median(train_sorted)),
               'cohort_train_percentile_median': float(np.median(percentiles)),
               'cases_above_train_p90': int(sum(1 for x in percentiles if x > 0.90)),
               'cases_above_train_p95': int(sum(1 for x in percentiles if x > 0.95)),
               'cases_above_train_max': int(sum(1 for x in percentiles if x >= 1.0))}
    gcsam_miss = [x['case_id'] for x in rows['gcsam'] if not x['automatic_point_hits_gt']]
    report = {'schema': 'aflow.ucsd_brats2024test.posthoc_error_audit', 'schema_version': 1,
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'descriptive_only': True, 'new_forwards': 0, 'model_inference': False,
              'empty_prediction_precision_convention': 1.0,
              'per_model': summary, 'posthoc_deltas': deltas,
              'train_target_support': support,
              'cache_target_voxels_128_mean': statistics.fmean(targets),
              'gcsam_point_miss_cases': gcsam_miss,
              'raw_report_sha256': p.sha(p.RECORD / 'RAW_UCSD_BRATS2024TEST56.json'),
              'no_router_threshold_or_repeat_authorized': True}
    p.write_new(out, report)
    print(json.dumps({'per_model': summary, 'posthoc_deltas': deltas, 'train_target_support': support,
                      'gcsam_point_miss_cases': gcsam_miss}, indent=2))


if __name__ == '__main__':
    main()
