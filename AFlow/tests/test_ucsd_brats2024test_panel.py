"""Contract tests for the UCSD BraTS-2024-test frozen panel (synthetic only; no pixels, no models)."""
from __future__ import annotations

import statistics

import pytest

from scripts import ucsd_brats2024test_protocol as p
from scripts import evaluate_ucsd_brats2024test_panel as ev


def synthetic_rows(offset=0.0):
    rows = []
    for i, case in enumerate(p.cases()):
        dsc = 0.5 + (i % 7) * 0.05 + offset
        rows.append({'case_id': case['case_id'], 'subject_id': case['subject_id'], 'locator_dsc': dsc,
                     'automatic_point_hits_gt': i % 3 != 0, 'automatic_point_flat_index': i,
                     'predicted_wt_voxels': 10 + i, 'target_wt_voxels': 12 + i, 'intersection_voxels': 8})
    return rows


def test_frozen_documents_and_cohort_shape():
    rows = p.cases()
    assert len(rows) == 56
    assert len({r['subject_id'] for r in rows}) == 41
    assert not any(r['subject_id'] == 'UCSD-PTGBM-0038' for r in rows)
    assert all(r['image_remote_path'].endswith('_T1post.nii.gz') for r in rows)
    assert all(r['mask_remote_path'].endswith('_BraTS_tumor_seg.nii.gz') for r in rows)


def test_summarize_subject_aggregation():
    rows = synthetic_rows()
    summary = ev.summarize(rows)
    assert summary['cases'] == 56 and summary['subjects'] == 41
    means = ev.subject_means(rows)
    assert summary['locator_subject_dsc_mean'] == pytest.approx(statistics.fmean(means.values()))
    multi = [s for s in means if sum(1 for r in rows if r['subject_id'] == s) > 1]
    assert multi, 'cohort must contain multi-scan subjects'
    s = multi[0]
    per_scan = [r['locator_dsc'] for r in rows if r['subject_id'] == s]
    assert means[s] == pytest.approx(statistics.fmean(per_scan))


def test_compare_paired_and_counts():
    left = synthetic_rows()
    right = synthetic_rows(offset=0.01)
    right[0]['locator_dsc'] = left[0]['locator_dsc']
    comparison = ev.compare(ev.summarize(left), ev.summarize(right), 123, 'delta')
    assert comparison['delta']['scan_dice'] == pytest.approx((0.01 * 55) / 56)
    assert comparison['cases_improved'] == 55 and comparison['cases_tied'] == 1 and comparison['cases_worse'] == 0
    assert comparison['subjects_improved'] + comparison['subjects_worse'] + comparison['subjects_tied'] == 41
    lo, hi = comparison['paired_scan_dice_95ci']
    assert lo <= comparison['delta']['scan_dice'] <= hi


def test_interval_deterministic():
    values = [0.01 * i for i in range(56)]
    assert ev.interval(values, 777) == ev.interval(values, 777)
    assert ev.interval(values, 777) != ev.interval(values, 778)


def test_interpretation_gate_branches():
    def comparison(delta, lo):
        return {'k': {'scan_dice': delta}, 'paired_scan_dice_95ci': [lo, lo + 0.05]}
    assert ev.interp(comparison(0.01, 0.001), 'k') == 'supported_on_ucsd_brats2024test56'
    assert ev.interp(comparison(0.01, -0.001), 'k') == 'positive_but_uncertain'
    assert ev.interp(comparison(-0.01, -0.02), 'k') == 'not_supported_or_reversed'


def test_summarize_rejects_wrong_cohort():
    rows = synthetic_rows()[:-1]
    with pytest.raises(ValueError):
        ev.summarize(rows)
