"""Independently recompute frozen 108 results from integer confusion counts.

No logits, masks or labels are loaded and no model is called. All methods must
use the complete original manifest order; bootstrap draws preserve patients.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .governance import ROOT, RECORD, atomic_json, sha

MANIFEST = ROOT / 'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/validation_t1ce_wt.jsonl'


def validate(records, manifest):
    if [(r['case_id'], r['subject_id']) for r in records] != [
            (r['case_id'], r['subject_id']) for r in manifest]:
        raise ValueError('missing, reordered or substituted validation case')
    for row in records:
        tp, fp, fn = (row[key] for key in ('tp', 'fp', 'fn'))
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in (tp, fp, fn)):
            raise ValueError('confusion counts must be nonnegative integers')
        denominator = 2 * tp + fp + fn
        expected = float(np.float32(2 * tp) / np.float32(denominator)) if denominator else 1.
        if not np.isfinite(row['locator_dsc']) or abs(row['locator_dsc'] - expected) > 1e-7:
            raise ValueError('reported Dice does not follow integer counts')
        if not isinstance(row['automatic_point_hits_gt'], bool):
            raise ValueError('Point must be a boolean per scan')


def aggregate(records):
    patients = sorted({row['subject_id'] for row in records})
    values = np.asarray([row['locator_dsc'] for row in records], dtype=np.float64)
    groups = [np.asarray([i for i, row in enumerate(records) if row['subject_id'] == patient])
              for patient in patients]
    sums = np.asarray([values[group].sum() for group in groups])
    counts = np.asarray([len(group) for group in groups])
    means = sums / counts
    take = np.random.default_rng(20261029).integers(len(patients), size=(10000, len(patients)))
    scan = sums[take].sum(axis=1) / counts[take].sum(axis=1)
    subject = means[take].mean(axis=1)
    return dict(scans=len(records), subjects=len(patients), scan_dice=float(values.mean()),
                subject_dice=float(means.mean()), point_hits=sum(row['automatic_point_hits_gt'] for row in records),
                scan_ci95=np.quantile(scan, [.025, .975]).tolist(),
                subject_ci95=np.quantile(subject, [.025, .975]).tolist())


def paired(records, reference):
    if [(r['case_id'], r['subject_id']) for r in records] != [
            (r['case_id'], r['subject_id']) for r in reference]:
        raise ValueError('unpaired methods')
    delta = np.asarray([r['locator_dsc'] - b['locator_dsc'] for r, b in zip(records, reference)])
    subjects = sorted({r['subject_id'] for r in records})
    groups = [np.asarray([i for i, r in enumerate(records) if r['subject_id'] == subject]) for subject in subjects]
    counts = np.asarray([len(g) for g in groups])
    sums = np.asarray([delta[g].sum() for g in groups])
    means = sums / counts
    take = np.random.default_rng(20261029).integers(len(groups), size=(10000, len(groups)))
    worse = [dict(case_id=r['case_id'], subject_id=r['subject_id'], delta=float(d))
             for r, d in zip(records, delta) if d < 0]
    return dict(scan_delta=float(delta.mean()), subject_delta=float(means.mean()),
                scan_ci95=np.quantile(sums[take].sum(1) / counts[take].sum(1), [.025, .975]).tolist(),
                subject_ci95=np.quantile(means[take].mean(1), [.025, .975]).tolist(),
                point_delta=sum(r['automatic_point_hits_gt'] - b['automatic_point_hits_gt']
                                for r, b in zip(records, reference)),
                improved=int((delta > 0).sum()), worsened=int((delta < 0).sum()), equal=int((delta == 0).sum()),
                all_worsened=sorted(worse, key=lambda row: row['delta']))


def main(paths):
    manifest = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    assert len(manifest) == 108 and len({row['subject_id'] for row in manifest}) == 50
    methods = {}
    for path in paths:
        data = json.loads(path.read_text())
        for name, records in data['per_configuration'].items():
            validate(records, manifest)
            methods[f'{path.parent.name}:{name}'] = records
    if not methods:
        raise ValueError('no method results')
    summary = {name: aggregate(records) for name, records in methods.items()}
    result = dict(status='passed', cohort=dict(scans=108, subjects=50), input_sha256={str(p): sha(p) for p in paths},
                  summaries=summary, ranking=sorted(summary, key=lambda name: (-summary[name]['scan_dice'], name)),
                  interpretation='independent arithmetic audit; reused validation, descriptive after selection',
                  neural_forwards=0, new_pixel_access=False)
    atomic_json(RECORD / 'VALIDATION_ARITHMETIC_AUDIT.json', result)
    print(json.dumps({'status': result['status'], 'configurations': len(methods)}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', type=Path)
    main(parser.parse_args().reports)
