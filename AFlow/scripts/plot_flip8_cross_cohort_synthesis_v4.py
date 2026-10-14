#!/usr/bin/env python3
"""V4 fixed-Flip8-vs-identity cross-cohort synthesis (adds UCSD-BraTS2024test56; saved reports only, no new inference)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "internal": (ROOT / "archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/RAW_TEST270.json", None),
    "africa": (ROOT / "archive/research_records/2026-09-06_brats_africa95_frozen_baselines/RAW_EXTERNAL95.json", "c9fb07be4412c160129037104936e4c34b6390fa733b998b8a8f35524ccd30b3"),
    "cptac": (ROOT / "archive/research_records/2026-09-06_cptac33_africa_informed_confirmation/RAW_CPTAC33.json", "4b9d8c90e1b5ec45b0e664e560301db20affd6a96387f416a62b561f8d57b4fd"),
    "waw": (ROOT / "archive/research_records/2026-09-06_waw_gliomai152_frozen_panel/RAW_WAW152.json", "ef35301a78f6e47ac0574e5727aa7c5015b2f30cd73e1c8c45487fa8591dde91"),
    "waw_postop": (ROOT / "archive/research_records/2026-09-06_waw_postop_only31_frozen_panel/RAW_WAW_POSTOP31.json", "72269a5a872389db9d6dd316d7d46b614bad8d81796b0bb72625814907d98a98"),
    "brats24test": (ROOT / "archive/research_records/2026-09-06_ucsd_brats2024test_frozen_panel/RAW_UCSD_BRATS2024TEST56.json", "74466d1cdc56d864d48e5d8604e00d004ecdb636882366546b4738a1b794201c"),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def external_row(name, scope, report, denominator):
    comparison = report['comparisons']['full_flip8_minus_identity']
    return {'cohort': name, 'scope': scope,
            'delta': comparison['flip8_minus_identity']['scan_dice'],
            'delta_ci': comparison['paired_scan_dice_95ci'],
            'point_delta': comparison['flip8_minus_identity']['point_hits'],
            'point_denominator': denominator,
            'cases_improved': comparison['cases_improved'], 'cases_worse': comparison['cases_worse']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', type=Path, required=True)
    ap.add_argument('--png', type=Path, required=True)
    ap.add_argument('--pdf', type=Path, required=True)
    a = ap.parse_args()
    if any(x.exists() for x in (a.json, a.png, a.pdf)):
        raise FileExistsError('output')
    if a.png.parent != a.json.parent or a.pdf.parent != a.json.parent:
        raise ValueError('outputs must share one directory')
    for path, expected in SOURCES.values():
        if expected is not None and sha(path) != expected:
            raise ValueError('source SHA ' + str(path))
    internal = json.load(open(SOURCES['internal'][0]))
    rows = [{'cohort': 'Reused internal test270', 'scope': 'reused, not blind/external',
             'delta': internal['flip8_minus_identity']['flip8_minus_identity']['scan_dice'],
             'delta_ci': internal['flip8_minus_identity']['scan_dice_95ci'],
             'point_delta': internal['point_hits']['delta'], 'point_denominator': 270,
             'cases_improved': internal['flip8_minus_identity']['scan_improved'],
             'cases_worse': internal['flip8_minus_identity']['scan_worse']}]
    rows.append(external_row('BraTS-Africa95', 'public post-challenge; institution-disjoint', json.load(open(SOURCES['africa'][0])), 95))
    rows.append(external_row('CPTAC33', 'public Training labels; external to project', json.load(open(SOURCES['cptac'][0])), 33))
    rows.append(external_row('WAW-pre152', 'public preoperative; institution-disjoint', json.load(open(SOURCES['waw'][0])), 152))
    rows.append(external_row('WAW-post31', 'public early postop; patient-disjoint from preop', json.load(open(SOURCES['waw_postop'][0])), 31))
    rows.append(external_row('UCSD-BraTS2024test56', 'public post-challenge BraTS2024 test; patient-disjoint, NOT institution-disjoint', json.load(open(SOURCES['brats24test'][0])), 56))
    fully_positive = [r['cohort'] for r in rows if r['delta_ci'][0] > 0]
    result = {'schema': 'aflow.flip8.cross_cohort_synthesis', 'schema_version': 4,
              'status': 'descriptive_separate_cohorts_no_pooled_ranking',
              'checkpoint_sha256': 'f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b',
              'cohorts': rows,
              'interpretation': 'Fixed Flip8 minus identity Scan point estimates are positive on all six reported cohorts; paired CIs are entirely positive on internal270/CPTAC33/WAW-pre152 and cross zero on Africa95/WAW-post31/UCSD-BraTS2024test56. Point deltas are never negative. Deployment stays 8x; no view/weight/threshold search is authorized and cohorts must not be pooled.',
              'fully_positive_ci_cohorts': fully_positive,
              'source_sha256': {name: (expected if expected else sha(path)) for name, (path, expected) in SOURCES.items()},
              'new_model_inference': False, 'analyzer_sha256': sha(Path(__file__))}
    a.json.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    labels = [r['cohort'].replace(' ', '\n', 1) for r in rows]
    x = np.arange(len(rows))
    delta = np.array([r['delta'] for r in rows])
    cis = np.array([r['delta_ci'] for r in rows])
    fig, ax = plt.subplots(figsize=(12, 5.2), constrained_layout=True)
    ax.axhline(0, color='black', lw=1)
    ax.errorbar(x, delta, yerr=np.vstack((delta - cis[:, 0], cis[:, 1] - delta)), fmt='o', color='#0571b0', capsize=5)
    for i, r in enumerate(rows):
        ax.annotate(f"Point {r['point_delta']:+d}", (x[i], cis[i, 1]), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=9)
    ax.set(ylabel='Flip8 − identity Scan Dice', title='Fixed Flip8 vs identity across six reported cohorts (V4; separate cohorts, 8x cost)', xticks=x, xticklabels=labels)
    fig.savefig(a.png, dpi=160)
    fig.savefig(a.pdf)
    print(json.dumps({'cohorts': [{k: r[k] for k in ('cohort', 'delta', 'delta_ci', 'point_delta')} for r in rows],
                      'fully_positive_ci_cohorts': fully_positive,
                      'json_sha256': sha(a.json), 'png_sha256': sha(a.png), 'pdf_sha256': sha(a.pdf)}, indent=2))


if __name__ == '__main__':
    main()
