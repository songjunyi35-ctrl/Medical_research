#!/usr/bin/env python3
"""V5 cross-cohort synthesis of saved 1x Full972-vs-GCSAM evidence (adds UCSD-BraTS2024test56; no new inference)."""
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
    "internal": (ROOT / "workspace/BraTS/sammed3d_t1ce_wt/gcsam_full972_stability_20260831/GCSAM_INNER_E184_TEST270.json", "d0e2f48e1f499d65adddb3dbe30758caad27768bb75f737ea2cfe4a877c335d5"),
    "ucsd": (ROOT / "archive/research_records/2026-09-06_ucsd_official_float_source_correction/RAW_OFFICIAL_FLOAT.json", "a1155914b16b648f97f11a7d7a20cb27096d7eed9f9662db73f4905860da04d2"),
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


def panel_row(name, scope, report, points_full, points_gcsam, denominator):
    return {'cohort': name, 'scope': scope,
            'full_scan': report['full972_kd168']['identity']['locator_dsc_mean'],
            'full_ci': report['full972_kd168']['identity_bootstrap']['scan_dsc_mean_95ci'],
            'full_points': points_full, 'point_denominator': denominator,
            'gcsam_scan': report['gcsam184']['identity']['locator_dsc_mean'],
            'gcsam_ci': report['gcsam184']['identity_bootstrap']['scan_dsc_mean_95ci'],
            'gcsam_points': points_gcsam,
            'delta': report['comparisons']['gcsam_identity_minus_full_identity']['gcsam_minus_full']['scan_dice'],
            'delta_ci': report['comparisons']['gcsam_identity_minus_full_identity']['paired_scan_dice_95ci']}


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
    full_report = ROOT / 'workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824/FULLTRAJECTORY_TEST270.json'
    if any(sha(path) != expected for path, expected in SOURCES.values()) or sha(full_report) != '9827a75b88dc9239902c7e473a7d75e2d6058f4f515bfd2b4d93003ba7bd34cd':
        raise ValueError('source SHA')
    internal = json.load(open(SOURCES['internal'][0]))
    ucsd = json.load(open(SOURCES['ucsd'][0]))
    africa = json.load(open(SOURCES['africa'][0]))
    cptac = json.load(open(SOURCES['cptac'][0]))
    waw = json.load(open(SOURCES['waw'][0]))
    waw_postop = json.load(open(SOURCES['waw_postop'][0]))
    b24 = json.load(open(SOURCES['brats24test'][0]))
    rows = [
        {'cohort': 'Reused internal test270', 'scope': '270 scans/119 subjects; reused, not blind/external',
         'full_scan': internal['current_internal_best']['locator_dsc_mean'], 'full_ci': json.load(open(full_report))['candidate_bootstrap']['scan_dsc_mean_95ci'], 'full_points': 262, 'point_denominator': 270,
         'gcsam_scan': internal['gcsam']['locator_dsc_mean'], 'gcsam_ci': internal['gcsam_bootstrap']['scan_dsc_mean_95ci'], 'gcsam_points': 252,
         'delta': internal['gcsam_minus_current_internal_best']['ema_minus_raw']['scan'], 'delta_ci': internal['gcsam_minus_current_internal_best']['ema_minus_raw']['scan_95ci']},
        {'cohort': 'UCSD75 official-float', 'scope': 'same-cohort source correction; reused public external',
         'full_scan': ucsd['full972_kd168']['summary']['locator_dsc_mean'], 'full_ci': ucsd['full972_kd168']['bootstrap']['scan_dsc_mean_95ci'], 'full_points': 70, 'point_denominator': 75,
         'gcsam_scan': ucsd['gcsam184']['summary']['locator_dsc_mean'], 'gcsam_ci': ucsd['gcsam184']['bootstrap']['scan_dsc_mean_95ci'], 'gcsam_points': 70,
         'delta': -ucsd['official_float_model_comparison_full972_minus_gcsam']['primary_minus_secondary']['scan_dsc_mean'],
         'delta_ci': [-ucsd['official_float_model_comparison_full972_minus_gcsam']['paired_scan_dsc_mean_95ci'][1], -ucsd['official_float_model_comparison_full972_minus_gcsam']['paired_scan_dsc_mean_95ci'][0]]},
        panel_row('BraTS-Africa95', 'public post-challenge; four centers; institution-disjoint/no patient bridge', africa, 87, 84, 95),
        panel_row('CPTAC33 public Training', 'public labels external to this project; not hidden', cptac, 33, 33, 33),
        panel_row('WAW-GliomAI152', 'public preoperative; institution-disjoint/no patient bridge', waw, 141, 140, 152),
        panel_row('WAW early-postop31', 'public day0-22 earliest scan; patients absent from preop152', waw_postop, 15, 14, 31),
        panel_row('UCSD-BraTS2024test56', 'public post-challenge BraTS-GLI 2024 test; patient-disjoint by official mapping; NOT institution-disjoint', b24, 56, 52, 56),
    ]
    result = {'schema': 'aflow.cross_cohort.frozen_checkpoint_synthesis', 'schema_version': 5,
              'status': 'descriptive_separate_cohorts_no_pooled_ranking',
              'models': {'full972_sha256': 'f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b',
                         'gcsam184_sha256': 'acf2fce112370e3cdb58fe295642680d5db1db3ceb619defdc47824ef4016d0d'},
              'cohorts': rows,
              'interpretation': 'GCSAM-minus-Full Scan Dice now spans both确定 directions: Africa95 interval entirely above zero, UCSD-BraTS2024test56 interval entirely below zero (-0.012312, [-0.025822,-0.000092]), and internal270/UCSD75/CPTAC33/WAW-pre152/WAW-post31 all cross zero. Point differences are -10/0/-3/0/-1/-1/-4. The direction is cohort-reversible; any "GCSAM generally higher externally" wording is retired. Cohorts must not be pooled and no checkpoint/threshold selection may flow back.',
              'new_model_inference': False, 'selection_or_tuning': False, 'analyzer_sha256': sha(Path(__file__))}
    a.json.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    labels = [r['cohort'].replace(' ', '\n', 1) for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 9), constrained_layout=True)
    for series, color, key, ci_key in (('Full972 KD168 identity', '#1b7837', 'full_scan', 'full_ci'), ('GCSAM184 identity', '#d95f02', 'gcsam_scan', 'gcsam_ci')):
        vals = np.array([r[key] for r in rows])
        cis = np.array([r[ci_key] for r in rows])
        offset = -0.12 if key == 'full_scan' else 0.12
        axes[0].errorbar(x + offset, vals, yerr=np.vstack((vals - cis[:, 0], cis[:, 1] - vals)), fmt='o', color=color, capsize=4, label=series)
    axes[0].set(ylabel='Scan Dice', title='Absolute scores with 95% CI (separate cohorts; do not pool)', xticks=x, xticklabels=labels, ylim=(0.45, 0.89))
    axes[0].legend(frameon=False)
    delta = np.array([r['delta'] for r in rows])
    cis = np.array([r['delta_ci'] for r in rows])
    axes[1].axhline(0, color='black', lw=1)
    axes[1].errorbar(x, delta, yerr=np.vstack((delta - cis[:, 0], cis[:, 1] - delta)), fmt='o', color='#7b3294', capsize=5)
    axes[1].set(ylabel='GCSAM − Full Scan Dice', title='Within-cohort paired differences (V5: direction is cohort-reversible)', xticks=x, xticklabels=labels)
    fig.savefig(a.png, dpi=160)
    fig.savefig(a.pdf)
    print(json.dumps({'cohorts': [{k: r[k] for k in ('cohort', 'delta', 'delta_ci')} for r in rows],
                      'json_sha256': sha(a.json), 'png_sha256': sha(a.png), 'pdf_sha256': sha(a.pdf)}, indent=2))


if __name__ == '__main__':
    main()
