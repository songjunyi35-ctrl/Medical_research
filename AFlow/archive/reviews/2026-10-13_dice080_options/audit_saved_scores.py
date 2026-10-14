"""Describe two saved test reports; no images, weights, or new predictions."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCES = {
    'leader': 'archive/research_records/2026-09-21_t1ce_test270_10h/reports/panel_10.json',
    'multiscale': 'archive/research_records/2026-09-29_segresnet_multiscale_e166_test270/RAW_TEST270.json',
}


def main():
    saved = {k: json.loads((ROOT / p).read_text()) for k, p in SOURCES.items()}
    rows = saved['leader']['per_case']
    other = saved['multiscale']['summary']['per_case']
    by_id = {r['case_id']: r for r in other}
    assert len(rows) == len(other) == len(by_id) == 270
    assert len({r['case_id'] for r in rows}) == 270
    assert set(by_id) == {r['case_id'] for r in rows}
    assert all(r['subject_id'] == by_id[r['case_id']]['subject_id'] for r in rows)
    assert len({r['subject_id'] for r in rows}) == 119
    a = np.array([r['locator_dsc'] for r in rows])
    b = np.array([by_id[r['case_id']]['locator_dsc'] for r in rows])
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert ((a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)).all()
    assert abs(a.mean() - saved['leader']['metrics']['scan_dice']) < 1e-12
    assert abs(b.mean() - saved['multiscale']['summary']['locator_dsc_mean']) < 1e-12
    low = np.argsort(a, kind='stable')[:20]
    result = {
        'recorded_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'status': 'saved_report_descriptive_audit_complete',
        'test_tuned_diagnostic': True,
        'new_patient_pixel_access': False,
        'new_model_forwards': 0,
        'training_updates': 0,
        'new_outer_confirmation_access': False,
        'new_method_evaluated': False,
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES.values()},
        'scans': 270, 'subjects': 119, 'grid': 'native 1mm crop, padding excluded',
        'leader_scan_dice': float(a.mean()),
        'gap_to_080': float(.8 - a.mean()),
        'sum_case_dice_gain_needed': float(270 * (.8 - a.mean())),
        'leader_below_050': int((a < .5).sum()),
        'leader_below_020': int((a < .2).sum()),
        'leader_lowest20_mean': float(a[low].mean()),
        'leader_lowest20_subjects': len({rows[i]['subject_id'] for i in low}),
        'leader_lowest20_share_of_total_dice_deficit': float((1-a[low]).sum() / (1-a).sum()),
        'multiscale_mean_on_leader_lowest20': float(b[low].mean()),
        'multiscale_improved_on_leader_lowest20': int((b[low] > a[low]).sum()),
        'paired_case_score_pearson': float(np.corrcoef(a, b)[0, 1]),
        'multiscale_better_cases': int((b > a).sum()),
        'multiscale_better_by_more_than_005': int((b > a+.05).sum()),
        'multiscale_worse_by_more_than_005': int((b < a-.05).sum()),
        'leader_mean_fnr': float(np.mean([r['fnr'] for r in rows])),
        'leader_mean_fdr': float(np.mean([r['fdr'] for r in rows])),
        'interpretation': 'Case-score association does not measure voxel disagreement, predict ensemble Dice, prove causes, or bound achievable accuracy. Low-score grouping is post hoc. No per-case GT routing or synthetic method score computed.',
    }
    (OUT / 'SAVED_REPORT_AUDIT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
