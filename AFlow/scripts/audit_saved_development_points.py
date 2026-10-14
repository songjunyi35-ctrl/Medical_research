"""Verify saved point membership and target counts against authorized dev labels."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from scripts import structured_background_protocol as p

RECORD = p.ROOT / 'archive/reviews/2026-09-07_development_error_description'
STUDIES = {'auxiliary': p.RECORD,
           'subject_sampling': p.ROOT / 'archive/research_records/2026-09-07_subject_balanced_sampling'}


def check_point(row, mask):
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError('nonbinary WT cache')
    count = int(np.count_nonzero(mask))
    if count <= 0:
        raise ValueError('empty target')
    point = row['point_flat_index']
    if isinstance(point, bool) or not isinstance(point, (int, np.integer)) or not 0 <= point < mask.size:
        raise ValueError('invalid point index')
    hit = bool(mask.reshape(-1, order='C')[point])
    return {'target_count_exact': count == row['target_voxels'],
            'point_membership_exact': hit == row['point_hit'],
            'recomputed_target_voxels': count, 'recomputed_point_hit': hit}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', choices=STUDIES, required=True)
    args = parser.parse_args()
    out = RECORD / f'{args.study.upper()}_POINT_AUDIT.json'
    pre = RECORD / f'{args.study.upper()}_POINT_PREACCESS.json'
    if out.exists() or pre.exists():
        raise FileExistsError('audit already started; inspect saved boundary before retrying')
    source = STUDIES[args.study]
    verification = json.loads((source / 'VERIFICATION_RECEIPT.json').read_text())
    assert verification['status'] == 'passed'
    p.checked(source / 'RESULT.json', verification['result_sha256'])
    raw_path = source / 'RAW_DEVELOPMENT.json'
    p.checked(raw_path, verification['raw_sha256'])
    raw = json.loads(raw_path.read_text())
    _, dev = p.check_roles()
    expected = [(r['case_id'], r['subject_id']) for r in dev]
    p.checked(p.CACHE / 'index.json', p.INDEX_SHA)
    index = json.loads((p.CACHE / 'index.json').read_text())
    assert index['model_shape'] == [128, 128, 128] and index['label_dtype'] == 'uint8'
    label_path = p.CACHE / 'labels.u8'
    p.checked(label_path, index['labels_sha256'])
    shape = (index['count'], *index['model_shape'])
    assert label_path.stat().st_size == int(np.prod(shape))
    slots = {case: i for i, case in enumerate(index['case_ids'])}
    assert len(slots) == index['count']
    models = {}
    if args.study == 'auxiliary':
        models = {k: v['per_case'] for k, v in raw['results'].items()}
    else:
        for seed, result in raw['results'].items():
            for arm, model in result.items():
                models[f'seed{seed}_{arm}'] = model['per_case']
    assert len(models) == 4
    assert all([(r['case_id'], r['subject_id']) for r in rows] == expected for rows in models.values())
    p.write_new(pre, {'created_utc': datetime.now(timezone.utc).isoformat(),
                      'source_raw_sha256': p.sha(raw_path), 'dev_sha256': p.DEV_SHA,
                      'cache_index_sha256': p.INDEX_SHA, 'labels_sha256': index['labels_sha256'],
                      'script_sha256': p.sha(Path(__file__)),
                      'plan_sha256': p.sha(RECORD / 'POINT_AUDIT_PLAN.md'),
                      'label_arrays_accessed_at_write': 0, 'new_forwards': 0})
    labels = np.memmap(label_path, dtype=np.uint8, mode='r', shape=shape)
    checks = {key: [] for key in models}
    try:
        for ordinal, (case, subject) in enumerate(expected):
            mask = np.asarray(labels[slots[case]])
            for key, rows in models.items():
                checks[key].append({'case_id': case, 'subject_id': subject, **check_point(rows[ordinal], mask)})
        passed = all(row['target_count_exact'] and row['point_membership_exact']
                     for rows in checks.values() for row in rows)
        result = {'created_utc': datetime.now(timezone.utc).isoformat(),
                  'status': 'passed' if passed else 'failed_saved_counts_or_points',
                  'preaccess_sha256': p.sha(pre), 'study': args.study,
                  'unique_development_label_arrays_read': 180, 'saved_point_records_checked': 720,
                  'models': {k: {'all_target_counts_exact': all(r['target_count_exact'] for r in rows),
                                  'all_point_membership_exact': all(r['point_membership_exact'] for r in rows),
                                  'recomputed_point_hits': sum(r['recomputed_point_hit'] for r in rows)}
                             for k, rows in checks.items()},
                  'per_case_checks': checks,
                  'access': {'reused_development_label_arrays': True, 'new_model_forwards': 0,
                             'images_or_checkpoints': False, 'test_outer_confirmation_external': False},
                  'limitation': 'Does not independently verify logit argmax or predicted-mask overlap.'}
        p.write_new(out, result)
        print(json.dumps({k: v for k, v in result.items() if k != 'per_case_checks'}, indent=2))
        if not passed:
            raise ValueError('saved count/point mismatch; no automatic model retry')
    except BaseException as error:
        if not out.exists():
            p.write_new(out, {'status': 'failed_after_label_preaccess', 'error': repr(error),
                              'preaccess_sha256': p.sha(pre), 'new_forwards': 0})
        raise


if __name__ == '__main__':
    main()
