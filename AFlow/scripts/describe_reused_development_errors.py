"""Offline count-only error description; never load images, labels or checkpoints."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from scripts import structured_background_protocol as p

RECORD = p.ROOT / 'archive/reviews/2026-09-07_development_error_description'
STUDIES = {'auxiliary': p.RECORD,
           'subject_sampling': p.ROOT / 'archive/research_records/2026-09-07_subject_balanced_sampling'}
CUTS = (18874, 40435, 80247)


def summarize_counts(rows):
    if not rows:
        return {'scans': 0, 'subjects': 0, 'metrics': None}
    subjects = defaultdict(list)
    missed, extra = [], []
    for row in rows:
        inter, pred, target = (row[k] for k in ('intersection', 'predicted_voxels', 'target_voxels'))
        if not (target > 0 and pred >= 0 and 0 <= inter <= min(pred, target)):
            raise ValueError('invalid saved overlap counts')
        dice = 2 * inter / (pred + target)
        if abs(dice - row['dice']) > 1e-12:
            raise ValueError('saved Dice does not match counts')
        subjects[row['subject_id']].append(dice)
        missed.append((target - inter) / target)
        extra.append((pred - inter) / target)
    return {'scans': len(rows), 'subjects': len(subjects),
            'metrics': {'scan_dice': statistics.fmean(r['dice'] for r in rows),
                        'subject_dice': statistics.fmean(statistics.fmean(v) for v in subjects.values()),
                        'point_hits': sum(r['point_hit'] for r in rows),
                        'empty_predictions': sum(r['predicted_voxels'] == 0 for r in rows),
                        'mean_missed_target_fraction': statistics.fmean(missed),
                        'mean_extra_prediction_per_target': statistics.fmean(extra)}}


def describe_pair(control, candidate, volumes):
    if len({r['case_id'] for r in control}) != len(control):
        raise ValueError('duplicate cases')
    pairs = list(zip(control, candidate, strict=True))
    for a, b in pairs:
        if any(a[k] != b[k] for k in ('case_id', 'subject_id', 'target_voxels')):
            raise ValueError('pair identities or targets differ')
        if volumes[a['case_id']] <= 0:
            raise ValueError('invalid native volume')
    selectors = {'all': pairs}
    for index in range(4):
        selectors[f'native_volume_group{index + 1}'] = [
            (a, b) for a, b in pairs
            if sum(volumes[a['case_id']] > cut for cut in CUTS) == index]
    result = {}
    for name, selected in selectors.items():
        left = summarize_counts([a for a, _ in selected])
        right = summarize_counts([b for _, b in selected])
        result[name] = {'control': left, 'candidate': right,
                        'candidate_minus_control': None if not selected else {
                            k: right['metrics'][k] - left['metrics'][k] for k in left['metrics']},
                        'dice_changes': {'improved': sum(b['dice'] > a['dice'] for a, b in selected),
                                         'worse': sum(b['dice'] < a['dice'] for a, b in selected),
                                         'tied': sum(b['dice'] == a['dice'] for a, b in selected)}}
    assert sum(result[f'native_volume_group{i}']['control']['scans'] for i in range(1, 5)) == len(control)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--study', choices=STUDIES)
    args = parser.parse_args()
    if args.freeze:
        assert args.study is None
        assert all(not (r / 'DEV_PREACCESS.json').exists() for r in STUDIES.values())
        p.write_new(RECORD / 'DESIGN_FREEZE.json', {
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'plan_sha256': p.sha(RECORD / 'PLAN.md'),
            'script_sha256': p.sha(Path(__file__)),
            'development_preaccess_absent_both_studies': True,
            'native_volume_cuts': CUTS})
        print('count-only diagnostic design frozen before both development readouts')
        return
    if args.study is None:
        parser.error('--study or --freeze required')
    frozen = json.loads((RECORD / 'DESIGN_FREEZE.json').read_text())
    p.checked(RECORD / 'PLAN.md', frozen['plan_sha256'])
    p.checked(Path(__file__), frozen['script_sha256'])
    source = STUDIES[args.study]
    verification = json.loads((source / 'VERIFICATION_RECEIPT.json').read_text())
    assert verification['status'] == 'passed'
    p.checked(source / 'RESULT.json', verification['result_sha256'])
    raw_path = source / 'RAW_DEVELOPMENT.json'
    p.checked(raw_path, verification['raw_sha256'])
    raw = json.loads(raw_path.read_text())
    _, dev = p.check_roles()
    expected = [(r['case_id'], r['subject_id']) for r in dev]
    p.checked(p.NATIVE, p.NATIVE_SHA)
    full_volumes = json.loads(p.NATIVE.read_text())['native_wt_voxels']
    volumes = {case: full_volumes[case] for case, _ in expected}
    results = {}
    for seed in p.SEEDS:
        if args.study == 'auxiliary':
            left = raw['results'][f'seed{seed}_control']['per_case']
            right = raw['results'][f'seed{seed}_candidate']['per_case']
        else:
            left = raw['results'][str(seed)]['reused_control']['per_case']
            right = raw['results'][str(seed)]['candidate']['per_case']
        assert [(r['case_id'], r['subject_id']) for r in left] == expected
        results[str(seed)] = describe_pair(left, right, volumes)
    output = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'descriptive_not_a_promotion_gate', 'study': args.study,
              'results': results, 'native_volume_cuts': CUTS,
              'source_raw_sha256': p.sha(raw_path), 'native_sha256': p.NATIVE_SHA,
              'design_freeze_sha256': p.sha(RECORD / 'DESIGN_FREEZE.json'),
              'access': {'saved_reused_development_counts': True, 'new_forwards': 0,
                         'images_or_label_arrays': False, 'test_outer_confirmation_external': False}}
    p.write_new(RECORD / f'{args.study.upper()}.json', output)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
