"""Post-hoc saved-row description of point transitions, with no model access."""
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts import subject_balanced_protocol as p

RECORD = p.ROOT / 'archive/reviews/2026-09-07_development_error_description'


def transitions(control, candidate):
    categories = {'kept_hit': [], 'lost_hit': [], 'gained_hit': [], 'kept_miss': []}
    keys = {(True, True): 'kept_hit', (True, False): 'lost_hit',
            (False, True): 'gained_hit', (False, False): 'kept_miss'}
    for a, b in zip(control, candidate, strict=True):
        if any(a[k] != b[k] for k in ('case_id', 'subject_id', 'target_voxels')):
            raise ValueError('case/subject/target mismatch')
        key = keys[(bool(a['point_hit']), bool(b['point_hit']))]
        categories[key].append({'case_id': a['case_id'], 'subject_id': a['subject_id'],
                                'control_dice': a['dice'], 'candidate_dice': b['dice'],
                                'dice_delta': b['dice'] - a['dice'],
                                'target_voxels': a['target_voxels'],
                                'control_predicted_voxels': a['predicted_voxels'],
                                'candidate_predicted_voxels': b['predicted_voxels'],
                                'control_point_index': a['point_flat_index'],
                                'candidate_point_index': b['point_flat_index']})
    counts = {k: len(v) for k, v in categories.items()}
    assert sum(counts.values()) == len(control)
    lost = categories['lost_hit']
    return {'counts': counts, 'net_point_delta': counts['gained_hit'] - counts['lost_hit'],
            'lost_point_cases_dice_improved': sum(r['dice_delta'] > 0 for r in lost),
            'lost_point_cases_dice_worse': sum(r['dice_delta'] < 0 for r in lost),
            'lost_point_cases_dice_tied': sum(r['dice_delta'] == 0 for r in lost),
            'all_changed_cases': {k: categories[k] for k in ('lost_hit', 'gained_hit')}}


def main():
    out = RECORD / 'SUBJECT_SAMPLING_POINT_TRANSITIONS.json'
    if out.exists():
        raise FileExistsError(out)
    result = json.loads((p.RECORD / 'RESULT.json').read_text())
    raw_path = p.RECORD / 'RAW_DEVELOPMENT.json'
    p.checked(raw_path, result['raw_sha256'])
    raw = json.loads(raw_path.read_text())
    output = {}
    for seed in p.SEEDS:
        pair = raw['results'][str(seed)]
        value = transitions(pair['reused_control']['per_case'], pair['candidate']['per_case'])
        assert value['net_point_delta'] == raw['comparisons'][str(seed)]['point_delta']
        output[str(seed)] = value
    first, second = [output[str(seed)] for seed in p.SEEDS]
    overlap = {category: sorted({r['case_id'] for r in first['all_changed_cases'][category]} &
                               {r['case_id'] for r in second['all_changed_cases'][category]})
               for category in ('lost_hit', 'gained_hit')}
    p.write_new(out, {'created_utc': datetime.now(timezone.utc).isoformat(),
                     'status': 'posthoc_descriptive_only_joint_gate_unchanged',
                     'per_seed': output, 'same_direction_changed_case_intersection': overlap,
                     'source_raw_sha256': p.sha(raw_path),
                     'plan_sha256': p.sha(RECORD / 'POINT_TRANSITIONS_PLAN.md'),
                     'script_sha256': p.sha(Path(__file__)),
                     'access': {'saved_development_rows_only': True, 'new_forwards': 0,
                                'image_label_checkpoint_arrays': False},
                     'limitations': 'Net changes are not all changed cases; no logits/tie explanation, new CI, or promotion.'})
    print(json.dumps({'per_seed': output, 'overlap': overlap}, indent=2))


if __name__ == '__main__':
    main()
