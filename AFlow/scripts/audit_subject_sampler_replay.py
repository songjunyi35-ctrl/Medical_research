"""Rebuild logged effective sampling indices using fit metadata, without models."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from scripts import subject_balanced_protocol as p

RECORD = p.ROOT / 'archive/reviews/2026-09-07_fit_subject_sampling'


def independent_weights(case_ids, subject_map, volumes):
    values = np.asarray([volumes[c] for c in case_ids], dtype=np.float64)
    if np.any(values <= 0):
        raise ValueError('nonpositive volume')
    native = np.clip(np.sqrt(np.median(values) / values), .5, 4.)
    totals = defaultdict(float)
    for case, weight in zip(case_ids, native.tolist(), strict=True):
        totals[subject_map[case]] += weight
    balanced = np.asarray([w / totals[subject_map[c]] for c, w in zip(case_ids, native.tolist(), strict=True)], dtype=np.float64)
    return {'control': torch.from_numpy(native), 'subject_balanced': torch.from_numpy(balanced)}


def replay(weights, seed, epochs, case_ids, subject_map, volumes):
    generator = torch.Generator().manual_seed(seed)
    output = []
    for epoch in range(1, epochs + 1):
        effective = torch.multinomial(weights, 972, replacement=True, generator=generator).numpy()[:968]
        cases = [case_ids[i] for i in effective]
        output.append({'epoch': epoch,
                       'sampler_sha256': hashlib.sha256(effective.astype(np.int64).tobytes()).hexdigest(),
                       'unique_scans': len(set(cases)),
                       'unique_subjects': len({subject_map[c] for c in cases}),
                       'small_volume_draws': sum(volumes[c] <= 18874 for c in cases)})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    out = RECORD / ('SAMPLER_REPLAY_FINAL.json' if args.final else 'SAMPLER_REPLAY_SNAPSHOT.json')
    if out.exists():
        raise FileExistsError(out)
    p.check_freeze()
    p.checked(p.FIT, p.FIT_SHA)
    p.checked(p.NATIVE, p.NATIVE_SHA)
    fit = p.rows(p.FIT)
    cases = [r['case_id'] for r in fit]
    subjects = {r['case_id']: r['subject_id'] for r in fit}
    volumes = json.loads(p.NATIVE.read_text())['native_wt_voxels']
    weights = independent_weights(cases, subjects, volumes)
    results = {}
    source_hashes = {}
    for seed in p.SEEDS:
        candidate_path = p.WORK / f'seed{seed}_subject_balanced' / ('TRAIN_DONE.json' if args.final else 'progress.json')
        control_path = p.SOURCE_WORK / f'seed{seed}_control/TRAIN_DONE.json'
        loaded = {}
        for arm, path in [('subject_balanced', candidate_path), ('control', control_path)]:
            body = path.read_bytes()
            loaded[arm] = json.loads(body)
            source_hashes[str(path.relative_to(p.ROOT))] = hashlib.sha256(body).hexdigest()
        epochs = len(loaded['subject_balanced']['history'])
        assert 0 < epochs <= 168
        if args.final:
            assert epochs == 168 and loaded['subject_balanced']['all_gates_passed']
        paired = {}
        for arm, weight in weights.items():
            expected = replay(weight, seed, epochs, cases, subjects, volumes)
            observed = loaded[arm]['history'][:epochs]
            assert len(observed) == epochs
            assert all(x['epoch'] == y['epoch'] and x['sampler_sha256'] == y['sampler_sha256']
                       for x, y in zip(expected, observed, strict=True)), (seed, arm)
            digest = hashlib.sha256(weight.numpy().tobytes()).hexdigest()
            if arm == 'control' or args.final:
                assert digest == loaded[arm]['sampling_weight_sha256']
            paired[arm] = {'epochs_verified': epochs, 'all_effective_index_hashes_exact': True,
                           'weight_sha256': digest,
                           'mean_unique_subjects_per_epoch': statistics.fmean(x['unique_subjects'] for x in expected),
                           'mean_unique_scans_per_epoch': statistics.fmean(x['unique_scans'] for x in expected),
                           'small_volume_fraction_of_effective_draws': sum(x['small_volume_draws'] for x in expected) / (epochs * 968),
                           'per_epoch': expected}
        results[str(seed)] = paired
    result = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'passed_full168' if args.final else 'passed_completed_epoch_snapshot_only',
              'source_snapshot_sha256': source_hashes, 'results': results,
              'fit_sha256': p.FIT_SHA, 'native_sha256': p.NATIVE_SHA,
              'script_sha256': p.sha(Path(__file__)), 'plan_sha256': p.sha(RECORD / 'REPLAY_PLAN.md'),
              'access': {'fit_metadata_and_training_logs': True, 'images_labels_checkpoints': False,
                         'development_metrics': False, 'new_forwards': 0},
              'interpretation': 'Sampling mechanism only; training Dice uses different sampled populations, not a paired accuracy comparison.'}
    p.write_new(out, result)
    print(json.dumps({'status': result['status'], 'results': {s: {a: {k: v for k, v in d.items() if k != 'per_epoch'} for a, d in arms.items()} for s, arms in results.items()}}, indent=2))


if __name__ == '__main__':
    main()
