"""Summarize the two closed development studies; no new models or selection."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

import numpy as np

from scripts import structured_background_protocol as a
from scripts import subject_balanced_protocol as b
from scripts.audit_subject_sampler_replay import independent_weights

RECORD = a.ROOT / 'archive/research_records/2026-09-07_reused_development_program'


def weighting_reference(case_ids, subject_map, volumes):
    """Analytic comparison only: never supplied to any training entry point."""
    counts = Counter(subject_map.values())
    reference = np.asarray([1 / (len(counts) * counts[subject_map[c]]) for c in case_ids])
    policies = independent_weights(case_ids, subject_map, volumes)
    result = {}
    for name, tensor in policies.items():
        prob = tensor.numpy() / float(tensor.sum())
        result[name] = {'total_variation_from_patient_and_scan_uniform': float(np.abs(prob - reference).sum() / 2),
                        'probability_relative_to_reference_min': float((prob / reference).min()),
                        'probability_relative_to_reference_max': float((prob / reference).max())}
    return {'fit_scans': len(case_ids), 'fit_subjects': len(counts), 'policies': result,
            'reference_only_not_a_new_training_policy': True,
            'interpretation': 'Patient marginal is equal; within-patient native weights remain nonuniform. Training soft loss also differs from thresholded evaluation Dice.'}


def main():
    out = RECORD / 'RESULT.json'
    if out.exists():
        raise FileExistsError(out)
    a.check_freeze()
    b.check_freeze()
    studies = {}
    models = []
    training = []
    for name, p in [('auxiliary', a), ('subject_sampling', b)]:
        seal = json.loads((p.RECORD / 'ARCHIVE_SEAL.json').read_text())
        for path, digest in seal['files'].items():
            p.checked(p.ROOT / path, digest)
        result = json.loads((p.RECORD / 'RESULT.json').read_text())
        raw = json.loads((p.RECORD / 'RAW_DEVELOPMENT.json').read_text())
        studies[name] = {'status': result['status'], 'joint_gate_passed': result['all_gates_passed'],
                         'comparisons': result['comparisons'],
                         'average_scan_delta': statistics.fmean(c['scan_delta'] for c in result['comparisons'].values()),
                         'average_subject_delta': statistics.fmean(c['subject_delta'] for c in result['comparisons'].values()),
                         'result_path': str((p.RECORD / 'RESULT.json').relative_to(p.ROOT)),
                         'result_sha256': p.sha(p.RECORD / 'RESULT.json'),
                         'raw_sha256': p.sha(p.RECORD / 'RAW_DEVELOPMENT.json'),
                         'new_forwards': raw['forwards'] if name == 'auxiliary' else raw['new_forwards'],
                         'evaluation_seconds': raw['seconds']}
        for seed in p.SEEDS:
            arms = ('control', 'candidate') if name == 'auxiliary' else ('subject_balanced',)
            for arm in arms:
                key = f'seed{seed}_{arm}'
                report_path = p.WORK / key / 'TRAIN_DONE.json'
                report = json.loads(report_path.read_text())
                metric = result['metrics'][key] if name == 'auxiliary' else result['metrics'][str(seed)]['candidate']
                models.append({'study': name, 'seed': seed, 'arm': arm, 'metrics': metric,
                               'checkpoint': report['checkpoint'], 'train_report_sha256': p.sha(report_path)})
                training.append({'study': name, 'seed': seed, 'arm': arm,
                                 'seconds': report['elapsed_seconds'],
                                 'peak_bytes': report['peak_gpu_memory_bytes'],
                                 'checkpoint_bytes': Path(report['checkpoint']['path']).stat().st_size,
                                 'recovery_bytes': (p.WORK / key / 'recovery_latest.pth').stat().st_size})
    fit, dev = a.check_roles()
    volumes = json.loads(a.NATIVE.read_text())['native_wt_voxels']
    alignment = weighting_reference([r['case_id'] for r in fit], {r['case_id']: r['subject_id'] for r in fit}, volumes)
    best_scan = max(models, key=lambda m: m['metrics']['scan_dice'])
    best_subject = max(models, key=lambda m: m['metrics']['subject_dice'])
    best_point = max(models, key=lambda m: m['metrics']['point_hits'])
    rc = json.loads((a.RECORD / 'AUX_CACHE_RECEIPT.json').read_text())
    costs = {'unique_training_runs': len(training),
             'training_gpu_hours': sum(r['seconds'] for r in training) / 3600,
             'peak_per_gpu_bytes': max(r['peak_bytes'] for r in training),
             'unique_checkpoint_bytes': sum(r['checkpoint_bytes'] for r in training),
             'recovery_bytes': sum(r['recovery_bytes'] for r in training),
             'shared_auxiliary_cache_bytes': Path(rc['rc_file']).stat().st_size,
             'formal_development_forwards': sum(r['new_forwards'] for r in studies.values()),
             'formal_evaluation_seconds': sum(r['evaluation_seconds'] for r in studies.values()),
             'deployment_forwards_per_model': 1}
    result = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'two_studies_complete_both_joint_gates_failed_segmentation_gain_retained',
              'cohort': {'fit_scans': len(fit), 'fit_subjects': 313, 'development_scans': len(dev),
                         'development_subjects': 78, 'role': 'user_authorized_reused_development'},
              'fit_sha256': a.FIT_SHA, 'development_sha256': a.DEV_SHA,
              'studies': studies, 'unique_models': models, 'training': training, 'costs': costs,
              'descriptive_maxima_not_checkpoint_selection': {
                  key: {'study': row['study'], 'seed': row['seed'], 'arm': row['arm'], 'metrics': row['metrics']}
                  for key, row in [('scan', best_scan), ('subject', best_subject), ('point', best_point)]},
              'weighting_alignment_reference': alignment,
              'access': {'saved_development_reports_and_fit_metadata': True, 'new_model_forwards': 0,
                         'new_protected_cohort_access': False},
              'selection': {'new_checkpoint_selected_for_test': False,
                            'failed_seed_dropped': False, 'point_gate_relaxed': False},
              'goal_duration_claim': 'not addressed by this scientific summary; verify active Goal separately',
              'script_sha256': a.sha(Path(__file__))}
    a.write_new(out, result)
    print(json.dumps({'status': result['status'], 'costs': costs, 'weighting_alignment_reference': alignment}, indent=2))


if __name__ == '__main__':
    main()
