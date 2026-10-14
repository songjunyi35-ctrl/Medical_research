"""Recheck two completed studies' receipt ordering and content bindings."""
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts import structured_background_protocol as source
from scripts import subject_balanced_protocol as sampler

ROOT = source.ROOT
RECORD = ROOT / 'archive/reviews/2026-09-07_30h_completion_audit'


def read(path):
    return json.loads(Path(path).read_text())


def timestamp(value):
    return datetime.fromisoformat(value)


def verify_study(protocol, arms):
    protocol.check_freeze()
    record = protocol.RECORD
    seal = read(record / 'ARCHIVE_SEAL.json')
    for name, digest in seal['files'].items():
        protocol.checked(ROOT / name, digest)
    pre_path = record / 'TRAIN_PREACCESS.json'
    pre = read(pre_path)
    protocol.checked(record / 'IMPLEMENTATION_FREEZE.json', pre['implementation_freeze_sha256'])
    protocol.checked(record / 'SMOKE_RECEIPT.json', pre['smoke_sha256'])
    protocol.checked(record / 'AUX_CACHE_RECEIPT.json', pre['aux_cache_receipt_sha256'])
    dev_path = record / 'DEV_PREACCESS.json'
    dev_pre = read(dev_path)
    raw = read(record / 'RAW_DEVELOPMENT.json')
    result = read(record / 'RESULT.json')
    verification = read(record / 'VERIFICATION_RECEIPT.json')
    chain = read(record / 'CHAIN_DONE.json')
    assert chain['status'] == 'completed' and verification['status'] == 'passed'
    protocol.checked(record / 'RESULT.json', chain['result_sha256'])
    protocol.checked(record / 'RESULT.json', verification['result_sha256'])
    protocol.checked(record / 'VERIFICATION_RECEIPT.json', chain['verification_sha256'])
    protocol.checked(record / 'RAW_DEVELOPMENT.json', result['raw_sha256'])
    protocol.checked(dev_path, raw['dev_preaccess_sha256'])
    assert timestamp(pre['created_utc']) < timestamp(dev_pre['created_utc']) < timestamp(result['created_utc'])
    training = {}
    for seed in protocol.SEEDS:
        for arm in arms:
            key = f'seed{seed}_{arm}'
            started_path = protocol.WORK / key / 'STARTED.json'
            report_path = protocol.WORK / key / 'TRAIN_DONE.json'
            started, done = read(started_path), read(report_path)
            protocol.checked(pre_path, started['train_preaccess_sha256'])
            assert timestamp(pre['created_utc']) <= timestamp(started['created_utc']) < timestamp(dev_pre['created_utc'])
            assert started['seed'] == done['seed'] == seed and started['arm'] == done['arm'] == arm
            assert started['fit_sha256'] == done['fit_sha256'] == protocol.FIT_SHA
            assert started['initial_state_sha256'] == done['initial_state_sha256']
            assert not started['teacher_checkpoints'] and not done['teacher_checkpoints']
            assert not started['development_metrics_accessed'] and not done['access']['development_metrics']
            assert all(not done['access'][k] for k in ('test', 'confirmation', 'outer', 'external'))
            assert done['all_gates_passed'] and done['fixed_epochs'] == len(done['history']) == 168
            assert all(row['steps'] == 121 for row in done['history']) and done['export_max_abs_error'] == 0
            protocol.checked(Path(done['checkpoint']['path']), done['checkpoint']['sha256'])
            if protocol is source:
                assert dev_pre['reports'][key]['train_report_sha256'] == protocol.sha(report_path)
                assert dev_pre['reports'][key]['checkpoint_sha256'] == done['checkpoint']['sha256']
            else:
                assert dev_pre['candidate_checkpoints'][str(seed)] == done['checkpoint']
                control = pre['source_controls'][str(seed)]
                protocol.checked(Path(control['train_report_path']), control['train_report_sha256'])
                prior = read(control['train_report_path'])
                assert prior['arm'] == 'control' and prior['seed'] == seed
                assert prior['initial_state_sha256'] == done['initial_state_sha256']
                assert dev_pre['source_controls'][str(seed)] == control
            training[key] = {'started_utc': started['created_utc'],
                             'started_sha256': protocol.sha(started_path),
                             'train_report_sha256': protocol.sha(report_path),
                             'checkpoint_sha256': done['checkpoint']['sha256'],
                             'same_initial_as_start': True, 'teacher_free': True,
                             'epochs': 168, 'export_error': 0}
    assert raw['access']['reused_development180'] is True
    assert all(not raw['access'][k] for k in ('confirmation', 'external', 'fixed_validation108', 'outer', 'test270'))
    return {'archive_seal_sha256': protocol.sha(record / 'ARCHIVE_SEAL.json'),
            'sealed_files_rehashed': len(seal['files']),
            'train_preaccess_utc': pre['created_utc'], 'development_preaccess_utc': dev_pre['created_utc'],
            'result_utc': result['created_utc'], 'training': training,
            'raw_sha256': protocol.sha(record / 'RAW_DEVELOPMENT.json'),
            'result_sha256': protocol.sha(record / 'RESULT.json'),
            'access': raw['access'], 'formal_joint_gate_passed': result['all_gates_passed']}


def main():
    out = RECORD / 'RECEIPT_LINKS.json'
    if out.exists():
        raise FileExistsError(out)
    studies = {'auxiliary': verify_study(source, source.ARMS),
               'subject_sampling': verify_study(sampler, sampler.ARMS)}
    train_pre = read(sampler.RECORD / 'TRAIN_PREACCESS.json')
    sampler.checked(source.RECORD / 'RAW_DEVELOPMENT.json', train_pre['source_raw_development_sha256'])
    source_raw = read(source.RECORD / 'RAW_DEVELOPMENT.json')
    sampler_raw = read(sampler.RECORD / 'RAW_DEVELOPMENT.json')
    for seed in source.SEEDS:
        assert sampler_raw['results'][str(seed)]['reused_control'] == source_raw['results'][f'seed{seed}_control']
    design = read(sampler.RECORD / 'DESIGN_FREEZE.json')
    assert not design['preceding_dev_preaccess_exists_at_check'] and not design['preceding_raw_development_exists_at_check']
    assert design['conditional_on_prior_performance'] is False
    assert read(sampler.RECORD / 'CHAIN_STARTED.json')['conditional_on_source_performance'] is False
    source.checked(sampler.RECORD / 'IDEA.md', design['idea_sha256'])
    source.checked(sampler.RECORD / 'PREREG.md', design['prereg_sha256'])
    pths = [ROOT / 'archive/reviews/2026-09-07_fit_subject_sampling/SAMPLER_REPLAY_FINAL.json',
            ROOT / 'archive/reviews/2026-09-07_development_error_description/AUXILIARY_POINT_AUDIT.json',
            ROOT / 'archive/reviews/2026-09-07_development_error_description/SUBJECT_SAMPLING_POINT_AUDIT.json']
    for path in pths:
        assert read(path)['status'] in ('passed', 'passed_full168')
    source.write_new(out, {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
                          'studies': studies, 'control_rows_reused_exactly': True,
                          'sampler_proposal_preceded_source_dev_in_recorded_history': True,
                          'supplementary_evidence_sha256': {str(path.relative_to(ROOT)): source.sha(path) for path in pths},
                          'script_sha256': source.sha(Path(__file__)),
                          'new_forwards': 0,
                          'limits': ['Timestamps are local recorded provenance, not third-party notarized time.',
                                     'Hash checks bind current contents, not source medical-label validity.',
                                     'No independent model forward replay; no claim that the 30-hour duration is complete.']})
    print(json.dumps({'status': 'passed', 'completed_training_runs': 6,
                      'sealed_file_entries_rehashed': sum(v['sealed_files_rehashed'] for v in studies.values()),
                      'control_rows_reused_exactly': True}))


if __name__ == '__main__':
    main()
