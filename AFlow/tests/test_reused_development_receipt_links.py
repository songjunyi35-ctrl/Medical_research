import json

import pytest

from scripts import audit_reused_development_receipt_links as audit


def make_study(tmp_path, monkeypatch, flaw=None):
    p = audit.source
    record = tmp_path / 'record'
    work = tmp_path / 'work'
    record.mkdir()
    run = work / 'seed1_control'
    run.mkdir(parents=True)
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    monkeypatch.setattr(p, 'RECORD', record)
    monkeypatch.setattr(p, 'WORK', work)
    monkeypatch.setattr(p, 'SEEDS', (1,))
    monkeypatch.setattr(p, 'FIT_SHA', 'fit')
    monkeypatch.setattr(p, 'check_freeze', lambda: None)

    def save(path, data):
        path.write_text(json.dumps(data))
        return p.sha(path)

    bound = {}
    for name, key in [('IMPLEMENTATION_FREEZE.json', 'implementation_freeze_sha256'),
                      ('SMOKE_RECEIPT.json', 'smoke_sha256'),
                      ('AUX_CACHE_RECEIPT.json', 'aux_cache_receipt_sha256')]:
        bound[key] = save(record / name, {})
    pre_sha = save(record / 'TRAIN_PREACCESS.json', {'created_utc': '2026-09-07T01:00:00+00:00', **bound})
    started = {'created_utc': '2026-09-07T01:01:00+00:00', 'train_preaccess_sha256': pre_sha,
               'seed': 1, 'arm': 'control', 'fit_sha256': 'fit', 'initial_state_sha256': 'init',
               'teacher_checkpoints': [], 'development_metrics_accessed': False}
    if flaw == 'ordering':
        started['created_utc'] = '2026-09-07T00:59:00+00:00'
    save(run / 'STARTED.json', started)
    checkpoint = run / 'fixed.pth'
    checkpoint.write_bytes(b'synthetic hash placeholder; never a loaded model')
    report = {'seed': 1, 'arm': 'control', 'fit_sha256': 'fit', 'initial_state_sha256': 'init',
              'teacher_checkpoints': ['unexpected'] if flaw == 'teacher' else [],
              'all_gates_passed': True, 'fixed_epochs': 168,
              'history': [{'steps': 121} for _ in range(168)], 'export_max_abs_error': 0,
              'access': dict.fromkeys(['development_metrics', 'test', 'confirmation', 'outer', 'external'], False),
              'checkpoint': {'path': str(checkpoint), 'sha256': p.sha(checkpoint)}}
    report_sha = save(run / 'TRAIN_DONE.json', report)
    dev = {'created_utc': '2026-09-07T03:00:00+00:00', 'reports': {'seed1_control': {
        'train_report_sha256': 'wrong' if flaw == 'reference' else report_sha,
        'checkpoint_sha256': p.sha(checkpoint)}}}
    dev_sha = save(record / 'DEV_PREACCESS.json', dev)
    access = dict.fromkeys(['confirmation', 'external', 'fixed_validation108', 'outer', 'test270'], False)
    access['reused_development180'] = True
    raw_sha = save(record / 'RAW_DEVELOPMENT.json', {'dev_preaccess_sha256': dev_sha, 'access': access})
    result_sha = save(record / 'RESULT.json', {'created_utc': '2026-09-07T03:01:00+00:00',
                                             'raw_sha256': raw_sha, 'all_gates_passed': False})
    verify_sha = save(record / 'VERIFICATION_RECEIPT.json', {'status': 'passed', 'result_sha256': result_sha})
    save(record / 'CHAIN_DONE.json', {'status': 'completed', 'result_sha256': result_sha,
                                     'verification_sha256': verify_sha})
    save(record / 'ARCHIVE_SEAL.json', {'files': {'record/RESULT.json': result_sha}})
    return p


def test_completed_negative_study_has_valid_receipt_chain(tmp_path, monkeypatch):
    p = make_study(tmp_path, monkeypatch)
    result = audit.verify_study(p, ('control',))
    assert result['formal_joint_gate_passed'] is False
    assert result['training']['seed1_control']['teacher_free']


@pytest.mark.parametrize('flaw', ['ordering', 'teacher', 'reference'])
def test_receipt_chain_rejects_order_teacher_or_report_mismatch(tmp_path, monkeypatch, flaw):
    p = make_study(tmp_path, monkeypatch, flaw)
    with pytest.raises(AssertionError):
        audit.verify_study(p, ('control',))
