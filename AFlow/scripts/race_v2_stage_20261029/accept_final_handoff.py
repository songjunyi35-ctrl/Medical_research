"""Root policy: register one fixed best only after the whole training run ends.

Registers validation108 work only. It never registers or launches test270.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
import time

from .governance import ROOT, RECORD, atomic_json, guard, sha

TRAIN = ROOT / 'archive/research_records/2026-10-29_race_v2_training'


def accept():
    guard(1154121)
    done_path, handoff_path = TRAIN / 'DONE.json', TRAIN / 'HANDOFF.json'
    done = json.loads(done_path.read_text())
    handoff = json.loads(handoff_path.read_text())
    budget = json.loads((TRAIN / 'BUDGET_DECISION.json').read_text())
    recipe = json.loads((TRAIN / 'RECIPE.json').read_text())
    frozen = json.loads((TRAIN / 'EXECUTION_FREEZE.json').read_text())
    assert done['state'] == 'complete'
    assert done['new_epoch'] == done['target_new_epochs'] == budget['new_epochs']
    assert done['new_updates'] == 122 * done['new_epoch'] == handoff['total_stage_new_updates']
    assert handoff['total_stage_new_epochs'] == done['new_epoch']
    assert handoff['run_id'] == recipe['run_id']
    assert handoff['strict_cpu_reload'] and handoff['optimizer_new']
    assert handoff['execution_freeze_sha256'] == sha(TRAIN / 'EXECUTION_FREEZE.json')
    assert handoff['recipe_sha256'] == sha(TRAIN / 'RECIPE.json') == frozen['recipe_sha256']
    for relative, digest in frozen['source_sha256'].items():
        assert sha(ROOT / relative) == digest, relative
    if handoff['actual_new_epochs'] == 0:
        atomic_json(RECORD / 'FINAL_HANDOFF_NO_NEW_MODEL.json', dict(
            state='source_e164_remains_best_no_duplicate_forwards', training_done_sha256=sha(done_path),
            handoff_sha256=sha(handoff_path), training_new_epochs=done['new_epoch'],
            training_new_updates=done['new_updates'], new_validation_forwards_registered=0))
        return
    assert 0 < handoff['actual_new_epochs'] <= done['new_epoch']
    assert handoff['actual_new_updates'] == 122 * handoff['actual_new_epochs']
    assert handoff['construction'] == dict(width=32, dropout=.1, dim=16, enabled=True)
    assert handoff['architecture'] == 'scripts.race_v2_20261029.model.RaceBenefitSeg'
    assert handoff['input_geometry']['validation_canvas'] == [160, 208, 160]
    checkpoint = Path(handoff['checkpoint_path']).resolve()
    assert checkpoint.parent == (TRAIN / 'checkpoints').resolve() and '_fixed_' in checkpoint.name
    assert sha(checkpoint) == handoff['checkpoint_sha256']
    manifest = ROOT / 'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/validation_t1ce_wt.jsonl'
    cohort = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    cases = handoff['validation']['per_case']
    assert len(cases) == len({case['case_id'] for case in cases}) == 108
    assert {(case['case_id'], case['subject_id']) for case in cases} == {
        (case['case_id'], case['subject_id']) for case in cohort}
    job = dict(job_id='race_v2_final_best108', registered_by='root',
               approval_scope='final_completed_training_snapshot108_only', run_id=handoff['run_id'],
               handoff_path=str(handoff_path), handoff_sha256=sha(handoff_path),
               training_done_path=str(done_path), training_done_sha256=sha(done_path),
               registered_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
    destination = RECORD / 'SNAPSHOT108_QUEUE/race_v2_final_best108.json'
    assert not destination.exists()
    atomic_json(destination, job)
    atomic_json(RECORD / 'FINAL_HANDOFF_ACCEPTED.json', dict(
        state='registered_fixed_final_best_for_validation108', job=job,
        checkpoint_sha256=handoff['checkpoint_sha256'], actual_new_epochs=handoff['actual_new_epochs'],
        actual_new_updates=handoff['actual_new_updates'], final_test_candidates_registered=0))


def main():
    while True:
        guard(1154121)
        if (TRAIN / 'DONE.json').exists() and (TRAIN / 'HANDOFF.json').exists():
            accept()
            return
        if list(TRAIN.glob('COORDINATOR_FAILURE_*.json')):
            atomic_json(RECORD / 'FINAL_HANDOFF_WAIT_STOPPED.json', dict(
                state='training_coordinator_failure_requires_review', new_model_registered=False))
            return
        time.sleep(30)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        atomic_json(RECORD / 'FINAL_HANDOFF_ACCEPTANCE_FAILURE.json',
                    {'at_unix': time.time(), 'error': repr(error), 'new270_registered': False})
        raise
