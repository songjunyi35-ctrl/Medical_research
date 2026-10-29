"""Predeclared root selection: one single model and one pair, using108 only."""
from __future__ import annotations

import datetime
import json
from pathlib import Path
import time

from .governance import ROOT, RECORD, atomic_json, guard, sha
from .audit_validation import validate, aggregate

TRAIN = ROOT / 'archive/research_records/2026-10-29_race_v2_training'
A = ROOT / 'archive/research_records/2026-10-29_race_v2_inference_a'
B = ROOT / 'archive/research_records/2026-10-29_race_v2_inference_b'
OLD = ROOT / 'archive/research_records/2026-10-29_race_e164_inference_test270'
POLICY = RECORD / 'FINAL_SELECTION_POLICY_FREEZE.json'


def read(path):
    return json.loads(Path(path).read_text())


def evidence(path):
    return dict(path=str(path), sha256=sha(path))


def model(model_id):
    if model_id == 'race164':
        ready = A / 'final270/reuse_race164/ATOMS_READY.json'
        return dict(model_id=model_id,
                    checkpoint_path=str(ROOT / 'workspace/BraTS/segresnet_t1ce_wt/race_20261028/race/best.pth'),
                    checkpoint_sha256='4bd6c35f7e12ecdf1540bf88128f9d3f9853340d801b32e77779fc4ba56eb3d7',
                    architecture='sammed3d.race_seg.RaceSeg',
                    construction=dict(width=32, dropout=.1, dim=16, enabled=True),
                    atom_source='reuse_race164', owner='inference_a', atoms_ready_path=str(ready),
                    atoms_ready_sha256=sha(ready), historical_atom_reports=[
                        evidence(OLD / f'FORWARD_COMPLETE_{rank}.json') for rank in (0, 1)])
    if model_id == 'w32e171':
        return dict(model_id=model_id,
                    checkpoint_path=str(ROOT / 'archive/research_records/2026-10-28_segresnet_w32_e171_test270/checkpoint_epoch171.pth'),
                    checkpoint_sha256='02a60b2d37da0d8de8018c221e1c00566106bd21d0d713b95ac3d9f94d87fade',
                    architecture='sammed3d.t1ce_wt_capg3d.AutomaticPromptLocator',
                    construction=dict(center_aware=False, init_filters=32, dropout=.1),
                    atom_source='new_w32', owner='inference_b',
                    atoms_ready_path=str(B / 'final270/models/w32e171/ATOMS_READY.json'))
    assert model_id == 'race_v2_final'
    handoff = read(TRAIN / 'HANDOFF.json')
    registration = RECORD / 'SNAPSHOT108_QUEUE/race_v2_final_best108.json'
    return dict(model_id=model_id, checkpoint_path=handoff['checkpoint_path'],
                checkpoint_sha256=handoff['checkpoint_sha256'], architecture=handoff['architecture'],
                construction=handoff['construction'], atom_source='new_snapshot', owner='inference_a',
                atoms_ready_path=str(A / 'final270/models/race_v2_final/ATOMS_READY.json'),
                handoff_path=str(TRAIN / 'HANDOFF.json'), handoff_sha256=sha(TRAIN / 'HANDOFF.json'),
                registration_108_path=str(registration), registration_108_sha256=sha(registration))


def select():
    guard(1154121)
    policy = read(POLICY)
    for relative, digest in policy['source_sha256'].items():
        assert sha(ROOT / relative) == digest, relative
    done = read(TRAIN / 'DONE.json')
    assert done['state'] == 'complete'
    assert done['new_epoch'] == done['target_new_epochs'] == read(TRAIN / 'BUDGET_DECISION.json')['new_epochs']
    assert done['new_updates'] == 122 * done['new_epoch']
    assert read(TRAIN / 'RESULT.json')['state'] == 'complete'
    assert read(RECORD / 'JOINT_INFERENCE108_AUDIT.json')['state'] == 'passed'
    selected = read(A / 'SELECTED_RECIPE.json')['candidate']
    initial_a = read(A / 'RANKING.json')
    initial_b = read(B / 'RESULT.json')
    single_options = [dict(origin='old', model_id='race164', recipe=selected,
                           metrics=initial_a['best_metrics_ci'], sources=[evidence(A / 'SELECTED_RECIPE.json'),
                                                                        evidence(A / 'RANKING.json')])]
    old_fusion = initial_b['fusion_ranking'][0]
    fusion_options = [dict(origin='old', ids=['race164', 'w32e171'],
                           fusion=dict(primary_weight=old_fusion['race_weight'], space=old_fusion['space'],
                                       decoder=old_fusion['decoder']), metrics=old_fusion['metrics'],
                           sources=[evidence(B / 'RESULT.json')], stability=initial_b['oof_metrics'])]
    audit_paths = [evidence(RECORD / 'JOINT_INFERENCE108_AUDIT.json')]
    best_epoch = read(TRAIN / 'HANDOFF.json')['actual_new_epochs']
    if best_epoch > 0:
        new_a_dir = A / 'snapshots/race_v2_final_best108'
        new_a = read(new_a_dir / 'RESULT.json')
        new_b = read(B / 'final_snapshot/RESULT.json')
        assert new_a['state'] == new_b['state'] == 'complete'
        assert new_a['selected_transfer_id'] == selected['id']
        handoff = read(TRAIN / 'HANDOFF.json')
        assert new_a['checkpoint_sha256'] == handoff['checkpoint_sha256']
        assert new_b['frozen_partner']['candidate_checkpoint_sha256'] == handoff['checkpoint_sha256']
        cohort_path = ROOT / 'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/validation_t1ce_wt.jsonl'
        cohort = [json.loads(line) for line in cohort_path.read_text().splitlines() if line.strip()]
        summaries = {}
        reports = [new_a_dir / 'RAW_VALIDATION108.json', B / 'final_snapshot/RAW_VALIDATION.json']
        for report_path in reports:
            for name, cases in read(report_path)['per_configuration'].items():
                validate(cases, cohort)
                summaries[f'{report_path.parent.name}:{name}'] = aggregate(cases)
        audit = RECORD / 'FINAL_VALIDATION_ARITHMETIC_AUDIT.json'
        atomic_json(audit, dict(state='passed', report_sha256={str(path): sha(path) for path in reports},
                                summaries=summaries, new_neural_forwards=0, new_pixel_access=False))
        audit_paths.append(evidence(audit))
        single_options.append(dict(origin='new', model_id='race_v2_final', recipe=selected,
                                   metrics=new_a['summary'][selected['id']]['metrics'],
                                   sources=[evidence(new_a_dir / 'RESULT.json')]))
        pair = new_b['fusion_ranking'][0]
        fusion_options.append(dict(origin='new', ids=['race_v2_final', new_b['global_partner']],
                                   fusion=dict(primary_weight=pair['candidate_weight'], space=pair['space'],
                                               decoder=pair['decoder']), metrics=pair['metrics'],
                                   sources=[evidence(B / 'final_snapshot/RESULT.json')], stability=new_b['oof_metrics']))
    else:
        assert (RECORD / 'FINAL_HANDOFF_NO_NEW_MODEL.json').exists()
        audit_paths.append(evidence(RECORD / 'FINAL_HANDOFF_NO_NEW_MODEL.json'))
    # The order deliberately gives old configurations priority on exact ties.
    single = max(single_options, key=lambda value: value['metrics']['scan_dice'])
    pair = max(fusion_options, key=lambda value: value['metrics']['scan_dice'])
    recipe_path = OLD / 'SELECTED_RECIPE.json'
    candidates = [dict(candidate_id='single', kind='single', models=[model(single['model_id'])],
                       single_recipe=single['recipe'], validation_metrics=single['metrics'],
                       validation_sources=single['sources'], origin=single['origin'], deployment_forwards=8),
                  dict(candidate_id='fusion', kind='fusion', models=[model(name) for name in pair['ids']],
                       fusion=pair['fusion'], full_recipe_path=str(recipe_path), full_recipe_sha256=sha(recipe_path),
                       validation_metrics=pair['metrics'], validation_sources=pair['sources'],
                       patient_fold_stability=pair['stability'], origin=pair['origin'], deployment_forwards=16)]
    entry_sources = dict(policy['source_sha256'])
    for path in (A / 'FINAL270_ENTRY_FREEZE.json', B / 'FINAL270_ENTRY_FREEZE.json'):
        frozen = read(path)
        for relative, digest in frozen['source_sha256'].items():
            assert sha(ROOT / relative) == digest, relative
            if relative in entry_sources:
                assert entry_sources[relative] == digest
            entry_sources[relative] = digest
    for candidate in candidates:
        assert len(candidate['models']) == (1 if candidate['kind'] == 'single' else 2)
        for item in candidate['models']:
            assert sha(item['checkpoint_path']) == item['checkpoint_sha256']
    cache = read(OLD / 'CACHE_READY.json')
    result = dict(schema='aflow.race_v2.final_test270.v1', registered_by='root',
                  approval_scope='final_frozen_test270_only', at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  selection_policy_sha256=sha(POLICY), training_terminal=dict(
                      done_path=str(TRAIN / 'DONE.json'), done_sha256=sha(TRAIN / 'DONE.json'),
                      budget_path=str(TRAIN / 'BUDGET_DECISION.json'), budget_sha256=sha(TRAIN / 'BUDGET_DECISION.json'),
                      result_path=str(TRAIN / 'RESULT.json'), result_sha256=sha(TRAIN / 'RESULT.json')),
                  validation_audit=audit_paths,
                  cohort=dict(manifest_path=str(ROOT / 'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/sealed_test/test_cases_t1ce.jsonl'),
                              manifest_sha256='a5ddc0ed2c0b269431758b14061da3c6163dd0c48133b60a2ef8efd7f23aa8f9',
                              scans=270, subjects=119, canvas=[160, 208, 168], spacing_mm=[1, 1, 1], modality='T1ce',
                              scoring='native1mm original crop excludes padding'),
                  data=dict(native_cache='/tmp/aflow_race_e164_test270_20261029/native',
                            index_sha256=cache['index_sha256'], payload_sha256=cache['payload_sha256'],
                            existing_cache_audit_path=str(A / 'EXISTING270_CACHE_AUDIT.json'),
                            existing_cache_audit_sha256=sha(A / 'EXISTING270_CACHE_AUDIT.json')),
                  candidates=candidates, entry_source_sha256=entry_sources,
                  single_selection_options=single_options, fusion_selection_options=fusion_options,
                  new270_used_for_selection=False, inherited_test_tuned_history=True, independent_confirmation=False)
    path = RECORD / 'FINAL_TEST_SELECTION.json'
    assert not path.exists(), 'final selection is immutable'
    atomic_json(path, result)
    path.chmod(0o444)


def main():
    while True:
        guard(1154121)
        if (RECORD / 'FINAL_TEST_SELECTION.json').exists():
            return
        ready = (POLICY.exists() and (TRAIN / 'RESULT.json').exists() and
                 (A / 'FINAL270_ENTRY_FREEZE.json').exists() and (B / 'FINAL270_ENTRY_FREEZE.json').exists())
        if ready:
            best = read(TRAIN / 'HANDOFF.json')['actual_new_epochs']
            ready = ((RECORD / 'FINAL_HANDOFF_NO_NEW_MODEL.json').exists() if best == 0 else
                     (A / 'snapshots/race_v2_final_best108/RESULT.json').exists() and
                     (B / 'final_snapshot/RESULT.json').exists())
        if ready:
            select()
            return
        time.sleep(30)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        atomic_json(RECORD / 'FINAL_SELECTION_FAILURE.json', {'at_unix': time.time(), 'error': repr(error),
                                                             'selection_not_bypassed': True})
        raise
