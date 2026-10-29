"""Select at most one single and one SegResNet pair from complete108 evidence.

The original training, original selector and original entry freezes remain
unchanged. This replacement includes the two bounded inference innovations.
It never reads270 labels, predictions or scores when selecting candidates.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
import time

from .governance import ROOT, RECORD, atomic_json, guard, sha
from .audit_validation import MANIFEST, validate, aggregate, paired
from .select_final import A, B, TRAIN, OLD, evidence, model, read

IA = ROOT / 'archive/research_records/2026-10-29_v2_inference_innovation_a'
IB = B / 'innovation_b'
POLICY = RECORD / 'FINAL_SELECTION_INNOVATION_POLICY_FREEZE.json'
AUDIT = RECORD / 'INNOVATION108_ARITHMETIC_AUDIT.json'


def verify_freeze(path):
    frozen = read(path)
    for relative, digest in frozen['source_sha256'].items():
        assert sha(ROOT / relative) == digest, relative
    for relative, digest in frozen.get('evidence_sha256', {}).items():
        assert sha(ROOT / relative) == digest, relative
    return frozen['source_sha256']


def stage_guard():
    state = guard(1154121)
    authorization = read(RECORD / 'AUTHORIZATION.json')
    folders = {ROOT / value for value in authorization['new_persistent_roots']}
    folders.update((IA, ROOT / 'scripts/race_v2_inference_innovation_a_20261029'))
    total = sum(path.stat().st_size for folder in folders for path in folder.rglob('*') if path.is_file())
    assert total <= authorization['persistent_new_max_bytes'], total
    state['extended_stage_bytes'] = total
    return state


def audited_reports(paths):
    cohort = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    assert len(cohort) == 108 and len({row['subject_id'] for row in cohort}) == 50
    reports, summaries = {}, {}
    for path in paths:
        path = Path(path)
        current = read(path)['per_configuration']
        for name, records in current.items():
            validate(records, cohort)
            summaries[f'{path.relative_to(ROOT)}:{name}'] = aggregate(records)
        reports[str(path)] = current
    return reports, summaries


def audit_innovations():
    """Independent integer audit, retaining the original initial-stage audit."""
    paths = [IA / 'RAW_VALIDATION108.json', IB / 'RAW_VALIDATION.json']
    reports, summaries = audited_reports(paths)
    old_a = read(A / 'RAW_VALIDATION108.json')['per_configuration']['equal8_logit_t05']
    old_b = read(B / 'RAW_VALIDATION.json')['per_configuration']
    controls = [(reports[str(paths[0])]['equal8_logit_t05'], old_a),
                (reports[str(paths[1])]['race_equal8'], old_a),
                (reports[str(paths[1])]['w32_equal8'], old_b['w32_equal8']),
                (reports[str(paths[1])]['old_best_fixed075'], old_b['race0.75_logit_raw'])]
    keys = ('case_id', 'subject_id', 'tp', 'fp', 'fn', 'locator_dsc',
            'automatic_point', 'automatic_point_hits_gt')
    for current, reference in controls:
        for row, original in zip(current, reference):
            assert {key: row[key] for key in keys} == {key: original[key] for key in keys}
    best_a = read(IA / 'RESULT.json')['best']
    best_b = read(IB / 'RESULT.json')['best']
    for directory, raw, key, reported in (
        (IA, paths[0], best_a['id'], best_a['metrics']),
        (IB, paths[1], best_b['recipe']['id'], best_b['metrics'])):
        metrics = summaries[f'{raw.relative_to(ROOT)}:{key}']
        for field in ('scan_dice', 'subject_dice', 'point_hits'):
            assert abs(metrics[field] - reported[field]) < 1e-12, (directory, field)
    result = dict(state='passed', at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source_report_sha256={str(path): sha(path) for path in paths},
                  configurations=len(summaries), summaries=summaries,
                  four_references_counts_dice_points_exact=True, no_new_neural_forwards=True,
                  no_clinical_arrays_read=True, new270_access=False)
    if AUDIT.exists():
        assert read(AUDIT)['source_report_sha256'] == result['source_report_sha256']
    else:
        atomic_json(AUDIT, result)
    return result


def terminal_ready():
    required = [TRAIN / 'DONE.json', TRAIN / 'RESULT.json', TRAIN / 'HANDOFF.json',
                A / 'FINAL270_INNOVATION_ENTRY_FREEZE.json',
                B / 'FINAL270_INNOVATION_ENTRY_FREEZE.json', IA / 'DEPLOYMENT_RECIPE.json',
                IB / 'SELECTED_RECIPE.json', IA / 'RESULT.json', IB / 'RESULT.json']
    if not all(path.exists() for path in required):
        return False
    best = read(TRAIN / 'HANDOFF.json')['actual_new_epochs']
    if best == 0:
        return (RECORD / 'FINAL_HANDOFF_NO_NEW_MODEL.json').exists()
    return all(path.exists() for path in (
        A / 'snapshots/race_v2_final_best108/RESULT.json', B / 'final_snapshot/RESULT.json',
        IA / 'final_snapshot/RESULT.json', IB / 'final_snapshot/RESULT.json'))


def select():
    disk = stage_guard()
    sources = verify_freeze(POLICY)
    assert read(RECORD / 'JOINT_INFERENCE108_AUDIT.json')['state'] == 'passed'
    assert read(AUDIT)['state'] == 'passed'
    done, handoff = read(TRAIN / 'DONE.json'), read(TRAIN / 'HANDOFF.json')
    assert done['state'] == read(TRAIN / 'RESULT.json')['state'] == 'complete'
    assert done['new_epoch'] == done['target_new_epochs'] == read(TRAIN / 'BUDGET_DECISION.json')['new_epochs']
    assert done['new_updates'] == done['new_epoch'] * 122
    assert handoff['total_stage_new_epochs'] == done['new_epoch']
    assert handoff['total_stage_new_updates'] == done['new_updates']
    assert 0 <= handoff['actual_new_epochs'] <= done['new_epoch']
    assert handoff['actual_new_updates'] == 122 * handoff['actual_new_epochs']
    assert handoff['strict_cpu_reload'] is True and handoff['optimizer_new'] is True
    assert handoff['architecture'] == 'scripts.race_v2_20261029.model.RaceBenefitSeg'
    assert handoff['construction'] == dict(width=32, dropout=.1, dim=16, enabled=True)
    assert handoff['source_epoch'] == 164
    equal8 = read(A / 'SELECTED_RECIPE.json')['candidate']
    innovation_a = read(IA / 'DEPLOYMENT_RECIPE.json')['innovation_entry']
    innovation_b = read(IB / 'SELECTED_RECIPE.json')['candidate']
    assert innovation_a['mode'] == 'race8_local_dependency'
    assert innovation_b['kind'] == 'paired_covariance_shrinkage'
    assert innovation_a['config'] in read(IA / 'PLAN.json')['recipes']
    assert innovation_a['id'] == innovation_a['config']['id']
    assert innovation_b == read(IB / 'RESULT.json')['best']['recipe']
    paths = [IA / 'RAW_VALIDATION108.json', IB / 'RAW_VALIDATION.json']
    reports, _ = audited_reports(paths)
    initial_a = read(A / 'RANKING.json')
    initial_b = read(B / 'RESULT.json')
    old_a_rows = read(A / 'RAW_VALIDATION108.json')['per_configuration'][equal8['id']]
    old_b_rows = read(B / 'RAW_VALIDATION.json')['per_configuration']
    cohort = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    validate(old_a_rows, cohort)
    verified_old_single = aggregate(old_a_rows)
    for key in ('scan_dice', 'subject_dice', 'point_hits'):
        assert abs(verified_old_single[key] - initial_a['best_metrics_ci'][key]) < 1e-12
    single_options = [dict(origin='old_equal8', model_id='race164', recipe=equal8,
        metrics=verified_old_single, sources=[evidence(A / 'RANKING.json'), evidence(A / 'SELECTED_RECIPE.json')])]
    single_options.append(dict(origin='old_model_new_inference', model_id='race164', recipe=innovation_a,
        metrics=aggregate(reports[str(paths[0])][innovation_a['id']]),
        sources=[evidence(IA / 'RESULT.json'), evidence(IA / 'DEPLOYMENT_RECIPE.json')]))
    old_pair = initial_b['fusion_ranking'][0]
    validate(old_b_rows[old_pair['id']], cohort)
    verified_old_pair = aggregate(old_b_rows[old_pair['id']])
    for key in ('scan_dice', 'subject_dice', 'point_hits'):
        assert abs(verified_old_pair[key] - old_pair['metrics'][key]) < 1e-12
    pair_options = [dict(origin='old_global_fusion', ids=['race164', 'w32e171'],
        fusion=dict(primary_weight=old_pair['race_weight'], space=old_pair['space'], decoder=old_pair['decoder']),
        metrics=verified_old_pair, sources=[evidence(B / 'RESULT.json')], stability=initial_b['oof_metrics'])]
    new_fusion = dict(primary_weight=.75, space='paired_covariance_shrinkage', decoder='raw',
        algorithm_recipe=innovation_b, algorithm_recipe_path=str(IB / 'SELECTED_RECIPE.json'),
        algorithm_recipe_sha256=sha(IB / 'SELECTED_RECIPE.json'),
        algorithm_source_sha256=sha(ROOT / 'scripts/race_v2_inference_b_20261029/innovation_b/core.py'))
    pair_options.append(dict(origin='old_models_new_inference', ids=['race164', 'w32e171'], fusion=new_fusion,
        metrics=aggregate(reports[str(paths[1])][innovation_b['id']]),
        sources=[evidence(IB / 'RESULT.json'), evidence(IB / 'SELECTED_RECIPE.json')],
        stability=read(IB / 'RESULT.json')['held_out_metrics']))
    audits = [evidence(RECORD / 'JOINT_INFERENCE108_AUDIT.json'), evidence(AUDIT)]
    if handoff['actual_new_epochs'] > 0:
        new_a_dir = A / 'snapshots/race_v2_final_best108'
        a_new, b_new = read(new_a_dir / 'RESULT.json'), read(B / 'final_snapshot/RESULT.json')
        a_transfer, b_transfer = read(IA / 'final_snapshot/RESULT.json'), read(IB / 'final_snapshot/RESULT.json')
        assert all(result['state'] == 'complete' for result in (a_new, b_new, a_transfer, b_transfer))
        assert a_new['checkpoint_sha256'] == handoff['checkpoint_sha256']
        assert b_new['frozen_partner']['candidate_checkpoint_sha256'] == handoff['checkpoint_sha256']
        assert a_transfer['checkpoint_sha256'] == b_transfer['checkpoint_sha256'] == handoff['checkpoint_sha256']
        assert a_transfer['recipe_id'] == innovation_a['id']
        assert b_transfer['recipe'] == innovation_b and b_transfer['partner_model_id'] == b_new['global_partner']
        new_paths = [new_a_dir / 'RAW_VALIDATION108.json', B / 'final_snapshot/RAW_VALIDATION.json',
                     IA / 'final_snapshot/RAW_VALIDATION108.json', IB / 'final_snapshot/RAW_VALIDATION.json']
        new_reports, summary = audited_reports(new_paths)
        audit_path = RECORD / 'FINAL_INNOVATION_VALIDATION_ARITHMETIC_AUDIT.json'
        atomic_json(audit_path, dict(state='passed', report_sha256={str(path): sha(path) for path in new_paths},
                                    summaries=summary, new_neural_forwards=0, new_pixel_access=False))
        audits.append(evidence(audit_path))
        single_options.append(dict(origin='new_model_equal8', model_id='race_v2_final', recipe=equal8,
            metrics=aggregate(new_reports[str(new_paths[0])][equal8['id']]), sources=[evidence(new_a_dir / 'RESULT.json')]))
        single_options.append(dict(origin='new_model_new_inference', model_id='race_v2_final', recipe=innovation_a,
            metrics=aggregate(new_reports[str(new_paths[2])][innovation_a['id']]),
            sources=[evidence(IA / 'final_snapshot/RESULT.json'), evidence(IA / 'DEPLOYMENT_RECIPE.json')]))
        fixed_pair = b_new['fusion_ranking'][0]
        verified_fixed_pair = aggregate(new_reports[str(new_paths[1])][fixed_pair['id']])
        for key in ('scan_dice', 'subject_dice', 'point_hits'):
            assert abs(verified_fixed_pair[key] - fixed_pair['metrics'][key]) < 1e-12
        pair_options.append(dict(origin='new_model_global_fusion', ids=['race_v2_final', b_new['global_partner']],
            fusion=dict(primary_weight=fixed_pair['candidate_weight'], space=fixed_pair['space'], decoder=fixed_pair['decoder']),
            metrics=verified_fixed_pair, sources=[evidence(B / 'final_snapshot/RESULT.json')], stability=b_new['oof_metrics']))
        pair_options.append(dict(origin='new_model_new_inference', ids=['race_v2_final', b_new['global_partner']], fusion=new_fusion,
            metrics=aggregate(new_reports[str(new_paths[3])][innovation_b['id']]),
            sources=[evidence(IB / 'final_snapshot/RESULT.json'), evidence(IB / 'SELECTED_RECIPE.json')],
            stability='one fixed algorithm transfer; not a new12-grid or independent confirmation'))
    else:
        audits.append(evidence(RECORD / 'FINAL_HANDOFF_NO_NEW_MODEL.json'))
    # Highest complete108 Scan only. Existing options win exact ties by order.
    single = max(single_options, key=lambda option: option['metrics']['scan_dice'])
    pair = max(pair_options, key=lambda option: option['metrics']['scan_dice'])
    full_recipe = OLD / 'SELECTED_RECIPE.json'
    candidates = [dict(candidate_id='single', kind='single', models=[model(single['model_id'])],
        single_recipe=single['recipe'], validation_metrics=single['metrics'], validation_sources=single['sources'],
        origin=single['origin'], deployment_forwards=8),
        dict(candidate_id='fusion', kind='fusion', models=[model(identifier) for identifier in pair['ids']],
        fusion=pair['fusion'], full_recipe_path=str(full_recipe), full_recipe_sha256=sha(full_recipe),
        validation_metrics=pair['metrics'], validation_sources=pair['sources'],
        patient_fold_stability=pair['stability'], origin=pair['origin'], deployment_forwards=16)]
    for entry in (A, B):
        for relative, digest in verify_freeze(entry / 'FINAL270_INNOVATION_ENTRY_FREEZE.json').items():
            assert relative not in sources or sources[relative] == digest
            sources[relative] = digest
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
        validation_audit=audits,
        cohort=dict(manifest_path=str(ROOT / 'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/sealed_test/test_cases_t1ce.jsonl'),
            manifest_sha256='a5ddc0ed2c0b269431758b14061da3c6163dd0c48133b60a2ef8efd7f23aa8f9',
            scans=270, subjects=119, canvas=[160,208,168], spacing_mm=[1,1,1], modality='T1ce',
            scoring='native1mm original crop excludes padding'),
        data=dict(native_cache='/tmp/aflow_race_e164_test270_20261029/native', index_sha256=cache['index_sha256'],
            payload_sha256=cache['payload_sha256'], existing_cache_audit_path=str(A / 'EXISTING270_CACHE_AUDIT.json'),
            existing_cache_audit_sha256=sha(A / 'EXISTING270_CACHE_AUDIT.json')),
        candidates=candidates, entry_source_sha256=sources,
        single_selection_options=single_options, fusion_selection_options=pair_options,
        resource_guard=disk, new270_used_for_selection=False, inherited_test_tuned_history=True,
        independent_confirmation=False, training_scope_unchanged=True)
    destination = RECORD / 'FINAL_TEST_SELECTION.json'
    assert not destination.exists(), 'final selection is immutable'
    atomic_json(destination, result)
    destination.chmod(0o444)


def main():
    verify_freeze(POLICY)
    while True:
        stage_guard()
        if (RECORD / 'FINAL_TEST_SELECTION.json').exists():
            return
        if all(path.exists() for path in (IA / 'RESULT.json', IB / 'RESULT.json')):
            audit_innovations()
        if terminal_ready():
            select()
            return
        time.sleep(30)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        atomic_json(RECORD / 'FINAL_SELECTION_INNOVATION_FAILURE.json',
                    dict(at_unix=time.time(), error=repr(error), training_unchanged=True, selection_not_bypassed=True))
        raise
