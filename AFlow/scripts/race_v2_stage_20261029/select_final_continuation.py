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
from .select_final import A, B, TRAIN, OLD, evidence, model as base_model, read

IA = ROOT / 'archive/research_records/2026-10-29_v2_inference_innovation_a'
IB = B / 'innovation_b'
NA = ROOT / 'archive/research_records/2026-10-29_race_v2_inference_normalization_a'
NB = B / 'image_conflict_b'
SC = ROOT / 'archive/research_records/2026-10-29_race_v2_inference_soft_context_a'
EDGE = B / 'edge_input_b'
APPEARANCE = B / 'appearance_b'
POLICY = RECORD / 'FINAL_SELECTION_CONTINUATION_POLICY_FREEZE.json'
AUDIT = RECORD / 'INNOVATION108_ARITHMETIC_AUDIT.json'


def model(identifier):
    if identifier == 'w32e171_guided':
        base = base_model('w32e171')
        recipe_path = EDGE / 'INPUT_DEPLOYMENT_RECIPE.json'
        return dict(model_id=identifier, base_model_id='w32e171', checkpoint_path=base['checkpoint_path'],
            checkpoint_sha256=base['checkpoint_sha256'], architecture=base['architecture'], construction=base['construction'],
            atom_source='guided_input_w32', owner='inference_b',
            input_recipe=read(recipe_path)['input_recipe'], input_recipe_path=str(recipe_path), input_recipe_sha256=sha(recipe_path),
            atoms_ready_path=str(B / 'final270/models/w32e171_guided/ATOMS_READY.json'),
            learned_weight_count=1, deployment_neural_forwards=8)
    suffix = next((value for value in ('_healthnorm', '_softcontext') if identifier.endswith(value)), None)
    if suffix is None:
        return base_model(identifier)
    base_id = identifier.removesuffix(suffix)
    assert base_id in ('race164', 'race_v2_final')
    seed = base_model(base_id)
    recipe_path = (NA / 'NORMALIZATION_DEPLOYMENT_RECIPE.json' if suffix == '_healthnorm' else
                   SC / 'SOFT_CONTEXT_DEPLOYMENT_RECIPE.json')
    recipe = read(recipe_path)['innovation_entry']
    result = dict(model_id=identifier, base_model_id=base_id,
        checkpoint_path=seed['checkpoint_path'], checkpoint_sha256=seed['checkpoint_sha256'],
        architecture=seed['architecture'], construction=seed['construction'],
        atom_source='seed_health_normalization' if suffix == '_healthnorm' else 'seed_soft_context',
        owner='inference_a', seed_model=seed,
        atoms_ready_path=str(A / 'final270/models' / identifier / 'ATOMS_READY.json'),
        input_recipe=recipe, input_recipe_path=str(recipe_path), input_recipe_sha256=sha(recipe_path),
        learned_weight_count=1, deployment_neural_forwards=8,
        input_chain='one identity seed then seven transformed mirrors; original support and geometry retained')
    if base_id == 'race164':
        result.update(seed_atoms_ready_path=seed['atoms_ready_path'], seed_atoms_ready_sha256=seed['atoms_ready_sha256'])
    else:
        for key in ('handoff_path', 'handoff_sha256', 'registration_108_path', 'registration_108_sha256'):
            result[key] = seed[key]
    return result


def audit_continuation():
    paths = [NA / 'RAW_VALIDATION108.json', NB / 'RAW_VALIDATION.json']
    reports, summary = audited_reports(paths)
    original_a = read(A / 'RAW_VALIDATION108.json')['per_configuration']['equal8_logit_t05']
    original_b = read(B / 'RAW_VALIDATION.json')['per_configuration']['race0.75_logit_raw']
    covariance_id = read(IB / 'SELECTED_RECIPE.json')['candidate']['id']
    original_cov = read(IB / 'RAW_VALIDATION.json')['per_configuration'][covariance_id]
    pairs = [(reports[str(paths[0])]['equal8_logit_t05'], original_a),
             (reports[str(paths[1])]['race_equal8'], original_a),
             (reports[str(paths[1])]['old_best_fixed075'], original_b),
             (reports[str(paths[1])]['old_selected_covariance'], original_cov)]
    keys = ('case_id', 'subject_id', 'tp', 'fp', 'fn', 'locator_dsc', 'automatic_point', 'automatic_point_hits_gt')
    for records, baseline in pairs:
        for row, previous in zip(records, baseline):
            assert {key: row[key] for key in keys} == {key: previous[key] for key in keys}
    path = RECORD / 'CONTINUATION108_ARITHMETIC_AUDIT.json'
    sources = {str(item): sha(item) for item in paths}
    if path.exists():
        assert read(path)['source_report_sha256'] == sources
        return
    atomic_json(path, dict(state='passed', source_report_sha256=sources, summaries=summary,
        configurations=len(summary), new_neural_forwards_by_root=0, no_clinical_arrays_read=True,
        four_references_all_counts_dice_points_exact=True, new270_access=False))


def audit_input_phase4():
    paths = [SC / 'RAW_VALIDATION108.json', EDGE / 'RAW_VALIDATION.json']
    reports, summaries = audited_reports(paths)
    canonical = read(B / 'RAW_VALIDATION.json')['per_configuration']
    controls = {'race_equal8': canonical['race_equal8'], 'old_best_fixed075': canonical['race0.75_logit_raw'],
        'old_selected_covariance': read(IB / 'RAW_VALIDATION.json')['per_configuration'][read(IB / 'SELECTED_RECIPE.json')['candidate']['id']],
        'old_selected_contralateral': read(NB / 'RAW_VALIDATION.json')['per_configuration'][read(NB / 'DEPLOYMENT_RECIPE.json')['candidate']['id']]}
    pairs = [(reports[str(paths[0])]['equal8_logit_t05'], canonical['race_equal8'])]
    pairs.extend((reports[str(paths[1])][key], value) for key, value in controls.items())
    fields = ('case_id', 'subject_id', 'tp', 'fp', 'fn', 'locator_dsc', 'automatic_point', 'automatic_point_hits_gt')
    for rows, reference in pairs:
        for row, previous in zip(rows, reference):
            assert {key: row[key] for key in fields} == {key: previous[key] for key in fields}
    destination = RECORD / 'INPUT_PHASE4_108_ARITHMETIC_AUDIT.json'
    sources = {str(path): sha(path) for path in paths}
    if destination.exists():
        assert read(destination)['source_report_sha256'] == sources
    else:
        atomic_json(destination, dict(state='passed', source_report_sha256=sources, summaries=summaries,
            configurations=len(summaries), new_neural_forwards_by_root=0, new270_access=False,
            no_clinical_arrays_read=True))


def audit_appearance():
    path = APPEARANCE / 'RAW_VALIDATION.json'
    reports, summaries = audited_reports([path])
    edge = read(EDGE / 'RAW_VALIDATION.json')['per_configuration']
    controls = {key: edge[key] for key in ('race_equal8', 'old_best_fixed075', 'old_selected_covariance', 'old_selected_contralateral')}
    controls['old_selected_guided'] = edge[read(EDGE / 'DEPLOYMENT_RECIPE.json')['candidate']['id']]
    fields = ('case_id', 'subject_id', 'tp', 'fp', 'fn', 'locator_dsc', 'automatic_point', 'automatic_point_hits_gt')
    for key, reference in controls.items():
        for row, previous in zip(reports[str(path)][key], reference):
            assert {field: row[field] for field in fields} == {field: previous[field] for field in fields}
    output = RECORD / 'APPEARANCE108_ARITHMETIC_AUDIT.json'
    if output.exists():
        assert read(output)['source_report_sha256'] == sha(path)
    else:
        atomic_json(output, dict(state='passed', source_report_path=str(path), source_report_sha256=sha(path),
            summaries=summaries, new_neural_forwards_by_root=0, no_clinical_arrays_read=True, new270_access=False))


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
    folders.update((IA, ROOT / 'scripts/race_v2_inference_innovation_a_20261029',
                    NA, ROOT / 'scripts/race_v2_inference_normalization_a_20261029',
                    SC, ROOT / 'scripts/race_v2_inference_soft_context_a_20261029'))
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
                IB / 'SELECTED_RECIPE.json', IA / 'RESULT.json', IB / 'RESULT.json',
                NA / 'NORMALIZATION_DEPLOYMENT_RECIPE.json', NB / 'DEPLOYMENT_RECIPE.json',
                NA / 'RESULT.json', NB / 'RESULT.json',
                A / 'FINAL270_NORMALIZATION_ENTRY_FREEZE_V2.json',
                B / 'FINAL270_CONTRALATERAL_ENTRY_FREEZE.json',
                SC / 'RESULT.json', EDGE / 'RESULT.json', SC / 'SOFT_CONTEXT_DEPLOYMENT_RECIPE.json',
                EDGE / 'DEPLOYMENT_RECIPE.json', EDGE / 'INPUT_DEPLOYMENT_RECIPE.json',
                A / 'FINAL270_SOFT_CONTEXT_ENTRY_FREEZE.json', B / 'FINAL270_GUIDED_INPUT_ENTRY_FREEZE.json',
                APPEARANCE / 'RESULT.json', APPEARANCE / 'DEPLOYMENT_RECIPE.json',
                B / 'FINAL270_APPEARANCE_ENTRY_FREEZE.json']
    if not all(path.exists() for path in required):
        return False
    best = read(TRAIN / 'HANDOFF.json')['actual_new_epochs']
    if best == 0:
        return (RECORD / 'FINAL_HANDOFF_NO_NEW_MODEL.json').exists()
    return all(path.exists() for path in (
        A / 'snapshots/race_v2_final_best108/RESULT.json', B / 'final_snapshot/RESULT.json',
        IA / 'final_snapshot/RESULT.json', IB / 'final_snapshot/RESULT.json',
        NA / 'final_snapshot/RESULT.json', NB / 'final_snapshot/RESULT.json',
        SC / 'final_snapshot/RESULT.json', EDGE / 'final_snapshot/RESULT.json',
        APPEARANCE / 'final_snapshot/RESULT.json'))


def select():
    disk = stage_guard()
    sources = verify_freeze(POLICY)
    assert read(RECORD / 'JOINT_INFERENCE108_AUDIT.json')['state'] == 'passed'
    assert read(AUDIT)['state'] == 'passed'
    assert read(RECORD / 'CONTINUATION108_ARITHMETIC_AUDIT.json')['state'] == 'passed'
    assert read(RECORD / 'INPUT_PHASE4_108_ARITHMETIC_AUDIT.json')['state'] == 'passed'
    assert read(RECORD / 'APPEARANCE108_ARITHMETIC_AUDIT.json')['state'] == 'passed'
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
    normalization = read(NA / 'NORMALIZATION_DEPLOYMENT_RECIPE.json')['innovation_entry']
    contralateral = read(NB / 'DEPLOYMENT_RECIPE.json')['candidate']
    context = read(SC / 'SOFT_CONTEXT_DEPLOYMENT_RECIPE.json')['innovation_entry']
    guided = read(EDGE / 'DEPLOYMENT_RECIPE.json')['candidate']
    appearance = read(APPEARANCE / 'DEPLOYMENT_RECIPE.json')['candidate']
    assert normalization['mode'] == 'race8_seed_health_normalization'
    assert normalization['neural_case_forwards'] == 8
    assert normalization['config'] in read(NA / 'PLAN.json')['recipes']
    assert normalization['id'] == normalization['config']['id'] == read(NA / 'RESULT.json')['best_innovation']['id']
    assert contralateral == read(NB / 'RESULT.json')['best_new']['recipe']
    assert contralateral in read(NB / 'PLAN.json')['configurations']
    assert context['mode'] == 'race8_seed_soft_context' and context['neural_case_forwards'] == 8
    assert context['id'] == read(SC / 'RESULT.json')['best_innovation']['id']
    assert context['config'] in read(SC / 'PLAN.json')['recipes']
    assert guided == read(EDGE / 'RESULT.json')['best_new']['recipe']
    assert guided['input_spec'] in read(EDGE / 'PLAN.json')['inputs']
    assert appearance == read(APPEARANCE / 'RESULT.json')['best_new']['recipe']
    assert appearance['kind'] == 'image_appearance_unary'
    assert appearance in read(APPEARANCE / 'PLAN.json')['configurations']
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
    phase_paths = [NA / 'RAW_VALIDATION108.json', NB / 'RAW_VALIDATION.json']
    phase_reports, _ = audited_reports(phase_paths)
    single_options.append(dict(origin='old_weight_seed_health_normalization', model_id='race164_healthnorm',
        recipe=normalization, metrics=aggregate(phase_reports[str(phase_paths[0])][normalization['id']]),
        sources=[evidence(NA / 'RESULT.json'), evidence(NA / 'NORMALIZATION_DEPLOYMENT_RECIPE.json')]))
    contra_fusion = dict(primary_weight=.75, space='contralateral_image_evidence', decoder='raw',
        algorithm_recipe=contralateral, algorithm_recipe_path=str(NB / 'DEPLOYMENT_RECIPE.json'),
        algorithm_recipe_sha256=sha(NB / 'DEPLOYMENT_RECIPE.json'),
        algorithm_source_sha256=read(NB / 'DEPLOYMENT_RECIPE.json')['source_sha256'])
    pair_options.append(dict(origin='old_models_contralateral_image_evidence', ids=['race164', 'w32e171'],
        fusion=contra_fusion, metrics=aggregate(phase_reports[str(phase_paths[1])][contralateral['id']]),
        sources=[evidence(NB / 'RESULT.json'), evidence(NB / 'DEPLOYMENT_RECIPE.json')],
        stability='patient folds in source RESULT; conditional on previously selected models'))
    input_paths = [SC / 'RAW_VALIDATION108.json', EDGE / 'RAW_VALIDATION.json']
    input_reports, _ = audited_reports(input_paths)
    single_options.append(dict(origin='old_weight_seed_soft_context', model_id='race164_softcontext', recipe=context,
        metrics=aggregate(input_reports[str(input_paths[0])][context['id']]),
        sources=[evidence(SC / 'RESULT.json'), evidence(SC / 'SOFT_CONTEXT_DEPLOYMENT_RECIPE.json')]))
    if guided['decoder'] == 'fixed075':
        guided_fusion = dict(primary_weight=.75, space='logit', decoder='raw')
    else:
        assert guided['decoder'] == 'covariance' and guided['covariance_recipe'] == innovation_b
        guided_fusion = dict(new_fusion)
    pair_options.append(dict(origin='old_primary_guided_w32_input', ids=['race164', 'w32e171_guided'],
        fusion=guided_fusion, metrics=aggregate(input_reports[str(input_paths[1])][guided['id']]),
        sources=[evidence(EDGE / 'RESULT.json'), evidence(EDGE / 'DEPLOYMENT_RECIPE.json'),
                 evidence(EDGE / 'INPUT_DEPLOYMENT_RECIPE.json')],
        stability='patient grouped selection in source; reused historically selected weights/decoder'))
    appearance_path = APPEARANCE / 'RAW_VALIDATION.json'
    appearance_reports, _ = audited_reports([appearance_path])
    appearance_fusion = dict(primary_weight=.75, space='image_appearance_unary', decoder='raw',
        algorithm_recipe=appearance, algorithm_recipe_path=str(APPEARANCE / 'DEPLOYMENT_RECIPE.json'),
        algorithm_recipe_sha256=sha(APPEARANCE / 'DEPLOYMENT_RECIPE.json'),
        algorithm_source_sha256=read(APPEARANCE / 'DEPLOYMENT_RECIPE.json')['source_sha256'])
    pair_options.append(dict(origin='old_models_image_appearance', ids=['race164', 'w32e171'], fusion=appearance_fusion,
        metrics=aggregate(appearance_reports[str(appearance_path)][appearance['id']]),
        sources=[evidence(APPEARANCE / 'RESULT.json'), evidence(APPEARANCE / 'DEPLOYMENT_RECIPE.json')],
        stability='within-reused108 patient-grouped robustness; transductive image seeds are GT-free'))
    audits = [evidence(RECORD / 'JOINT_INFERENCE108_AUDIT.json'), evidence(AUDIT)]
    audits.append(evidence(RECORD / 'CONTINUATION108_ARITHMETIC_AUDIT.json'))
    audits.append(evidence(RECORD / 'INPUT_PHASE4_108_ARITHMETIC_AUDIT.json'))
    audits.append(evidence(RECORD / 'APPEARANCE108_ARITHMETIC_AUDIT.json'))
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
        phase_a, phase_b = read(NA / 'final_snapshot/RESULT.json'), read(NB / 'final_snapshot/RESULT.json')
        assert phase_a['state'] == phase_b['state'] == 'complete'
        assert phase_a['checkpoint_sha256'] == phase_b['checkpoint_sha256'] == handoff['checkpoint_sha256']
        assert phase_a['recipe_id'] == normalization['id']
        assert phase_b['recipe'] == contralateral and phase_b['partner_model_id'] == b_new['global_partner']
        phase_new_paths = [NA / 'final_snapshot/RAW_VALIDATION108.json', NB / 'final_snapshot/RAW_VALIDATION.json']
        phase_new, phase_summary = audited_reports(phase_new_paths)
        phase_audit = RECORD / 'FINAL_CONTINUATION_VALIDATION_ARITHMETIC_AUDIT.json'
        atomic_json(phase_audit, dict(state='passed', report_sha256={str(p): sha(p) for p in phase_new_paths},
            summaries=phase_summary, new_neural_forwards_by_root=0, no_clinical_arrays_read=True))
        audits.append(evidence(phase_audit))
        single_options.append(dict(origin='new_weight_seed_health_normalization', model_id='race_v2_final_healthnorm',
            recipe=normalization, metrics=aggregate(phase_new[str(phase_new_paths[0])][normalization['id']]),
            sources=[evidence(NA / 'final_snapshot/RESULT.json'), evidence(NA / 'NORMALIZATION_DEPLOYMENT_RECIPE.json')]))
        pair_options.append(dict(origin='new_model_contralateral_image_evidence',
            ids=['race_v2_final', b_new['global_partner']], fusion=contra_fusion,
            metrics=aggregate(phase_new[str(phase_new_paths[1])][contralateral['id']]),
            sources=[evidence(NB / 'final_snapshot/RESULT.json'), evidence(NB / 'DEPLOYMENT_RECIPE.json')],
            stability='one fixed image-evidence formula transfer; no new8-grid'))
        context_transfer, guided_transfer = read(SC / 'final_snapshot/RESULT.json'), read(EDGE / 'final_snapshot/RESULT.json')
        assert context_transfer['state'] == guided_transfer['state'] == 'complete'
        assert context_transfer['checkpoint_sha256'] == guided_transfer['checkpoint_sha256'] == handoff['checkpoint_sha256']
        assert context_transfer['recipe_id'] == context['id']
        assert guided_transfer['recipe'] == guided and guided_transfer['partner_model_id'] == 'w32e171_guided'
        input_new_paths = [SC / 'final_snapshot/RAW_VALIDATION108.json', EDGE / 'final_snapshot/RAW_VALIDATION.json']
        input_new, input_summary = audited_reports(input_new_paths)
        input_audit = RECORD / 'FINAL_INPUT_PHASE4_VALIDATION_ARITHMETIC_AUDIT.json'
        atomic_json(input_audit, dict(state='passed', report_sha256={str(p): sha(p) for p in input_new_paths},
            summaries=input_summary, new_neural_forwards_by_root=0, no_clinical_arrays_read=True))
        audits.append(evidence(input_audit))
        single_options.append(dict(origin='new_weight_seed_soft_context', model_id='race_v2_final_softcontext', recipe=context,
            metrics=aggregate(input_new[str(input_new_paths[0])][context['id']]),
            sources=[evidence(SC / 'final_snapshot/RESULT.json'), evidence(SC / 'SOFT_CONTEXT_DEPLOYMENT_RECIPE.json')]))
        pair_options.append(dict(origin='new_primary_guided_w32_input', ids=['race_v2_final', 'w32e171_guided'],
            fusion=guided_fusion, metrics=aggregate(input_new[str(input_new_paths[1])][guided['id']]),
            sources=[evidence(EDGE / 'final_snapshot/RESULT.json'), evidence(EDGE / 'INPUT_DEPLOYMENT_RECIPE.json')],
            stability='one fixed input/decoder transfer; no new4-grid'))
        appearance_transfer = read(APPEARANCE / 'final_snapshot/RESULT.json')
        assert appearance_transfer['state'] == 'complete' and appearance_transfer['checkpoint_sha256'] == handoff['checkpoint_sha256']
        assert appearance_transfer['recipe'] == appearance and appearance_transfer['partner_model_id'] == b_new['global_partner']
        appearance_new_path = APPEARANCE / 'final_snapshot/RAW_VALIDATION.json'
        appearance_new, appearance_summary = audited_reports([appearance_new_path])
        appearance_audit = RECORD / 'FINAL_APPEARANCE_VALIDATION_ARITHMETIC_AUDIT.json'
        atomic_json(appearance_audit, dict(state='passed', report_path=str(appearance_new_path),
            report_sha256=sha(appearance_new_path), summaries=appearance_summary, new_neural_forwards_by_root=0))
        audits.append(evidence(appearance_audit))
        pair_options.append(dict(origin='new_model_image_appearance', ids=['race_v2_final', b_new['global_partner']],
            fusion=appearance_fusion, metrics=aggregate(appearance_new[str(appearance_new_path)][appearance['id']]),
            sources=[evidence(APPEARANCE / 'final_snapshot/RESULT.json'), evidence(APPEARANCE / 'DEPLOYMENT_RECIPE.json')],
            stability='one fixed appearance-unary transfer; no new4-grid'))
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
    for entry in (A / 'FINAL270_SOFT_CONTEXT_ENTRY_FREEZE.json', B / 'FINAL270_APPEARANCE_ENTRY_FREEZE.json'):
        for relative, digest in verify_freeze(entry).items():
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
        if all(path.exists() for path in (NA / 'RESULT.json', NB / 'RESULT.json')):
            audit_continuation()
        if all(path.exists() for path in (SC / 'RESULT.json', EDGE / 'RESULT.json')):
            audit_input_phase4()
        if (APPEARANCE / 'RESULT.json').exists():
            audit_appearance()
        if terminal_ready():
            select()
            return
        time.sleep(30)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        atomic_json(RECORD / 'FINAL_SELECTION_CONTINUATION_FAILURE.json',
                    dict(at_unix=time.time(), error=repr(error), training_unchanged=True, selection_not_bypassed=True))
        raise
