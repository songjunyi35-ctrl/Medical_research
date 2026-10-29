"""Final-only global partner selection; no model, image, or test access."""
import argparse
import numpy as np
from scripts.race_v2_inference_b_20261029.common import *

S = R/'final_snapshot'
HISTORICAL = {
    'race164':dict(report=ROOT/'archive/research_records/2026-10-28_race_seg/race/validation_e164.json',
        checkpoint_sha256=RACE_SHA,ready=A/'ATOMS_READY.json'),
    'w32e171':dict(report=ROOT/'workspace/BraTS/segresnet_t1ce_wt/width_resume_20261027/w32/validation_e171.json',
        checkpoint_sha256=CHECKPOINT_SHA,ready=OUT/'ATOMS_READY.json')}

class NoNewBest(RuntimeError):pass

def verify_entry():
    verify();frozen=read(R/'SNAPSHOT_ENTRY_FREEZE.json')
    for path,digest in frozen['source_sha256'].items():assert sha(ROOT/path)==digest,path
    assert sha(R/'SNAPSHOT_PLAN.json')==frozen['plan_sha256']

def final_handoff(path):
    path=Path(path);handoff=read(path);done=read(path.with_name('DONE.json'))
    budget=read(path.with_name('BUDGET_DECISION.json'))
    assert done['state']=='complete' and done['new_epoch']==done['target_new_epochs']==budget['new_epochs']
    assert handoff['total_stage_new_epochs']==done['new_epoch']
    assert handoff['total_stage_new_updates']==done['new_updates']==done['new_epoch']*122
    assert handoff['strict_cpu_reload'] and handoff['actual_new_updates']==handoff['actual_new_epochs']*122
    if handoff['actual_new_epochs']==0:
        write(S/'NO_NEW_SELECTED.json',dict(at=now(),state='no_new_best',handoff_sha256=sha(path),
            total_new_epochs=handoff['total_stage_new_epochs'],new_model_forwards=0,reason='Final best has zero new epochs; do not duplicate RACE164 cache'))
        raise NoNewBest('no new validation-selected snapshot')
    checkpoint=Path(handoff['checkpoint_path']);assert 'latest' not in checkpoint.name
    assert checkpoint.exists() and sha(checkpoint)==handoff['checkpoint_sha256']
    cohort=rows();report=handoff['validation']['per_case'];lookup={r['case_id']:r for r in report}
    assert len(lookup)==108 and set(lookup)=={r['case_id'] for r in cohort}
    assert all(lookup[r['case_id']]['subject_id']==r['subject_id'] for r in cohort)
    assert handoff['input_geometry']['validation_canvas']==list(CANVAS)
    return handoff,[lookup[r['case_id']] for r in cohort]

def select(current,options,indices):
    indices=np.asarray(indices,dtype=int);candidate_mean=float(current[indices].mean());diagnostics=[]
    for option in options:
        values=option['values'][indices];candidate=current[indices]
        correlation=float(np.corrcoef(candidate,values)[0,1]) if candidate.std()>0 and values.std()>0 else 1.
        diagnostics.append(dict(**{k:v for k,v in option.items() if k!='values'},
            raw_scan_dice=float(values.mean()),raw_case_dice_correlation=correlation))
    eligible=[p for p in diagnostics if p['raw_scan_dice']>=candidate_mean-.03]
    if not eligible:
        selected=min(diagnostics,key=lambda p:(-p['raw_scan_dice'],p['id']));reason='neither within candidate minus0.03; stronger single reference'
    elif len(eligible)==1:selected=eligible[0];reason='only partner within candidate minus0.03'
    elif abs(eligible[0]['raw_case_dice_correlation']-eligible[1]['raw_case_dice_correlation'])>=.01:
        selected=min(eligible,key=lambda p:(p['raw_case_dice_correlation'],-p['raw_scan_dice'],p['id']))
        reason='lower raw per-scan Dice correlation by at least0.01; fixed global diversity proxy'
    else:
        selected=min(eligible,key=lambda p:(-p['raw_scan_dice'],p['id']));reason='correlation gap below0.01; stronger raw single reference'
    return dict(selected_partner=selected,all_partner_diagnostics=diagnostics,selection_reason=reason,
                candidate_raw_scan=candidate_mean)

def choose_partner(handoff_path):
    verify_entry();handoff,candidate=final_handoff(handoff_path)
    plan=read(R/'SNAPSHOT_PLAN.json');cohort=rows();current=np.array([r['locator_dsc'] for r in candidate],np.float64)
    options=[]
    for name,source in HISTORICAL.items():
        assert sha(source['report'])==plan['historical_report_sha256'][name]
        previous=read(source['report'])['per_case'];lookup={r['case_id']:r for r in previous}
        assert set(lookup)=={r['case_id'] for r in cohort}
        values=np.array([lookup[r['case_id']]['locator_dsc'] for r in cohort],np.float64)
        assert all(lookup[r['case_id']]['subject_id']==r['subject_id'] for r in cohort)
        ready=read(source['ready']);assert ready['checkpoint_sha256']==source['checkpoint_sha256']
        assert ready['manifest_sha256']==sha(MANIFEST) and ready['case_forwards']==864
        options.append(dict(id=name,values=values,checkpoint_sha256=source['checkpoint_sha256'],atoms_ready=str(source['ready']),
            atoms_ready_sha256=sha(source['ready']),report=str(source['report']),report_sha256=sha(source['report'])))
    global_choice=select(current,options,range(108))
    subjects=sorted({r['subject_id'] for r in cohort},key=lambda sid:hashlib.sha256(('race_v2_20261029:'+sid).encode()).hexdigest())
    fold_for={sid:i%5 for i,sid in enumerate(subjects)};folds=[]
    for fold in range(5):
        development=[i for i,r in enumerate(cohort) if fold_for[r['subject_id']]!=fold]
        held_out=[i for i,r in enumerate(cohort) if fold_for[r['subject_id']]==fold]
        choice=select(current,options,development)
        folds.append(dict(fold=fold,development_slots=development,held_out_slots=held_out,**choice))
    package=dict(at=now(),state='global_and_fourfold_partners_frozen_before_new8view_fusion',run_id=handoff['run_id'],
        training_handoff=str(handoff_path),handoff_sha256=sha(handoff_path),candidate_checkpoint_sha256=handoff['checkpoint_sha256'],
        total_new_epochs=handoff['total_stage_new_epochs'],**global_choice,patient_folds=folds,
        selection_scope='one global deployment partner; each held-out fold partner determined exclusively by four development folds; no per-case routing; no270',
        grid='same12 per partner: candidate weight0.5/0.75/0.9 x logit/probability x raw/full',
        max_cached_decode_comparisons=24,models_per_deployment=2,forwards_per_scan=16)
    exclusive(S/'PARTNER_FREEZE.json',package)
    return package

def verify_prepared(handoff_path):
    verify_entry();handoff,_=final_handoff(handoff_path);package=read(S/'PARTNER_FREEZE.json')
    assert package['handoff_sha256']==sha(handoff_path)
    assert package['candidate_checkpoint_sha256']==handoff['checkpoint_sha256']
    partners=[package['selected_partner']]+[f['selected_partner'] for f in package['patient_folds']]
    for partner in partners:assert sha(partner['atoms_ready'])==partner['atoms_ready_sha256']
    return package

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--handoff',required=True);a=p.parse_args()
    try:choose_partner(a.handoff)
    except NoNewBest:pass
