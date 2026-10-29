"""New final270 controller; old frozen neural producer remains unchanged.

Accepts root's legacy equal8 or fixed local-dependency single entry, and tolerates
the independently owned paired-covariance fusion metadata. No test selection.
"""
import argparse,fcntl,time
import numpy as np
import torch
from scripts.race_v2_inference_innovation_a_20261029.common import *
from scripts.race_v2_inference_innovation_a_20261029.deployment import deploy
from scripts.race_v2_inference_innovation_a_20261029.io_helpers import record_mask,audit_masks
from scripts.race_v2_inference_a_20261029 import final270 as producer
from scripts.race_v2_inference_a_20261029.statistics import summary_ci,paired

FREEZE=BASE/'FINAL270_INNOVATION_ENTRY_FREEZE.json';SELECTION=STAGE/'FINAL_TEST_SELECTION.json'
F=BASE/'final270';OLD=ROOT/'archive/research_records/2026-10-29_race_e164_inference_test270'

def checked(path,digest):
    assert sha(path)==digest,str(path)
    return read(path)

def freeze():
    assert not FREEZE.exists() and not SELECTION.exists()
    sources=dict(read(BASE/'FINAL270_ENTRY_FREEZE.json')['source_sha256'])
    for path in (ROOT/'scripts/race_v2_inference_innovation_a_20261029').glob('*.py'):sources[str(path.relative_to(ROOT))]=sha(path)
    entry=read(R/'DEPLOYMENT_RECIPE.json')['innovation_entry'];generator=torch.Generator().manual_seed(39)
    logits=[torch.randn(19,21,23,generator=generator) for _ in range(8)];image=np.ones((19,21,23),np.float32)
    refined=deploy(logits,image,entry);equal=deploy(logits,image,read(BASE/'SELECTED_RECIPE.json')['candidate'])
    assert refined['mask'].shape==equal['mask'].shape and torch.isfinite(refined['logits']).all()
    gate_path=R/'FINAL270_ADAPTER_CPU_GATE.json';write(gate_path,dict(at=now(),status='passed',legacy_equal_supported=True,
        normalized_innovation_supported=True,new_fusion_space_accepted_by_contract='paired_covariance_shrinkage',
        dynamic_crop=[19,21,23],synthetic_only=True,clinical_arrays_loaded=False,network_forwards=0))
    write(FREEZE,dict(at=now(),source_sha256=sources,legacy_producer_freeze_sha256=sha(BASE/'FINAL270_ENTRY_FREEZE.json'),
        deployment_recipe_path=str(R/'DEPLOYMENT_RECIPE.json'),deployment_recipe_sha256=sha(R/'DEPLOYMENT_RECIPE.json'),
        initial_innovation_result_sha256=sha(R/'RESULT.json'),CPU_gate_path=str(gate_path),CPU_gate_sha256=sha(gate_path),
        final_selection_path=str(SELECTION),new_test_access_by_freeze=False,old_entry_preserved=True,
        new_single_mode='race8_local_dependency',fusion_space_compatibility=['logit','probability','paired_covariance_shrinkage']))

def verify():
    guard();f=read(FREEZE)
    for p,d in f['source_sha256'].items():assert sha(ROOT/p)==d,p
    assert sha(R/'DEPLOYMENT_RECIPE.json')==f['deployment_recipe_sha256']
    assert sha(BASE/'FINAL270_ENTRY_FREEZE.json')==f['legacy_producer_freeze_sha256']
    assert checked(f['CPU_gate_path'],f['CPU_gate_sha256'])['status']=='passed'
    return f

def validate(selection):
    assert not selection.get('schema_only_not_authorization',False)
    assert selection['schema']=='aflow.race_v2.final_test270.v1' and selection['registered_by']=='root'
    assert selection['approval_scope']=='final_frozen_test270_only';f=verify()
    for p,d in selection['entry_source_sha256'].items():assert sha(ROOT/p)==d,p
    for p,d in f['source_sha256'].items():assert selection['entry_source_sha256'].get(p)==d,p
    terminal=selection['training_terminal'];done=checked(terminal['done_path'],terminal['done_sha256'])
    budget=checked(terminal['budget_path'],terminal['budget_sha256']);result=checked(terminal['result_path'],terminal['result_sha256'])
    assert done['state']==result['state']=='complete'
    assert done['new_epoch']==done['target_new_epochs']==budget['new_epochs']==result['actual_new_epochs']
    assert done['new_updates']==budget['new_updates']==result['actual_new_updates']==done['new_epoch']*122
    for evidence in selection['validation_audit']:
        audit=checked(evidence['path'],evidence['sha256'])
        if audit.get('state',audit.get('status'))!='passed':
            assert done['best_new_epoch']==0 and audit['state']=='source_e164_remains_best_no_duplicate_forwards'
            assert audit['training_done_sha256']==terminal['done_sha256'] and audit['new_validation_forwards_registered']==0
    cohort=selection['cohort'];assert (cohort['scans'],cohort['subjects'])==(270,119)
    assert cohort['modality']=='T1ce' and cohort['spacing_mm']==[1,1,1] and cohort['scoring']=='native1mm original crop excludes padding'
    assert Path(cohort['manifest_path'])==producer.TEST_MANIFEST and sha(producer.TEST_MANIFEST)==cohort['manifest_sha256']
    rows_=[json.loads(line) for line in producer.TEST_MANIFEST.read_text().splitlines() if line.strip()]
    assert len(rows_)==270 and len({r['subject_id'] for r in rows_})==119
    candidates=selection['candidates'];assert len(candidates)==2 and {c['kind'] for c in candidates}=={'single','fusion'}
    models={};innovation=read(R/'DEPLOYMENT_RECIPE.json')['innovation_entry'];legacy=read(BASE/'SELECTED_RECIPE.json')['candidate']
    for c in candidates:
        assert len(c['models'])==(1 if c['kind']=='single' else 2) and len({m['model_id'] for m in c['models']})==len(c['models'])
        for evidence in c['validation_sources']:assert sha(evidence['path'])==evidence['sha256']
        if c['kind']=='single':
            assert c['candidate_id']=='single' and c['models'][0]['owner']=='inference_a'
            assert c['single_recipe'] in (innovation,legacy),'only root108-chosen fixed entries'
        else:
            fusion=c['fusion'];assert fusion['space'] in ('logit','probability','paired_covariance_shrinkage')
            assert fusion['decoder'] in ('raw','full') and fusion['primary_weight'] in (.5,.75,.9)
            if fusion['space']=='paired_covariance_shrinkage':
                assert checked(fusion['algorithm_recipe_path'],fusion['algorithm_recipe_sha256'])['candidate']==fusion['algorithm_recipe']
                assert fusion['algorithm_recipe']['kind']=='paired_covariance_shrinkage'
                assert sha(ROOT/'scripts/race_v2_inference_b_20261029/innovation_b/core.py')==fusion['algorithm_source_sha256']
            assert sha(c['full_recipe_path'])==c['full_recipe_sha256']
        for m in c['models']:
            mid=m['model_id'];assert mid in ('race164','race_v2_final','w32e171') and sha(m['checkpoint_path'])==m['checkpoint_sha256']
            if mid in models:assert models[mid]==m
            else:models[mid]=m
            if m['atom_source']=='reuse_race164':
                assert mid=='race164' and m['owner']=='inference_a' and m['checkpoint_sha256']==CHECKPOINT_SHA
                assert sha(m['atoms_ready_path'])==m['atoms_ready_sha256']
                for report in m['historical_atom_reports']:assert sha(report['path'])==report['sha256']
            elif m['atom_source']=='new_snapshot':
                assert mid=='race_v2_final' and m['owner']=='inference_a' and done['best_new_epoch']>0
                assert sha(m['handoff_path'])==m['handoff_sha256'] and sha(m['registration_108_path'])==m['registration_108_sha256']
                registration=read(m['registration_108_path']);assert registration['registered_by']=='root'
                assert registration['handoff_sha256']==m['handoff_sha256'] and registration['handoff_path']==m['handoff_path']
                assert Path(m['atoms_ready_path'])==F/'models/race_v2_final/ATOMS_READY.json'
            else:assert m['atom_source']=='new_w32' and mid=='w32e171' and m['owner']=='inference_b'
    assert sum(m['atom_source']=='new_snapshot' for m in models.values())<=1
    cache_audit=checked(selection['data']['existing_cache_audit_path'],selection['data']['existing_cache_audit_sha256'])
    assert cache_audit['status']=='passed' and cache_audit['all2160_exact']
    return rows_,models

def execute(statistics_only=False):
    assert SELECTION.exists();selection=read(SELECTION);digest=sha(SELECTION);cohort,models=validate(selection)
    receipt=F/'INNOVATION_SELECTION_ACCESS.json'
    if receipt.exists():assert read(receipt)['selection_sha256']==digest
    else:write(receipt,dict(at=now(),selection_sha256=digest,source_freeze_sha256=sha(FREEZE),new_network_yet=False))
    folder,index,canvas,images=producer.data(selection,cohort)
    lock_path=Path('/tmp/aflow_race_v2_20261029/locks/worker1154121_gpu0.lock');lock=lock_path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0';torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    own={}
    for mid,m in models.items():
        if m['owner']!='inference_a':continue
        if m['atom_source']=='reuse_race164' or statistics_only:
            own[mid]=producer.validate_atoms(m,selection,index,cohort)
        else:own[mid]=producer.produce_new_atoms(m,selection,digest,index,canvas,cohort,images)
    # GT is opened only after every owned model has committed its complete raw.
    labels=np.memmap(folder/'labels.u8',np.uint8,'r',shape=(270,*canvas))
    candidate=next(c for c in selection['candidates'] if c['kind']=='single');model=candidate['models'][0];entry=candidate['single_recipe']
    ready=own[model['model_id']];lookup={(a['slot'],a['view']):a for a in ready['atoms']};records=[];started=time.monotonic();gpu_seconds=0.
    torch.cuda.reset_peak_memory_stats()
    for slot,row in enumerate(cohort):
        guard();assert sha(SELECTION)==digest;sl=slices(index['geometry'][row['case_id']]);logits=[]
        for view in range(8):
            atom=lookup[slot,view];assert sha(atom['path'])==atom['sha256'];logits.append(torch.from_numpy(np.load(atom['path'],allow_pickle=False)).cuda())
        first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True);first.record()
        output=deploy(logits,np.array(images[slot][sl],np.float32),entry);last.record();last.synchronize();gpu_seconds+=first.elapsed_time(last)/1000
        target=np.array(labels[slot][sl]>0)
        records.append(record_mask(row,slot,output['mask'].cpu().numpy(),output['automatic_point'],output['base_automatic_point'],target,
            TMP/'final270/predictions'/(row['case_id']+'.npz'),output['diagnostic']))
        write(F/'INNOVATION_DECODE_JOURNAL.json',dict(at=now(),completed=slot+1,total=270,selection_sha256=digest,new_network_forwards_by_decoding=0))
        del logits,output
    f=F/'candidates/single';raw_path=f/'RAW_TEST270.json';write(raw_path,dict(at=now(),candidate_id='single',selection_sha256=digest,
        per_case=records,recipe=entry,checkpoint_sha256=model['checkpoint_sha256'],new_neural_forwards_by_decoding=0,current_test_parameter_search=False))
    audit_path=F/'MASKS_AUDIT_RECEIPT_SINGLE.json';audit_masks(raw_path,audit_path,cohort,index,labels,digest)
    public=F/'RAW_SINGLE.json';write(public,dict(read(raw_path),masks_audit_receipt_path=str(audit_path),masks_audit_receipt_sha256=sha(audit_path),
        canonical_raw_path=str(raw_path),canonical_raw_sha256=sha(raw_path)))
    metrics=summary_ci(records);comparisons={};worse={}
    for name in ('raw','enhanced'):
        reference=read(OLD/f'RAW_TEST270_{name}.json')['per_case'];comparisons[name]=paired(records,reference);right={r['case_id']:r for r in reference}
        worse[name]=sorted([dict(case_id=r['case_id'],subject_id=r['subject_id'],delta=r['locator_dsc']-right[r['case_id']]['locator_dsc']) for r in records if r['locator_dsc']<right[r['case_id']]['locator_dsc']],key=lambda r:r['delta'])
    result=dict(at=now(),state='complete',candidate_id='single',kind='single',metrics=metrics,Point_refined=metrics['point_hits'],
        Point_base=sum(r['base_automatic_point_hits_gt'] for r in records),comparisons_vs_historical_RACE164=comparisons,worse_cases=worse,
        selection_sha256=digest,recipe=entry,checkpoint_sha256=model['checkpoint_sha256'],
        aggregate_raw_path=str(public),aggregate_raw_sha256=sha(public),raw_sha256=sha(raw_path),
        masks_audit_receipt_path=str(audit_path),masks_audit_receipt_sha256=sha(audit_path),
        forward_cost_per_case=entry['neural_case_forwards'],gpu_algorithm_event_seconds=gpu_seconds,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,wall_seconds=time.monotonic()-started,
        source_freeze_sha256=sha(FREEZE),test_tuned_history=True,independent_confirmation=False,current_test_parameter_search=False)
    write(F/'RESULT_SINGLE.json',result);write(f/'RESULT.json',result)
    write(F/'INNOVATION_WATCH_STATUS.json',dict(at=now(),state='complete',selection_sha256=digest,raw_sha256=sha(public),new_parameter_search=False))

def main(watch,statistics_only):
    verify()
    if watch:
        while not SELECTION.exists():guard();time.sleep(30)
    execute(statistics_only)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--watch',action='store_true');p.add_argument('--statistics-only',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:
        try:main(a.watch,a.statistics_only)
        except BaseException as error:
            import traceback
            write(F/f'INNOVATION_FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc(),automatic_neural_retry=False));raise
