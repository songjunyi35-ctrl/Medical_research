"""Root-only final dispatcher for all frozen single-input RACE variants.

No test policy is chosen here. Ordinary RACE atoms remain shared with B, whose
global/covariance/contralateral fusion is independently scored on GPU1.
"""
import argparse,fcntl,json,time
import numpy as np
import torch
from scripts.race_v2_inference_soft_context_a_20261029.common import *
from scripts.race_v2_inference_soft_context_a_20261029.cache import generate,seal,validate_ready
from scripts.race_v2_inference_normalization_a_20261029 import final270 as normalization_dispatch
from scripts.race_v2_inference_soft_context_a_20261029.deployment import deploy,gate_network
from scripts.followup_test270_common import numeric_mode
from scripts.race_v2_inference_innovation_a_20261029.io_helpers import record_mask,audit_masks
from scripts.race_v2_inference_a_20261029 import final270 as producer
from scripts.race_v2_inference_a_20261029.predictor import load_model
from scripts.race_v2_inference_a_20261029.snapshot108 import load_handoff
from scripts.race_v2_inference_a_20261029.statistics import summary_ci,paired

FREEZE=BASE/'FINAL270_SOFT_CONTEXT_ENTRY_FREEZE.json';SELECTION=STAGE/'FINAL_TEST_SELECTION.json'
F=BASE/'final270';RECIPE=R/'SOFT_CONTEXT_DEPLOYMENT_RECIPE.json'
HEALTH=ROOT/'archive/research_records/2026-10-29_race_v2_inference_normalization_a/NORMALIZATION_DEPLOYMENT_RECIPE.json'
LOCAL=ROOT/'archive/research_records/2026-10-29_v2_inference_innovation_a'
OLD=ROOT/'archive/research_records/2026-10-29_race_e164_inference_test270'
FUSION_CORES={
    'paired_covariance_shrinkage':'scripts/race_v2_inference_b_20261029/innovation_b/core.py',
    'contralateral_image_evidence':'scripts/race_v2_inference_b_20261029/image_conflict_b/core.py',
    'image_appearance_unary':'scripts/race_v2_inference_b_20261029/appearance_b/core.py'}

def checked(path,digest):
    assert sha(path)==digest,str(path)
    return read(path)

def freeze():
    assert not FREEZE.exists() and not SELECTION.exists()
    sources=dict(read(BASE/'FINAL270_NORMALIZATION_ENTRY_FREEZE_V2.json')['source_sha256'])
    sources.update(read(R/'FINAL_SNAPSHOT_TRANSFER_FREEZE.json')['source_sha256'])
    for p in (ROOT/'scripts/race_v2_inference_soft_context_a_20261029').glob('*.py'):sources[str(p.relative_to(ROOT))]=sha(p)
    for name in ('final270.py','cache.py','deployment.py','algorithm.py','transfer.py','queue_barrier.py'):
        assert f'scripts/race_v2_inference_soft_context_a_20261029/{name}' in sources
    context=read(RECIPE)['innovation_entry'];normal=read(HEALTH)['innovation_entry'];legacy=read(BASE/'SELECTED_RECIPE.json')['candidate']
    local=read(LOCAL/'DEPLOYMENT_RECIPE.json')['innovation_entry'];rng=torch.Generator().manual_seed(353)
    logits=[torch.randn(19,21,23,generator=rng) for _ in range(8)];image=np.ones((19,21,23),np.float32)
    for entry in (context,normal,legacy,local):
        output=deploy(logits,image,entry);assert output['mask'].shape==image.shape and bool(output['logits'].isfinite().all())
    gate=R/'FINAL270_CPU_GATE.json';write(gate,dict(at=now(),status='passed',dynamic_crop=[19,21,23],
        legacy_equal_and_local_supported=True,health_normalization_supported=True,soft_context_supported=True,
        fusion_spaces=['logit','probability',*FUSION_CORES],clinical_arrays_accessed=False,new_network_forwards=0))
    write(FREEZE,dict(at=now(),source_sha256=sources,normalization_recipe_path=str(RECIPE),normalization_recipe_sha256=sha(RECIPE),
        local_recipe_sha256=sha(LOCAL/'DEPLOYMENT_RECIPE.json'),legacy_recipe_sha256=sha(BASE/'SELECTED_RECIPE.json'),
        health_recipe_sha256=sha(HEALTH),queue_barrier_freeze_sha256=sha(R/'SOURCE_QUEUE_BARRIER_FREEZE.json'),
        CPU_gate_path=str(gate),CPU_gate_sha256=sha(gate),old_entry_preserved=True,
        final_selection_path=str(SELECTION),root_selection_only=True,maximum_final_candidates=2,
        new270_access=False,new270_forward=False,no_automatic_neural_retry=True))

def verify_entry():
    guard();freeze_=read(FREEZE)
    for p,digest in freeze_['source_sha256'].items():assert sha(ROOT/p)==digest,p
    assert sha(RECIPE)==freeze_['normalization_recipe_sha256']
    assert sha(HEALTH)==freeze_['health_recipe_sha256']
    assert sha(R/'SOURCE_QUEUE_BARRIER_FREEZE.json')==freeze_['queue_barrier_freeze_sha256']
    assert sha(LOCAL/'DEPLOYMENT_RECIPE.json')==freeze_['local_recipe_sha256']
    assert sha(BASE/'SELECTED_RECIPE.json')==freeze_['legacy_recipe_sha256']
    assert checked(freeze_['CPU_gate_path'],freeze_['CPU_gate_sha256'])['status']=='passed'
    return freeze_

def validate(selection):
    assert not selection.get('schema_only_not_authorization',False)
    assert selection['schema']=='aflow.race_v2.final_test270.v1' and selection['registered_by']=='root'
    assert selection['approval_scope']=='final_frozen_test270_only';freeze_=verify_entry()
    for p,digest in selection['entry_source_sha256'].items():assert sha(ROOT/p)==digest,p
    for p,digest in freeze_['source_sha256'].items():assert selection['entry_source_sha256'].get(p)==digest,p
    terminal=selection['training_terminal'];done=checked(terminal['done_path'],terminal['done_sha256'])
    budget=checked(terminal['budget_path'],terminal['budget_sha256']);result=checked(terminal['result_path'],terminal['result_sha256'])
    assert done['state']==result['state']=='complete'
    assert done['new_epoch']==done['target_new_epochs']==budget['new_epochs']==result['actual_new_epochs']
    assert done['new_updates']==done['new_epoch']*122==budget['new_updates']==result['actual_new_updates']
    assert done['best_new_epoch']==result['best_new_epoch']
    assert selection['validation_audit']
    for evidence in selection['validation_audit']:
        audit=checked(evidence['path'],evidence['sha256'])
        if audit.get('status',audit.get('state'))!='passed':
            assert done['best_new_epoch']==0 and audit['state']=='source_e164_remains_best_no_duplicate_forwards'
            assert audit['training_done_sha256']==terminal['done_sha256']
            assert audit.get('new_validation_forwards_registered',audit.get('new108_forwards'))==0
    cohort=selection['cohort'];assert (cohort['scans'],cohort['subjects'])==(270,119)
    assert cohort['modality']=='T1ce' and cohort['spacing_mm']==[1,1,1]
    assert cohort['scoring']=='native1mm original crop excludes padding'
    assert Path(cohort['manifest_path'])==producer.TEST_MANIFEST and sha(producer.TEST_MANIFEST)==cohort['manifest_sha256']
    cohort_rows=[json.loads(line) for line in producer.TEST_MANIFEST.read_text().splitlines() if line.strip()]
    assert len(cohort_rows)==len({r['case_id'] for r in cohort_rows})==270 and len({r['subject_id'] for r in cohort_rows})==119
    candidates=selection['candidates'];assert 1<=len(candidates)<=2
    assert len({c['candidate_id'] for c in candidates})==len(candidates)
    assert sum(c['kind']=='single' for c in candidates)==1 and sum(c['kind']=='fusion' for c in candidates)<=1
    context=read(RECIPE)['innovation_entry'];normal=read(HEALTH)['innovation_entry'];legacy=read(BASE/'SELECTED_RECIPE.json')['candidate'];local=read(LOCAL/'DEPLOYMENT_RECIPE.json')['innovation_entry']
    models={}
    for candidate in candidates:
        assert len(candidate['models'])==(1 if candidate['kind']=='single' else 2)
        assert len({m['checkpoint_sha256'] for m in candidate['models']})==len(candidate['models']),'weight identities define model count'
        for evidence in candidate['validation_sources']:assert sha(evidence['path'])==evidence['sha256']
        if candidate['kind']=='single':
            assert candidate['candidate_id']=='single' and candidate['models'][0]['owner']=='inference_a'
            assert candidate['single_recipe'] in (context,normal,legacy,local)
            mode=candidate['single_recipe']['mode'];source=candidate['models'][0]['atom_source']
            assert (mode=='race8_seed_health_normalization')==(source=='seed_health_normalization')
            assert (mode=='race8_seed_soft_context')==(source=='seed_soft_context')
        else:
            fusion=candidate['fusion'];assert fusion['space'] in ('logit','probability',*FUSION_CORES)
            assert fusion['decoder'] in ('raw','full') and fusion['primary_weight'] in (.5,.75,.9)
            assert sha(candidate['full_recipe_path'])==candidate['full_recipe_sha256']
            if fusion['space'] in FUSION_CORES:
                assert checked(fusion['algorithm_recipe_path'],fusion['algorithm_recipe_sha256'])['candidate']==fusion['algorithm_recipe']
                assert fusion['algorithm_recipe']['kind']==fusion['space']
                assert sha(ROOT/FUSION_CORES[fusion['space']])==fusion['algorithm_source_sha256']
        for model in candidate['models']:
            mid=model['model_id'];assert mid in ('race164','race_v2_final','w32e171','race164_healthnorm','race_v2_final_healthnorm','race164_softcontext','race_v2_final_softcontext','w32e171_guided')
            assert sha(model['checkpoint_path'])==model['checkpoint_sha256']
            assert Path(model['checkpoint_path']).name!='latest.pth'
            if mid in models:assert models[mid]==model
            else:models[mid]=model
            source=model['atom_source']
            if source in ('seed_health_normalization','seed_soft_context'):
                assert model['owner']=='inference_a' and model['base_model_id'] in ('race164','race_v2_final')
                expected_path=HEALTH if source=='seed_health_normalization' else RECIPE
                expected_recipe=normal if source=='seed_health_normalization' else context
                suffix='_healthnorm' if source=='seed_health_normalization' else '_softcontext'
                assert mid==model['base_model_id']+suffix and model['input_recipe']==expected_recipe
                assert Path(model['input_recipe_path'])==expected_path and sha(expected_path)==model['input_recipe_sha256']
                assert Path(model['atoms_ready_path'])==F/'models'/mid/'ATOMS_READY.json'
                if model['base_model_id']=='race164':
                    assert model['checkpoint_sha256']==CHECKPOINT_SHA
                    assert Path(model['seed_atoms_ready_path'])==F/'reuse_race164/ATOMS_READY.json'
                    assert sha(model['seed_atoms_ready_path'])==model['seed_atoms_ready_sha256']
                else:validate_new_registration(model,done)
            elif source=='reuse_race164':
                assert mid=='race164' and model['owner']=='inference_a' and model['checkpoint_sha256']==CHECKPOINT_SHA
                assert sha(model['atoms_ready_path'])==model['atoms_ready_sha256']
                for report in model['historical_atom_reports']:assert sha(report['path'])==report['sha256']
            elif source=='new_snapshot':
                assert mid=='race_v2_final' and model['owner']=='inference_a';validate_new_registration(model,done)
                assert Path(model['atoms_ready_path'])==F/'models/race_v2_final/ATOMS_READY.json'
            elif source=='guided_input_w32':
                assert mid=='w32e171_guided' and model['owner']=='inference_b' and model['base_model_id']=='w32e171'
                assert Path(model['input_recipe_path'])==ROOT/'archive/research_records/2026-10-29_race_v2_inference_b/edge_input_b/INPUT_DEPLOYMENT_RECIPE.json'
                source_recipe=checked(model['input_recipe_path'],model['input_recipe_sha256'])
                assert model['input_recipe']==source_recipe['input_recipe']
                assert model['input_recipe']['kernel']==5 and model['input_recipe']['epsilon']==.04 and model['input_recipe']['beta'] in (.25,.5)
                assert model['checkpoint_sha256']==source_recipe['checkpoint_sha256']=='02a60b2d37da0d8de8018c221e1c00566106bd21d0d713b95ac3d9f94d87fade'
                assert source_recipe['source_sha256']==sha(ROOT/'scripts/race_v2_inference_b_20261029/edge_input_b/core.py')
            else:assert source=='new_w32' and mid=='w32e171' and model['owner']=='inference_b'
    audit=checked(selection['data']['existing_cache_audit_path'],selection['data']['existing_cache_audit_sha256'])
    assert audit['status']=='passed' and audit['all2160_exact']
    return cohort_rows,models

def validate_new_registration(model,done):
    assert done['best_new_epoch']>0
    handoff=checked(model['handoff_path'],model['handoff_sha256']);job=checked(model['registration_108_path'],model['registration_108_sha256'])
    assert job['registered_by']=='root' and job['approval_scope']=='final_completed_training_snapshot108_only'
    assert job['handoff_path']==model['handoff_path'] and job['handoff_sha256']==model['handoff_sha256']
    for key in ('checkpoint_path','checkpoint_sha256','architecture','construction'):assert handoff[key]==model[key],key
    assert handoff['actual_new_epochs']>0 and handoff['total_stage_new_epochs']==done['new_epoch']

def cache_metadata(model):
    return {key:model[key] for key in ('model_id','base_model_id','checkpoint_path','checkpoint_sha256','architecture','construction','input_recipe','input_recipe_sha256','atoms_ready_path')}

def normalized_atoms(model,selection,digest,index,canvas,cohort,images,own,statistics_only):
    if model['atom_source']=='seed_health_normalization':
        return normalization_dispatch.normalized_atoms(model,selection,digest,index,canvas,cohort,images,own,statistics_only)
    metadata=cache_metadata(model);folder=F/'models'/model['model_id'];manifest=selection['cohort']['manifest_sha256'];idx=selection['data']['index_sha256']
    if Path(model['atoms_ready_path']).exists():return validate_ready(metadata,index,cohort,manifest,idx)
    if (folder/'MODEL_PREACCESS.json').exists():
        seal(metadata,index,cohort,manifest,idx,folder,digest)
        return validate_ready(metadata,index,cohort,manifest,idx)
    assert not statistics_only,'statistics-only recovery requires complete cached raw'
    base=model['base_model_id'];source=None
    if base=='race164':source=Path(model['seed_atoms_ready_path']);network=load_model()
    else:
        if base in own:source=Path(next(m['atoms_ready_path'] for m in selection_models(selection) if m['model_id']==base))
        handoff,network=load_handoff(read(model['registration_108_path']))
        for key in ('checkpoint_path','checkpoint_sha256','architecture','construction'):assert handoff[key]==model[key]
        network=network.cuda()
    write(folder/'GPU_SYNTHETIC_GATE.json',dict(at=now(),**gate_network(network,canvas,6)))
    ready=generate(network,metadata,index,cohort,images,manifest,idx,folder,TMP/'final270/models'/model['model_id'],source,digest)
    del network;torch.cuda.empty_cache();return ready

def selection_models(selection):
    seen={}
    for candidate in selection['candidates']:
        for model in candidate['models']:seen[model['model_id']]=model
    return list(seen.values())

def execute(statistics_only=False):
    selection=read(SELECTION);digest=sha(SELECTION);cohort,models=validate(selection)
    receipt=F/'SOFT_CONTEXT_SELECTION_ACCESS.json'
    if receipt.exists():assert read(receipt)['selection_sha256']==digest
    else:write(receipt,dict(at=now(),selection_sha256=digest,source_freeze_sha256=sha(FREEZE),new_network_yet=False))
    folder,index,canvas,images=producer.data(selection,cohort)
    lock=LOCK.open('a')
    while True:
        guard()
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:time.sleep(2)
    assert __import__('os').environ.get('CUDA_VISIBLE_DEVICES')=='0';torch.set_num_threads(2)
    own={}
    # Produce ordinary sources first so normalization can share their identity seed.
    for mid,model in models.items():
        if model['owner']!='inference_a' or model['atom_source'] in ('seed_health_normalization','seed_soft_context'):continue
        if model['atom_source']=='reuse_race164' or statistics_only:own[mid]=producer.validate_atoms(model,selection,index,cohort)
        else:own[mid]=producer.produce_new_atoms(model,selection,digest,index,canvas,cohort,images)
    for mid,model in models.items():
        if model['atom_source'] in ('seed_health_normalization','seed_soft_context'):own[mid]=normalized_atoms(model,selection,digest,index,canvas,cohort,images,own,statistics_only)
    # Labels enter scoring only after every owned selected raw is complete.
    labels=np.memmap(folder/'labels.u8',np.uint8,'r',shape=(270,*canvas))
    candidate=next(c for c in selection['candidates'] if c['kind']=='single');model=candidate['models'][0];entry=candidate['single_recipe']
    lookup={(a['slot'],a['view']):a for a in own[model['model_id']]['atoms']};records=[];started=time.monotonic();gpu_seconds=0.
    torch.cuda.reset_peak_memory_stats()
    for slot,row in enumerate(cohort):
        guard();assert sha(SELECTION)==digest;g=index['geometry'][row['case_id']];zs=[]
        for view in range(8):
            atom=lookup[slot,view];assert sha(atom['path'])==atom['sha256'];zs.append(torch.from_numpy(np.load(atom['path'],allow_pickle=False)).cuda())
        first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True);first.record()
        output=deploy(zs,np.asarray(images[slot][slices(g)]),entry);last.record();last.synchronize();gpu_seconds+=first.elapsed_time(last)/1000.
        target=np.array(labels[slot][slices(g)]>0)
        records.append(record_mask(row,slot,output['mask'].cpu().numpy(),output['automatic_point'],output['base_automatic_point'],target,
            TMP/'final270/predictions'/f'{slot:03}.npz',output['diagnostic']))
        write(F/'SOFT_CONTEXT_DECODE_JOURNAL.json',dict(at=now(),completed=slot+1,total=270,selection_sha256=digest,new_network_forwards_by_decoding=0))
        del zs,output
    raw=F/'candidates/single/RAW_TEST270.json';write(raw,dict(at=now(),candidate_id='single',selection_sha256=digest,
        per_case=records,recipe=entry,checkpoint_sha256=model['checkpoint_sha256'],new_neural_forwards_by_decoding=0,current_test_parameter_search=False))
    audit=F/'MASKS_AUDIT_RECEIPT_SINGLE.json';audit_masks(raw,audit,cohort,index,labels,digest)
    public=F/'RAW_SINGLE.json';write(public,dict(read(raw),masks_audit_receipt_path=str(audit),masks_audit_receipt_sha256=sha(audit),
        canonical_raw_path=str(raw),canonical_raw_sha256=sha(raw)))
    metrics=summary_ci(records);comparisons={};worse={}
    for name in ('raw','enhanced'):
        reference=read(OLD/f'RAW_TEST270_{name}.json')['per_case'];comparisons[name]=paired(records,reference)
        lookup_ref={r['case_id']:r for r in reference}
        worse[name]=sorted([dict(case_id=r['case_id'],subject_id=r['subject_id'],delta=r['locator_dsc']-lookup_ref[r['case_id']]['locator_dsc']) for r in records if r['locator_dsc']<lookup_ref[r['case_id']]['locator_dsc']],key=lambda x:x['delta'])
    result=dict(at=now(),state='complete',candidate_id='single',kind='single',metrics=metrics,Point_refined=metrics['point_hits'],
        Point_base=sum(r['base_automatic_point_hits_gt'] for r in records),
        Point_auxiliary_rule='original identity seed argmax' if entry['mode'] in ('race8_seed_health_normalization','race8_seed_soft_context') else 'equal8 argmax',
        comparisons_vs_historical_RACE164=comparisons,worse_cases=worse,selection_sha256=digest,recipe=entry,
        checkpoint_sha256=model['checkpoint_sha256'],aggregate_raw_path=str(public),aggregate_raw_sha256=sha(public),raw_sha256=sha(raw),
        masks_audit_receipt_path=str(audit),masks_audit_receipt_sha256=sha(audit),
        forward_cost_per_case=8,actual_new_forwards_by_owned_producers=sum(r.get('new_model_forwards',r.get('actual_new_forwards',0)) for r in own.values()),
        model_count=1,source_freeze_sha256=sha(FREEZE),gpu_algorithm_event_seconds=gpu_seconds,
        wall_seconds=time.monotonic()-started,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        test_tuned_history=True,independent_confirmation=False,current_test_parameter_search=False)
    write(F/'RESULT_SINGLE.json',result);write(F/'candidates/single/RESULT.json',result)
    write(F/'SOFT_CONTEXT_WATCH_STATUS.json',dict(at=now(),state='complete',selection_sha256=digest,raw_sha256=sha(public)))

def main(watch,statistics_only):
    numeric_mode()
    write(R/'NUMERIC_MODE_RECEIPT_final270.json',dict(at=now(),pid=__import__('os').getpid(),same_process=True,
        numeric_source_sha256=sha(ROOT/'scripts/followup_test270_common.py'),
        backend_mode=dict(cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)))
    verify_entry()
    if watch:
        while not SELECTION.exists():guard();time.sleep(20)
    execute(statistics_only)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--watch',action='store_true');p.add_argument('--statistics-only',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:
        try:main(a.watch,a.statistics_only)
        except BaseException as error:
            import traceback
            write(F/f'SOFT_CONTEXT_FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc(),automatic_neural_retry=False));raise
