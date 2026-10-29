"""New fixed-algorithm final dispatcher; old numerical entries remain unchanged."""
import argparse
import subprocess
import sys
import numpy as np
import torch
from scripts.race_v2_inference_b_20261029.common import *
from scripts.race_v2_inference_b_20261029 import final270 as legacy
from scripts.race_v2_inference_b_20261029.innovation_b.core import deploy

I=R/'innovation_b';F=R/'final270';CORE=ROOT/'scripts/race_v2_inference_b_20261029/innovation_b/core.py'

def verify_entry():
    legacy.verify_entry()
    for source,digest in read(R/'FINAL270_INNOVATION_ENTRY_FREEZE.json')['source_sha256'].items():assert sha(ROOT/source)==digest,source

def selection(candidate_id=None):
    verify_entry();path=STAGE/'FINAL_TEST_SELECTION.json';root=read(path)
    assert root['schema']=='aflow.race_v2.final_test270.v1' and root['registered_by']=='root'
    assert root['approval_scope']=='final_frozen_test270_only' and not root.get('schema_only_not_authorization',False)
    assert 1<=len(root['candidates'])<=2
    for source,digest in root['entry_source_sha256'].items():assert sha(ROOT/source)==digest,source
    terminal=root['training_terminal']
    for prefix in ('done','budget','result'):assert sha(terminal[prefix+'_path'])==terminal[prefix+'_sha256']
    done=read(terminal['done_path']);budget=read(terminal['budget_path'])
    assert done['state']=='complete' and done['new_epoch']==done['target_new_epochs']==budget['new_epochs']
    for item in root['validation_audit']:assert sha(item['path'])==item['sha256']
    assert root['cohort']['scans']==270 and root['cohort']['subjects']==119
    assert sha(legacy.TEST_MANIFEST)==root['cohort']['manifest_sha256']==legacy.TEST_SHA
    if candidate_id is None:return root,None
    choices=[c for c in root['candidates'] if c['candidate_id']==candidate_id];assert len(choices)==1
    candidate=choices[0];assert candidate['kind']=='fusion' and len(candidate['models'])==2
    assert len({m['model_id'] for m in candidate['models']})==2
    assert set(candidate_id)<=set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-')
    for model in candidate['models']:
        assert sha(model['checkpoint_path'])==model['checkpoint_sha256']
        assert 'latest' not in Path(model['checkpoint_path']).name
        if model['atom_source']=='new_snapshot':
            assert sha(model['handoff_path'])==model['handoff_sha256'];handoff=read(model['handoff_path'])
            assert handoff['strict_cpu_reload'] and handoff['actual_new_epochs']>0
            assert handoff['checkpoint_sha256']==model['checkpoint_sha256'] and handoff['total_stage_new_epochs']==done['new_epoch']
    for item in candidate['validation_sources']:assert sha(item['path'])==item['sha256']
    fusion=candidate['fusion']
    if 'algorithm_recipe' not in fusion:return legacy.final_selection(candidate_id)
    assert fusion['space']=='paired_covariance_shrinkage' and fusion['decoder']=='raw' and fusion['primary_weight']==.75
    recipe=fusion['algorithm_recipe'];assert recipe['kind']=='paired_covariance_shrinkage'
    recipe_path=Path(candidate.get('algorithm_recipe_path',fusion.get('algorithm_recipe_path','')))
    recipe_sha=candidate.get('algorithm_recipe_sha256',fusion.get('algorithm_recipe_sha256'))
    assert recipe_sha and sha(recipe_path)==recipe_sha
    bound=read(recipe_path);assert bound['candidate']==recipe and bound['source_sha256']==sha(CORE)
    assert recipe==read(I/'SELECTED_RECIPE.json')['candidate'],'only one108-selected innovation formula may transfer'
    assert recipe['models']==2 and recipe['forwards_per_scan']==16
    return root,candidate

@torch.inference_mode()
def score(candidate_id):
    root,candidate=selection(candidate_id)
    if 'algorithm_recipe' not in candidate['fusion']:return legacy.score(candidate_id)
    lock=gpu_lock();numeric_mode();torch.cuda.set_per_process_memory_fraction(.9)
    index,cohort,images,labels=legacy.test_data(root,True);sources=[legacy.atoms(m,root) for m in candidate['models']]
    folder=F/'candidates'/candidate_id;records=[];recipe=candidate['fusion']['algorithm_recipe'];digest=sha(STAGE/'FINAL_TEST_SELECTION.json')
    write(folder/'INNOVATION_SCORING_ACCESS.json',dict(at=now(),selection_sha256=digest,recipe=recipe,
        algorithm_recipe_sha256=candidate.get('algorithm_recipe_sha256'),source_sha256=sha(CORE),new_model_forwards=0))
    for slot,row in enumerate(cohort):
        guard();assert sha(STAGE/'FINAL_TEST_SELECTION.json')==digest
        sl=slices(index['geometry'][row['case_id']]);image=np.array(images[slot][sl],np.float32);loaded=[]
        for source in sources:
            views=[]
            for view in range(8):
                entry=source[slot,view];path=Path(entry['path']);assert sha(path)==entry['sha256']
                value=np.load(path,allow_pickle=False);assert value.dtype==np.float32 and value.shape==image.shape and list(value.shape)==entry['shape']
                assert np.isfinite(value).all();views.append(torch.from_numpy(value).cuda())
            loaded.append(views)
        output=deploy(loaded[0],loaded[1],image,recipe);mask=output['mask'];point=output['automatic_point']
        # No labels or case identities reached deploy().
        target=np.array(labels[slot][sl]>0);tp=int((mask&target).sum());pred=int(mask.sum());positive=int(target.sum());den=pred+positive
        path=legacy.FTMP/'candidates'/candidate_id/'predictions'/(row['case_id']+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            saved=np.load(path,allow_pickle=False);old=np.unpackbits(saved['mask_bits'])[:mask.size].reshape(mask.shape)
            assert np.array_equal(old,mask) and saved['point'].tolist()==point
        else:
            temporary=path.with_name(path.name+'.tmp')
            with temporary.open('xb') as handle:np.savez_compressed(handle,mask_bits=np.packbits(mask.ravel()),shape=np.array(mask.shape),point=np.array(point))
            os.replace(temporary,path)
        records.append(dict(case_id=row['case_id'],subject_id=row['subject_id'],source_slot=slot,tp=tp,fp=pred-tp,fn=positive-tp,
            locator_dsc=float(np.float32(2*tp)/np.float32(den)) if den else 1.,automatic_point=point,
            automatic_point_hits_gt=bool(target[tuple(point)]),prediction_path=str(path),prediction_sha256=sha(path),prediction_shape=list(mask.shape)))
        write(folder/'INNOVATION_JOURNAL.json',dict(at=now(),completed=slot+1,total=270,per_case=records,new_model_forwards=0))
    raw=folder/'RAW_TEST270.json'
    write(raw,dict(at=now(),candidate_id=candidate_id,per_case=records,selection_sha256=digest,models=candidate['models'],
        fusion=candidate['fusion'],current_test_search=False,test_tuned_history=True,new_model_forwards=0))
    write(folder/'RESULT.json',dict(at=now(),state='complete',candidate_id=candidate_id,kind='fusion',metrics=legacy.metrics(records),
        models=2,forwards_per_scan=16,raw_sha256=sha(raw),new_model_forwards=0,algorithm_recipe=recipe))

def audit_and_publish(candidate):
    root,_=selection(candidate['candidate_id']);folder=F/'candidates'/candidate['candidate_id'];canonical=folder/'RAW_TEST270.json'
    raw=read(canonical);digest=sha(STAGE/'FINAL_TEST_SELECTION.json');assert raw['selection_sha256']==digest
    index,cohort,_,labels=legacy.test_data(root,True);records=raw['per_case'];audits=[]
    assert [(r['case_id'],r['subject_id']) for r in records]==[(r['case_id'],r['subject_id']) for r in cohort]
    for slot,record in enumerate(records):
        path=Path(record['prediction_path']);assert sha(path)==record['prediction_sha256']
        saved=np.load(path,allow_pickle=False);shape=tuple(saved['shape'].tolist())
        assert list(shape)==record['prediction_shape']==index['geometry'][record['case_id']]['extent']
        mask=np.unpackbits(saved['mask_bits'])[:int(np.prod(shape))].reshape(shape).astype(bool)
        target=np.array(labels[slot][slices(index['geometry'][record['case_id']])]>0)
        tp=int((mask&target).sum());fp=int((mask&~target).sum());fn=int((~mask&target).sum());den=2*tp+fp+fn
        dice=float(np.float32(2*tp)/np.float32(den)) if den else 1.;point=saved['point'].tolist()
        assert (tp,fp,fn,dice,point,bool(target[tuple(point)]))==(record['tp'],record['fp'],record['fn'],record['locator_dsc'],record['automatic_point'],record['automatic_point_hits_gt'])
        audits.append(dict(case_id=record['case_id'],prediction_sha256=sha(path),tp=tp,fp=fp,fn=fn,locator_dsc=dice,point=point))
    receipt=folder/'MASK_AUDIT_INNOVATION_DISPATCH.json'
    write(receipt,dict(at=now(),status='passed',scans=270,selection_sha256=digest,raw_path=str(canonical),raw_sha256=sha(canonical),
        all_masks_counts_points_exact=True,per_case=audits,new_model_forwards=0))
    write(F/'RAW_FUSION.json',dict(raw,candidate_id=candidate['candidate_id'],masks_audit_receipt_path=str(receipt),
        masks_audit_receipt_sha256=sha(receipt),canonical_raw_path=str(canonical),canonical_raw_sha256=sha(canonical)))
    write(F/'RESULT_FUSION.json',dict(read(folder/'RESULT.json'),masks_audit_receipt_path=str(receipt),
        aggregate_raw_path=str(F/'RAW_FUSION.json'),aggregate_raw_sha256=sha(F/'RAW_FUSION.json')))

def watch():
    verify_entry();exclusive(I/'FINAL_WATCH_START.json',dict(at=now(),pid=os.getpid(),worker=1154121,physical_gpu=1,
        no_artificial_load=True,automatic_neural_retry=False))
    while not (STAGE/'FINAL_TEST_SELECTION.json').exists():guard();time.sleep(30)
    root,_=selection();fusion=[c for c in root['candidates'] if c['kind']=='fusion'];assert len(fusion)<=1
    if not fusion:return
    candidate=fusion[0];folder=F/'candidates'/candidate['candidate_id']
    if not (folder/'RAW_TEST270.json').exists():
        if any(m['model_id']=='w32e171' for m in candidate['models']) and not legacy.WREADY.exists():
            subprocess.run([sys.executable,'-u','-m','scripts.race_v2_inference_b_20261029.final270','forward-w32'],check=True)
        while not all(Path(m['atoms_ready_path']).exists() for m in candidate['models']):guard();time.sleep(10)
        subprocess.run([sys.executable,'-u','-m','scripts.race_v2_inference_b_20261029.innovation_b.final270','score',
            '--candidate-id',candidate['candidate_id']],check=True)
    audit_and_publish(candidate)
    write(I/'FINAL_WATCH_STATUS.json',dict(at=now(),state='complete',candidate_id=candidate['candidate_id'],raw_sha256=sha(F/'RAW_FUSION.json')))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('watch','score'));parser.add_argument('--candidate-id');args=parser.parse_args()
    try:
        if args.stage=='watch':watch()
        else:assert args.candidate_id;score(args.candidate_id)
    except BaseException as error:
        import traceback
        write(I/f'FINAL_FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc(),automatic_neural_retry=False));raise
