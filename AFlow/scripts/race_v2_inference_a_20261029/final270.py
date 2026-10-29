"""Final frozen single-model270 executor and shared RACE atom producer.

Only the real root FINAL_TEST_SELECTION.json authorizes execution. Its sibling
SCHEMA is documentation. All new model atoms are committed before labels open;
decoder APIs accept only aligned logits and the input image. No neural retry.
"""
import argparse,fcntl,time
import numpy as np
import torch
from scripts.race_v2_inference_a_20261029.common import *
from scripts.race_v2_inference_a_20261029.predictor import forward_view,decode_all
from scripts.race_v2_inference_a_20261029.snapshot108 import load_handoff
from scripts.race_v2_inference_a_20261029.statistics import summary_ci,paired

F=R/'final270';ENTRY_FREEZE=R/'FINAL270_ENTRY_FREEZE.json'
SELECTION=STAGE/'FINAL_TEST_SELECTION.json'
SCHEMA='aflow.race_v2.final_test270.v1'
TEST_MANIFEST=ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/sealed_test/test_cases_t1ce.jsonl'

def checked_json(path,digest):
    assert sha(path)==digest,str(path)
    return read(path)

def freeze():
    assert not ENTRY_FREEZE.exists()
    gate=cpu_check();write(R/'FINAL270_CPU_CHECK.json',gate)
    sources=dict(read(R/'EXECUTION_FREEZE.json')['source_sha256'])
    sources.update(read(R/'SNAPSHOT108_FREEZE.json')['source_sha256'])
    for path in (Path(__file__),ROOT/'scripts/race_v2_inference_a_20261029/prepare270_reuse_manifest.py'):
        sources[str(path.relative_to(ROOT))]=sha(path)
    write(ENTRY_FREEZE,dict(at=now(),source_sha256=sources,cpu_gate_sha256=sha(R/'FINAL270_CPU_CHECK.json'),root_selection_only=str(SELECTION),schema=SCHEMA,candidates_maximum=2,models_new_owned_by_A_maximum=1,new_model_case_forwards=2160,views=list(range(8)),dynamic_canvas_from_split_index=True,synthetic_batch=8,synthetic_tail=6,no_automatic_neural_retry=True,no_root_selection_exists_at_freeze=not SELECTION.exists(),new270_scoring=False,new270_network=False))

def verify_entry():
    guard();freeze=read(ENTRY_FREEZE)
    for path,digest in freeze['source_sha256'].items():assert sha(ROOT/path)==digest,path
    assert sha(R/'FINAL270_CPU_CHECK.json')==freeze['cpu_gate_sha256']
    assert read(R/'FINAL270_CPU_CHECK.json')['status']=='passed'
    return freeze

def validate_selection(selection):
    assert selection.get('schema_only_not_authorization') is not True,'schema is not authorization'
    assert selection['schema']==SCHEMA and selection['registered_by']=='root'
    assert selection['approval_scope']=='final_frozen_test270_only'
    assert selection['at']
    freeze=verify_entry()
    assert selection['entry_source_sha256']
    for path,digest in selection['entry_source_sha256'].items():assert sha(ROOT/path)==digest,path
    for path,digest in freeze['source_sha256'].items():assert selection['entry_source_sha256'].get(path)==digest,path
    terminal=selection['training_terminal']
    done=checked_json(terminal['done_path'],terminal['done_sha256'])
    budget=checked_json(terminal['budget_path'],terminal['budget_sha256'])
    result=checked_json(terminal['result_path'],terminal['result_sha256'])
    assert done['state']=='complete' and result['state']=='complete'
    assert done['new_epoch']==done['target_new_epochs']==budget['new_epochs']==result['actual_new_epochs']
    assert done['new_updates']==done['new_epoch']*122==budget['new_updates']==result['actual_new_updates']
    assert done['best_new_epoch']==result['best_new_epoch']
    assert selection['validation_audit']
    for evidence in selection['validation_audit']:
        audit=checked_json(evidence['path'],evidence['sha256'])
        if audit.get('state',audit.get('status'))!='passed':
            assert done['best_new_epoch']==0 and audit['state']=='source_e164_remains_best_no_duplicate_forwards'
            assert audit['training_done_sha256']==terminal['done_sha256'] and audit['new_validation_forwards_registered']==0
    cohort=selection['cohort'];assert cohort['scans']==270 and cohort['subjects']==119
    assert cohort['modality']=='T1ce' and cohort['spacing_mm']==[1,1,1]
    assert cohort['scoring']=='native1mm original crop excludes padding'
    assert Path(cohort['manifest_path'])==TEST_MANIFEST and sha(TEST_MANIFEST)==cohort['manifest_sha256']
    rows_=[json.loads(line) for line in TEST_MANIFEST.read_text().splitlines() if line.strip()]
    assert len(rows_)==len({r['case_id'] for r in rows_})==270 and len({r['subject_id'] for r in rows_})==119
    candidates=selection['candidates'];assert 1<=len(candidates)<=2
    assert len({c['candidate_id'] for c in candidates})==len(candidates)
    assert sum(c['kind']=='single' for c in candidates)<=1 and sum(c['kind']=='fusion' for c in candidates)<=1
    models={}
    for c in candidates:
        assert c['kind'] in ('single','fusion')
        assert len(c['models'])==(1 if c['kind']=='single' else 2)
        assert len({m['model_id'] for m in c['models']})==len(c['models'])
        assert 8*len(c['models'])<=16
        assert c['validation_sources']
        for evidence in c['validation_sources']:assert sha(evidence['path'])==evidence['sha256']
        if c['kind']=='single':
            assert c['models'][0]['owner']=='inference_a'
            assert c['single_recipe']==read(R/'SELECTED_RECIPE.json')['candidate'],'fixed108 recipe transfer only'
        else:
            fusion=c['fusion'];assert fusion['primary_weight'] in (.5,.75,.9)
            assert fusion['space'] in ('logit','probability') and fusion['decoder'] in ('raw','full')
            assert sha(c['full_recipe_path'])==c['full_recipe_sha256']
        for m in c['models']:
            mid=m['model_id'];assert mid in ('race164','race_v2_final','w32e171')
            assert sha(m['checkpoint_path'])==m['checkpoint_sha256']
            assert m['atom_source'] in ('reuse_race164','new_snapshot','new_w32')
            if mid in models:
                assert models[mid]==m,'same model must have identical shared metadata'
            else:models[mid]=m
            if m['atom_source']=='reuse_race164':
                assert mid=='race164' and m['owner']=='inference_a' and m['checkpoint_sha256']==CHECKPOINT_SHA
                assert m['historical_atom_reports']
                for evidence in m['historical_atom_reports']:assert sha(evidence['path'])==evidence['sha256']
                assert m.get('atoms_ready_sha256') and sha(m['atoms_ready_path'])==m['atoms_ready_sha256']
            elif m['atom_source']=='new_snapshot':
                assert mid=='race_v2_final' and m['owner']=='inference_a' and done['best_new_epoch']>0
                assert sha(m['handoff_path'])==m['handoff_sha256']
                assert sha(m['registration_108_path'])==m['registration_108_sha256']
                registration=read(m['registration_108_path'])
                assert registration['registered_by']=='root' and registration['approval_scope']=='final_completed_training_snapshot108_only'
                assert registration['handoff_path']==m['handoff_path'] and registration['handoff_sha256']==m['handoff_sha256']
                assert Path(m['atoms_ready_path'])==F/'models'/mid/'ATOMS_READY.json'
            else:assert mid=='w32e171' and m['owner']=='inference_b'
    assert sum(m['owner']=='inference_a' and m['atom_source']=='new_snapshot' for m in models.values())<=1
    audit=checked_json(selection['data']['existing_cache_audit_path'],selection['data']['existing_cache_audit_sha256'])
    assert audit['status']=='passed' and audit['all2160_exact']
    return rows_,models

def commit_selection():
    assert SELECTION.exists(),'root FINAL_TEST_SELECTION.json is absent'
    digest=sha(SELECTION);selection=read(SELECTION);rows_,models=validate_selection(selection)
    F.mkdir(parents=True,exist_ok=True);receipt=F/'SELECTION_ACCESS.json'
    if receipt.exists():assert read(receipt)['selection_sha256']==digest,'cannot change frozen selection after access'
    else:
        with receipt.open('x') as out:json.dump(dict(at=now(),selection_path=str(SELECTION),selection_sha256=digest,entry_freeze_sha256=sha(ENTRY_FREEZE),network_yet=False,scoring_yet=False),out,ensure_ascii=False,indent=2)
    return selection,digest,rows_,models

def data(selection,cohort):
    folder=Path(selection['data']['native_cache']);assert sha(folder/'index.json')==selection['data']['index_sha256']
    index=read(folder/'index.json');canvas=tuple(index['canvas'])
    assert list(canvas)==selection['cohort']['canvas'] and all(n%8==0 for n in canvas)
    assert index['count']==270 and index['case_ids']==[r['case_id'] for r in cohort]
    expected=selection['data']['payload_sha256']
    assert (folder/'images.f32').stat().st_size==270*int(np.prod(canvas))*4
    assert (folder/'labels.u8').stat().st_size==270*int(np.prod(canvas))
    # Hashing confirms historical payload identity; labels are not decoded yet.
    assert sha(folder/'images.f32')==expected['native_image']
    assert sha(folder/'labels.u8')==expected['native_label']
    images=np.memmap(folder/'images.f32',np.float32,'r',shape=(270,*canvas))
    return folder,index,canvas,images

def validate_atoms(model,selection,index,cohort,actual_hashes=True):
    ready=read(model['atoms_ready_path']);assert ready['model_id']==model['model_id']
    assert ready['checkpoint_sha256']==model['checkpoint_sha256']
    assert ready['manifest_sha256']==selection['cohort']['manifest_sha256'] and ready['index_sha256']==selection['data']['index_sha256']
    assert ready['case_forwards']==2160 and len(ready['atoms'])==2160
    assert {(a['slot'],a['view']) for a in ready['atoms']}=={(s,v) for s in range(270) for v in range(8)}
    assert ready.get('canvas',index['canvas'])==index['canvas']
    for atom in ready['atoms']:
        assert atom['shape']==index['geometry'][cohort[atom['slot']]['case_id']]['extent']
        if actual_hashes:
            p=Path(atom['path']);assert sha(p)==atom['sha256']
            value=np.load(p,mmap_mode='r',allow_pickle=False)
            assert list(value.shape)==atom['shape'] and value.dtype==np.float32 and np.isfinite(value).all()
    return ready

def ready_from_journal(model,selection,digest,index,canvas,cohort,folder):
    journal=read(folder/'FORWARD_JOURNAL.json');atoms=journal['atoms']
    assert journal['selection_sha256']==digest and journal['checkpoint_sha256']==model['checkpoint_sha256']
    assert len(atoms)==2160 and {(a['slot'],a['view']) for a in atoms}=={(s,v) for s in range(270) for v in range(8)},'incomplete raw: no automatic neural retry'
    for atom in atoms:
        assert sha(atom['path'])==atom['sha256']
        value=np.load(atom['path'],mmap_mode='r',allow_pickle=False)
        assert list(value.shape)==atom['shape']==index['geometry'][cohort[atom['slot']]['case_id']]['extent'] and np.isfinite(value).all()
    assert sha(model['checkpoint_path'])==model['checkpoint_sha256']
    ready=dict(at=now(),status='complete',model_id=model['model_id'],checkpoint_path=model['checkpoint_path'],checkpoint_sha256=model['checkpoint_sha256'],architecture=model['architecture'],construction=model['construction'],manifest_sha256=selection['cohort']['manifest_sha256'],index_sha256=selection['data']['index_sha256'],cache_path=selection['data']['native_cache'],canvas=list(canvas),case_forwards=2160,new_model_forwards=2160,atoms=atoms,root_selection_sha256=digest,entry_freeze_sha256=sha(ENTRY_FREEZE),geometry_contract='R_view=T@R@T; flip support with image; invert logits before crop',network_seconds=journal.get('seconds'),peak_reserved_gib=journal.get('peak_reserved_gib'),raw_complete_statistics_only_recovery=True)
    write(model['atoms_ready_path'],ready)
    return ready

def produce_new_atoms(model,selection,digest,index,canvas,cohort,images):
    folder=F/'models'/model['model_id'];scratch=TMP/'final270/models'/model['model_id']
    if Path(model['atoms_ready_path']).exists():return validate_atoms(model,selection,index,cohort)
    if (folder/'MODEL_PREACCESS.json').exists():
        ready_from_journal(model,selection,digest,index,canvas,cohort,folder)
        return validate_atoms(model,selection,index,cohort)
    registration=read(model['registration_108_path']);handoff,network=load_handoff(registration)
    for key in ('checkpoint_path','checkpoint_sha256','architecture','construction'):assert handoff[key]==model[key],key
    folder.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True)
    assert not list(scratch.glob('atoms/*/*.npy')),'unregistered existing atom payload'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    from scripts.followup_test270_common import numeric_mode
    numeric_mode();torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.95)
    network=network.cuda();synthetic=torch.randn(8,1,*canvas,device='cuda')
    g=dict(canvas_offset=[0,0,0],extent=list(canvas),reference_enabled=True,reflection=[[-1,0,0,canvas[0]-1],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    with torch.inference_mode():
        full=forward_view(network,synthetic,[g]*8,7,canvas)
        tail=forward_view(network,synthetic[:6],[g]*6,0,canvas)
    peak=torch.cuda.max_memory_reserved()/2**30;assert peak<=38
    write(folder/'GPU_SYNTHETIC_GATE.json',dict(at=now(),status='passed',canvas=list(canvas),batch=8,tail=6,peak_reserved_gib=peak,CPU_strict_reload=True,patient_forwards=0,root_selection_sha256=digest))
    del synthetic,full,tail;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    with (folder/'MODEL_PREACCESS.json').open('x') as out:
        json.dump(dict(at=now(),checkpoint_sha256=model['checkpoint_sha256'],root_selection_sha256=digest,scans=270,subjects=119,views=list(range(8)),maximum_case_forwards=2160,automatic_neural_retry=False),out)
    atoms=[];started=time.monotonic()
    try:
        for first in range(0,270,8):
            guard();assert sha(SELECTION)==digest
            slots=list(range(first,min(first+8,270)));image=torch.from_numpy(np.array(images[slots],np.float32))[:,None].cuda()
            geometries=[{k:index['geometry'][cohort[slot]['case_id']][k] for k in ('canvas_offset','extent','reflection','reference_enabled')} for slot in slots]
            for view in range(8):
                outputs=forward_view(network,image,geometries,view,canvas)
                for slot,z in zip(slots,outputs):
                    path=scratch/'atoms'/f'{slot:03}'/f'flip{view}.npy';path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.tmp')
                    with temporary.open('xb') as out:np.save(out,z,allow_pickle=False)
                    os.replace(temporary,path);path.chmod(0o444)
                    atoms.append(dict(slot=slot,view=view,path=str(path),shape=list(z.shape),sha256=sha(path)))
                write(folder/'FORWARD_JOURNAL.json',dict(at=now(),root_selection_sha256=digest,selection_sha256=digest,checkpoint_sha256=model['checkpoint_sha256'],case_forwards=len(atoms),maximum=2160,atoms=atoms,seconds=time.monotonic()-started,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
                write(folder/'HEARTBEAT.json',dict(at=now(),stage='final270_neural_forward',case_forwards=len(atoms),maximum=2160))
            del image,outputs
        assert sha(SELECTION)==digest and sha(model['checkpoint_path'])==model['checkpoint_sha256']
        for p in (scratch/'atoms').iterdir():p.chmod(0o555)
        (scratch/'atoms').chmod(0o555)
        ready=ready_from_journal(model,selection,digest,index,canvas,cohort,folder)
        ready['raw_complete_statistics_only_recovery']=False;write(model['atoms_ready_path'],ready)
    except BaseException as e:
        import traceback
        write(folder/f'FORWARD_FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(e),traceback=traceback.format_exc(),completed_atom_count=len(atoms),automatic_neural_retry=False));raise
    finally:
        del network;torch.cuda.empty_cache()
    return validate_atoms(model,selection,index,cohort)

def score_single(candidate,model,ready,selection,digest,folder,index,canvas,cohort,images):
    # Ready is complete before the first label memmap is opened.
    assert ready['case_forwards']==2160;assert sha(SELECTION)==digest
    labels=np.memmap(folder/'labels.u8',np.uint8,'r',shape=(270,*canvas))
    cid=candidate['candidate_id'];outdir=F/'candidates'/cid;scratch=TMP/'final270/candidates'/cid/'predictions'
    outdir.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True)
    recipe=candidate['single_recipe'];lookup={(a['slot'],a['view']):a for a in ready['atoms']};records=[];started=time.monotonic()
    write(outdir/'DECODE_ACCESS.json',dict(at=now(),root_selection_sha256=digest,model_id=model['model_id'],atoms_ready_path=model['atoms_ready_path'],atoms_ready_sha256=sha(model['atoms_ready_path']),recipe=recipe,new_neural_forwards=0))
    for slot,row in enumerate(cohort):
        guard();assert sha(SELECTION)==digest;sl=slices(index['geometry'][row['case_id']]);logits=[]
        for view in range(8):
            atom=lookup[slot,view];assert sha(atom['path'])==atom['sha256'];logits.append(np.load(atom['path'],allow_pickle=False))
        # No GT or case identity is passed into decoding.
        output=decode_all(logits,np.array(images[slot][sl],np.float32),[recipe])[0]
        mask=output.pop('mask');point=output['automatic_point'];target=np.array(labels[slot][sl]>0)
        assert mask.shape==target.shape and all(0<=p<n for p,n in zip(point,mask.shape))
        path=scratch/(row['case_id']+'.npz')
        if path.exists():
            old=np.load(path,allow_pickle=False);previous=np.unpackbits(old['mask_bits'])[:mask.size].reshape(mask.shape).astype(bool)
            assert np.array_equal(previous,mask) and old['point'].tolist()==point
        else:
            temporary=path.with_name(path.name+'.tmp')
            with temporary.open('xb') as out:np.savez_compressed(out,mask_bits=np.packbits(mask.ravel()),shape=np.array(mask.shape),point=np.array(point))
            os.replace(temporary,path);path.chmod(0o444)
        saved=np.load(path,allow_pickle=False);saved_mask=np.unpackbits(saved['mask_bits'])[:mask.size].reshape(mask.shape).astype(bool)
        assert np.array_equal(saved_mask,mask)
        tp=int((saved_mask&target).sum());pred=int(saved_mask.sum());positive=int(target.sum());den=pred+positive
        records.append(dict(case_id=row['case_id'],subject_id=row['subject_id'],source_slot=slot,locator_dsc=float(np.float32(2*tp)/np.float32(den)) if den else 1.,tp=tp,fp=pred-tp,fn=positive-tp,automatic_point_hits_gt=bool(target[tuple(point)]),prediction_path=str(path),prediction_sha256=sha(path),prediction_shape=list(mask.shape),**output))
        write(outdir/'DECODE_JOURNAL.json',dict(at=now(),completed=slot+1,total=270,root_selection_sha256=digest,per_case=records))
    assert len(records)==270 and len({r['subject_id'] for r in records})==119
    references={name:read(OLD/f'RAW_TEST270_{name}.json')['per_case'] for name in ('raw','enhanced')}
    comparisons={};worse={}
    for name,reference in references.items():
        comparisons[name]=paired(records,reference);right={r['case_id']:r for r in reference}
        worse[name]=sorted([dict(case_id=r['case_id'],subject_id=r['subject_id'],delta=r['locator_dsc']-right[r['case_id']]['locator_dsc'],candidate=r['locator_dsc'],reference=right[r['case_id']]['locator_dsc']) for r in records if r['locator_dsc']<right[r['case_id']]['locator_dsc']],key=lambda r:r['delta'])
    raw=dict(at=now(),candidate_id=cid,kind='single',per_case=records,selection_sha256=digest,root_selection_sha256=digest,checkpoint_sha256=model['checkpoint_sha256'],recipe=recipe,network_case_forwards_added_by_decoding=0,test_tuned_history=True,current_test_parameter_search=False)
    write(outdir/'RAW_TEST270.json',raw);write(F/'RAW_SINGLE.json',raw)
    result=dict(at=now(),state='complete',candidate_id=cid,kind='single',metrics=summary_ci(records),comparisons_vs_historical_RACE164=comparisons,worse_cases=worse,root_selection_sha256=digest,checkpoint_sha256=model['checkpoint_sha256'],recipe=recipe,raw_sha256=sha(outdir/'RAW_TEST270.json'),atoms_ready_sha256=sha(model['atoms_ready_path']),forward_cost_per_case=recipe['neural_case_forwards'],retained_views_per_case=8,whole_original_crop_GT_including_outside_brain=True,seconds=time.monotonic()-started,test_tuned_history=True,independent_confirmation=False,current_test_parameter_search=False)
    write(outdir/'RESULT.json',result)
    audit_saved_masks(F/'RAW_SINGLE.json',selection,digest,result['metrics'])
    return result

def audit_saved_masks(raw_path,selection,digest,reported_metrics):
    """Second pass from saved packed masks, independent of prediction/decoder."""
    raw_digest=sha(raw_path);raw=read(raw_path);assert raw['selection_sha256']==digest==sha(SELECTION)
    cache=Path(selection['data']['native_cache']);assert sha(cache/'index.json')==selection['data']['index_sha256']
    assert sha(cache/'labels.u8')==selection['data']['payload_sha256']['native_label']
    index=read(cache/'index.json');canvas=tuple(index['canvas']);labels=np.memmap(cache/'labels.u8',np.uint8,'r',shape=(270,*canvas))
    assert len(raw['per_case'])==270;audits=[];rescored=[]
    for record in raw['per_case']:
        slot=record['source_slot'];assert index['case_ids'][slot]==record['case_id']
        geometry=index['geometry'][record['case_id']];path=Path(record['prediction_path'])
        assert sha(path)==record['prediction_sha256'];saved=np.load(path,allow_pickle=False)
        shape=tuple(map(int,saved['shape']));assert list(shape)==geometry['extent']==record['prediction_shape']
        mask=np.unpackbits(saved['mask_bits'])[:int(np.prod(shape))].reshape(shape).astype(bool)
        point=list(map(int,saved['point']));assert point==record['automatic_point']
        # Full original crop including positives outside any brain support.
        target=np.array(labels[slot][slices(geometry)]>0);assert target.shape==mask.shape
        tp=int(np.count_nonzero(mask&target));fp=int(np.count_nonzero(mask&~target));fn=int(np.count_nonzero(~mask&target))
        denominator=2*tp+fp+fn;dice=float(np.float32(2*tp)/np.float32(denominator)) if denominator else 1.
        hit=bool(target[tuple(point)])
        assert (tp,fp,fn,dice,hit)==(record['tp'],record['fp'],record['fn'],record['locator_dsc'],record['automatic_point_hits_gt'])
        audits.append(dict(case_id=record['case_id'],source_slot=slot,tp=tp,fp=fp,fn=fn,locator_dsc=dice,automatic_point=point,point_hit=hit,prediction_sha256=record['prediction_sha256']))
        rescored.append(dict(case_id=record['case_id'],subject_id=record['subject_id'],locator_dsc=dice,automatic_point_hits_gt=hit))
    metrics=summary_ci(rescored);assert metrics==reported_metrics
    write(F/'MASK_AUDIT_SINGLE.json',dict(at=now(),state='passed',status='passed',raw_path=str(raw_path),raw_sha256=raw_digest,selection_sha256=digest,index_sha256=selection['data']['index_sha256'],label_sha256=selection['data']['payload_sha256']['native_label'],audit_source_file=str(Path(__file__).relative_to(ROOT)),audit_source_sha256=sha(Path(__file__)),scans=270,metrics=metrics,per_case=audits,all_saved_masks_counts_dice_point_exact=True,whole_original_crop_including_outside_support=True,model_forwards_added=0,new_parameter_search=False))

def execute(statistics_only=False):
    selection,digest,cohort,models=commit_selection();folder,index,canvas,images=data(selection,cohort)
    lock_path=Path('/tmp/aflow_race_v2_20261029/locks/worker1154121_gpu0.lock');lock_path.parent.mkdir(parents=True,exist_ok=True);lock=lock_path.open('a')
    own_ready={}
    with lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for mid,model in models.items():
            if model['owner']!='inference_a':continue
            if model['atom_source']=='reuse_race164':own_ready[mid]=validate_atoms(model,selection,index,cohort)
            elif statistics_only:
                assert Path(model['atoms_ready_path']).exists(),'statistics-only cannot launch network'
                own_ready[mid]=validate_atoms(model,selection,index,cohort)
            else:own_ready[mid]=produce_new_atoms(model,selection,digest,index,canvas,cohort,images)
    results=[]
    for candidate in selection['candidates']:
        if candidate['kind']!='single':continue
        model=candidate['models'][0];results.append(score_single(candidate,model,own_ready[model['model_id']],selection,digest,folder,index,canvas,cohort,images))
    write(F/'A_EXECUTION_COMPLETE.json',dict(at=now(),state='complete',root_selection_sha256=digest,owned_model_manifests={m:dict(path=models[m]['atoms_ready_path'],sha256=sha(models[m]['atoms_ready_path'])) for m in own_ready},single_results=results,fusion_delegated_to='inference_b',statistics_only=statistics_only,automatic_parameter_search=False))

def cpu_check():
    schema=read(STAGE/'FINAL_TEST_SELECTION_SCHEMA.json')
    try:validate_selection(schema)
    except AssertionError as error:assert 'schema is not authorization' in str(error)
    else:raise AssertionError('schema accidentally authorized execution')
    rng=np.random.default_rng(19);zs=[rng.normal(size=(7,9,11)).astype(np.float32) for _ in range(8)]
    recipe=dict(id='equal8_logit_t05',mode='equal',config=None,neural_case_forwards=8)
    output=decode_all(zs,np.ones((7,9,11),np.float32),[recipe])[0]
    expected=torch.zeros_like(torch.from_numpy(zs[0]))
    for z in zs:expected.add_(torch.from_numpy(z),alpha=.125)
    assert np.array_equal(output['mask'],expected.numpy()>0)
    assert output['automatic_point']==list(map(int,np.unravel_index(int(expected.argmax()),expected.shape)))
    return dict(at=now(),status='passed',schema_cannot_trigger=True,equal_mode_with_null_config=True,synthetic_only=True,clinical_arrays_loaded=False,new270_scoring=False,network_forwards=0,entry_source_sha256=sha(Path(__file__)))

def main(watch=False,statistics_only=False):
    if watch:
        while not SELECTION.exists():guard();time.sleep(30)
    execute(statistics_only)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--watch',action='store_true');p.add_argument('--statistics-only',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:
        try:main(a.watch,a.statistics_only)
        except BaseException as e:
            import traceback
            write(F/f'EXECUTION_FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(e),traceback=traceback.format_exc(),automatic_neural_retry=False));raise
