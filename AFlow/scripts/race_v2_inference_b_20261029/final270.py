"""Root-triggered fixed fusion only. A owns RACE atoms; B owns new w32 atoms."""
import argparse
import re
import numpy as np
import torch
from scripts.race_v2_inference_b_20261029.common import *
from scripts.race_v2_inference_b_20261029.forward import load_model
from scripts.race_v2_inference_b_20261029.analyze import metrics,mix,decode,record

F=R/'final270';FTMP=TMP/'test270_b'
TEST_MANIFEST=ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/sealed_test/test_cases_t1ce.jsonl'
TEST_SHA='a5ddc0ed2c0b269431758b14061da3c6163dd0c48133b60a2ef8efd7f23aa8f9'
NATIVE=Path('/tmp/aflow_race_e164_test270_20261029/native')
WREADY=F/'models/w32e171/ATOMS_READY.json'

def verify_entry():
    verify()
    for path,digest in read(R/'FINAL_ENTRY_FREEZE.json')['source_sha256'].items():assert sha(ROOT/path)==digest,path

def final_selection(candidate_id=None):
    verify_entry();path=STAGE/'FINAL_TEST_SELECTION.json';selection=read(path)
    assert selection['schema']=='aflow.race_v2.final_test270.v1'
    assert selection['registered_by']=='root' and selection['approval_scope']=='final_frozen_test270_only'
    assert not selection.get('schema_only_not_authorization',False)
    assert 1<=len(selection['candidates'])<=2
    for source,digest in selection['entry_source_sha256'].items():assert sha(ROOT/source)==digest,source
    terminal=selection['training_terminal']
    for prefix in ('done','budget','result'):assert sha(terminal[prefix+'_path'])==terminal[prefix+'_sha256']
    done=read(terminal['done_path']);budget=read(terminal['budget_path'])
    assert done['state']=='complete' and done['new_epoch']==done['target_new_epochs']==budget['new_epochs']
    for item in selection['validation_audit']:assert sha(item['path'])==item['sha256']
    assert selection['cohort']['scans']==270 and selection['cohort']['subjects']==119
    assert selection['cohort']['manifest_sha256']==sha(TEST_MANIFEST)==TEST_SHA
    assert Path(selection['cohort']['manifest_path'])==TEST_MANIFEST
    if candidate_id is None:return selection,None
    matches=[c for c in selection['candidates'] if c['candidate_id']==candidate_id];assert len(matches)==1
    candidate=matches[0];assert candidate['kind']=='fusion' and len(candidate['models'])==2
    assert len({m['model_id'] for m in candidate['models']})==2
    assert re.fullmatch(r'[A-Za-z0-9_-]+',candidate_id)
    assert candidate['fusion']['primary_weight'] in (.5,.75,.9)
    assert candidate['fusion']['space'] in ('logit','probability') and candidate['fusion']['decoder'] in ('raw','full')
    assert sha(candidate['full_recipe_path'])==candidate['full_recipe_sha256']
    for source in candidate['validation_sources']:assert sha(source['path'])==source['sha256']
    for model in candidate['models']:
        assert sha(model['checkpoint_path'])==model['checkpoint_sha256']
        assert 'latest' not in Path(model['checkpoint_path']).name
        if model['atom_source']=='new_snapshot':
            assert sha(model['handoff_path'])==model['handoff_sha256']
            handoff=read(model['handoff_path']);assert handoff['strict_cpu_reload'] and handoff['actual_new_epochs']>0
            assert handoff['checkpoint_sha256']==model['checkpoint_sha256']
            assert handoff['total_stage_new_epochs']==done['new_epoch']
    return selection,candidate

def test_data(selection,scoring=False):
    info=selection['data'];cache=Path(info['native_cache']);assert cache==NATIVE
    index=read(cache/'index.json');assert sha(cache/'index.json')==info['index_sha256']
    assert sha(info['existing_cache_audit_path'])==info['existing_cache_audit_sha256']
    cohort=[json.loads(line) for line in TEST_MANIFEST.read_text().splitlines() if line.strip()]
    assert len(cohort)==270 and len({r['subject_id'] for r in cohort})==119
    assert index['case_ids']==[r['case_id'] for r in cohort] and index['split']=='test'
    canvas=tuple(index['canvas']);assert list(canvas)==selection['cohort']['canvas']
    assert all(s%8==0 for s in canvas);count=270*int(np.prod(canvas))
    assert (cache/'images.f32').stat().st_size==count*4
    assert sha(cache/'images.f32')==info['payload_sha256']['native_image']
    images=np.memmap(cache/'images.f32',np.float32,'r',shape=(270,*canvas));labels=None
    if scoring:
        assert (cache/'labels.u8').stat().st_size==count
        assert sha(cache/'labels.u8')==info['payload_sha256']['native_label']
        labels=np.memmap(cache/'labels.u8',np.uint8,'r',shape=(270,*canvas))
    return index,cohort,images,labels

@torch.inference_mode()
def synthetic(model,canvas):
    probes=[]
    for batch in (4,8):
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        try:
            image=torch.zeros((batch,1,*canvas),device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):value=model(image).segmentation_logits
            assert value.shape==image.shape and bool(value.isfinite().all())
            torch.cuda.synchronize();begun=time.monotonic()
            for _ in range(3):
                with torch.autocast('cuda',dtype=torch.bfloat16):value=model(image).segmentation_logits
            torch.cuda.synchronize();seconds=time.monotonic()-begun
            reserved=torch.cuda.max_memory_reserved()/2**30
            probes.append(dict(batch=batch,passed=reserved<=38,cases_per_second=3*batch/seconds,peak_reserved_gib=reserved))
            del image,value
        except torch.cuda.OutOfMemoryError as error:
            probes.append(dict(batch=batch,passed=False,error=repr(error),patient_forwards=0))
            if 'image' in locals():del image
            if 'value' in locals():del value
            torch.cuda.empty_cache()
    viable=[p for p in probes if p['passed']];assert viable
    fastest=max(p['cases_per_second'] for p in viable);batch=min(p['batch'] for p in viable if p['cases_per_second']>=fastest*.97)
    tail=270%batch
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    if tail:
        image=torch.zeros((tail,1,*canvas),device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):value=model(image).segmentation_logits
        assert value.shape==image.shape and bool(value.isfinite().all());del image,value
    assert torch.cuda.max_memory_reserved()/2**30<=38
    result=dict(at=now(),status='passed',canvas=list(canvas),batch=batch,tail=tail,probes=probes,
        checkpoint_sha256=CHECKPOINT_SHA,strict_cpu_reload=True,patient_forwards=0,no_test_pixels=True)
    write(F/'models/w32e171/NUMERIC_GATE.json',result);return batch

@torch.inference_mode()
def forward_w32():
    selection,_=final_selection();selected=[m for c in selection['candidates'] for m in c['models'] if m['model_id']=='w32e171']
    assert selected,'root did not select w32'
    assert all(m['checkpoint_sha256']==CHECKPOINT_SHA and m['atom_source']=='new_w32' and m['owner']=='inference_b' for m in selected)
    assert all(Path(m['atoms_ready_path'])==WREADY for m in selected)
    assert not WREADY.exists(),'complete model cache: decode/statistics only'
    model_record=F/'models/w32e171';assert not (model_record/'MODEL_PREACCESS.json').exists(),'no automatic neural retry'
    lock=gpu_lock();numeric_mode();index,cohort,images,_=test_data(selection,False);canvas=tuple(index['canvas'])
    model=load_model();torch.cuda.set_per_process_memory_fraction(.94);batch=synthetic(model,canvas)
    exclusive(model_record/'MODEL_PREACCESS.json',dict(at=now(),selection_sha256=sha(STAGE/'FINAL_TEST_SELECTION.json'),
        checkpoint_sha256=CHECKPOINT_SHA,maximum_case_forwards=2160,views=list(range(8)),split='test',automatic_retry=False))
    entries=[];calls=0;started=time.monotonic()
    for first in range(0,270,batch):
        guard();slots=list(range(first,min(first+batch,270)));image=torch.from_numpy(np.array(images[slots],np.float32))[:,None].cuda()
        for view in range(8):
            dims=tuple(a+2 for a in range(3) if view&(1<<a));transformed=torch.flip(image,dims) if dims else image
            with torch.autocast('cuda',dtype=torch.bfloat16):value=model(transformed).segmentation_logits
            value=value.float();value=torch.flip(value,dims) if dims else value;assert bool(value.isfinite().all());calls+=1
            for k,slot in enumerate(slots):
                path=FTMP/'models/w32e171/atoms'/f'{slot:03}'/f'flip{view}.npy';path.parent.mkdir(parents=True,exist_ok=True)
                array=value[(k,0)+slices(index['geometry'][cohort[slot]['case_id']])].contiguous().cpu().numpy()
                temp=path.with_name(path.name+'.tmp')
                with temp.open('xb') as handle:np.save(handle,array,allow_pickle=False)
                assert not path.exists();os.replace(temp,path)
                entries.append(dict(slot=slot,view=view,path=str(path),shape=list(array.shape),sha256=sha(path)))
            write(model_record/'FORWARD_JOURNAL.json',dict(at=now(),case_forwards=len(entries),maximum=2160,atoms=entries))
        del image,value,transformed
    assert len(entries)==2160 and len({(a['slot'],a['view']) for a in entries})==2160
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    ready=dict(at=now(),status='complete',model_id='w32e171',checkpoint_sha256=CHECKPOINT_SHA,
        manifest_sha256=TEST_SHA,index_sha256=selection['data']['index_sha256'],cache_path=str(NATIVE),canvas=list(canvas),
        architecture='sammed3d.t1ce_wt_capg3d.AutomaticPromptLocator',construction=dict(center_aware=False,init_filters=32,dropout=.1),
        case_forwards=2160,new_model_forwards=2160,batch_calls=calls,seconds=time.monotonic()-started,atoms=entries)
    write(WREADY,ready)
    for entry in entries:Path(entry['path']).chmod(0o444)
    for folder in (FTMP/'models/w32e171/atoms').iterdir():folder.chmod(0o555)
    (FTMP/'models/w32e171/atoms').chmod(0o555)

def atoms(model,selection):
    ready=read(model['atoms_ready_path']);assert ready['model_id']==model['model_id']
    assert ready['checkpoint_sha256']==model['checkpoint_sha256'] and ready['manifest_sha256']==TEST_SHA
    assert ready['index_sha256']==selection['data']['index_sha256'] and ready['case_forwards']==2160
    entries={(a['slot'],a['view']):a for a in ready['atoms']}
    assert len(entries)==2160 and set(entries)=={(s,v) for s in range(270) for v in range(8)}
    return entries

@torch.inference_mode()
def predict(left,right,image,fusion,config):
    """Deployment receives aligned logits/image/fixed global recipe only."""
    from scripts.segresnet_auto_inference12h_20261028.deploy_core_v3 import aggregate
    values=[];full=fusion['decoder']=='full';weight=fusion['primary_weight'];space=fusion['space']
    for logits in (left,right):
        if full:p,_=aggregate(logits,config['aggregation'],torch.from_numpy(image).cuda())
        elif space=='probability':p=sum(z.sigmoid() for z in logits.values()).div(8)
        else:p=sum(logits.values()).div(8)
        values.append(p)
    if not full and space=='logit':point_score=values[0]*weight+values[1]*(1-weight);probability=point_score.sigmoid()
    else:probability=mix(values[0],values[1],weight,space);point_score=probability
    original=mix(left[0].sigmoid(),right[0].sigmoid(),weight,space)>.5
    mask,cut=decode(probability,image,config,full,original.cpu().numpy())
    if not full and space=='logit':mask=(point_score>0).cpu().numpy()
    point=list(map(int,np.unravel_index(int(point_score.argmax()),mask.shape)))
    return mask,point,cut

@torch.inference_mode()
def score(candidate_id):
    selection,candidate=final_selection(candidate_id);lock=gpu_lock();numeric_mode()
    index,cohort,images,labels=test_data(selection,True);config=read(candidate['full_recipe_path'])
    sources=[atoms(model,selection) for model in candidate['models']];records=[];folder=F/'candidates'/candidate_id
    write(folder/'SCORING_PREACCESS.json',dict(at=now(),selection_sha256=sha(STAGE/'FINAL_TEST_SELECTION.json'),candidate=candidate,
         GT_used_for_scoring_only=True,current_test_search=False,neural_forwards_added_by_decode=0))
    for slot,row in enumerate(cohort):
        guard();sl=slices(index['geometry'][row['case_id']]);image=np.array(images[slot][sl],np.float32);logits=[]
        for source in sources:
            tensors={}
            for view in range(8):
                entry=source[slot,view];path=Path(entry['path']);assert sha(path)==entry['sha256']
                value=np.load(path,allow_pickle=False);assert value.dtype==np.float32 and value.shape==image.shape and list(value.shape)==entry['shape']
                assert np.isfinite(value).all();tensors[view]=torch.from_numpy(value).cuda()
            logits.append(tensors)
        mask,point,cut=predict(logits[0],logits[1],image,candidate['fusion'],config)
        # Target read only after the image-only predictor has finalized mask/point.
        target=np.array(labels[slot][sl]>0);tp=int((mask&target).sum());pred=int(mask.sum());positive=int(target.sum());den=pred+positive
        path=FTMP/'candidates'/candidate_id/'predictions'/(row['case_id']+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            old=np.load(path,allow_pickle=False);restored=np.unpackbits(old['mask_bits'])[:mask.size].reshape(mask.shape)
            assert np.array_equal(restored,mask) and old['point'].tolist()==point
        else:np.savez_compressed(path,mask_bits=np.packbits(mask.ravel()),shape=np.array(mask.shape),point=np.array(point))
        records.append(dict(case_id=row['case_id'],subject_id=row['subject_id'],source_slot=slot,
            locator_dsc=float(np.float32(2*tp)/np.float32(den)) if den else 1.,tp=tp,fp=pred-tp,fn=positive-tp,
            automatic_point=point,automatic_point_hits_gt=bool(target[tuple(point)]),quantized_cut=cut,
            prediction_path=str(path),prediction_sha256=sha(path),prediction_shape=list(mask.shape)))
        write(folder/'JOURNAL.json',dict(at=now(),completed=slot+1,total=270,per_case=records))
    write(folder/'RAW_TEST270.json',dict(at=now(),per_case=records,selection_sha256=sha(STAGE/'FINAL_TEST_SELECTION.json'),
        models=candidate['models'],fusion=candidate['fusion'],test_tuned_history=True,current_test_search=False,model_forwards_added=0))
    result=dict(at=now(),state='complete',candidate_id=candidate_id,kind='fusion',metrics=metrics(records),
        models=2,forwards_per_scan=16,model_forwards_added_by_decode=0,raw_sha256=sha(folder/'RAW_TEST270.json'),
        interpretation='Root-fixed recipe after108 selection; historically test-tuned reused270, not independent confirmation; root computes paired CIs to selected single/reference')
    write(folder/'RESULT.json',result)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('synthetic','forward-w32','score'));p.add_argument('--candidate-id');a=p.parse_args()
    if a.stage=='synthetic':
        verify_entry();lock=gpu_lock();numeric_mode();model=load_model();synthetic(model,(160,208,168))
    elif a.stage=='forward-w32':forward_w32()
    else:assert a.candidate_id;score(a.candidate_id)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        import traceback
        write(F/f'FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc(),automatic_neural_retry=False));raise
