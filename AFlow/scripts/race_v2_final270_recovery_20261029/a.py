"""Reconstruct the exact historical RACE atoms, then evaluate one frozen single."""
import argparse,time,os,traceback
import numpy as np
import torch
from scripts.race_v2_final270_recovery_20261029.a_common import *
from scripts.race_v2_final270_recovery_20261029 import common as control
from scripts.race_inference_test270_20261029.predictor import forward_view,clean_geometry
from scripts.race_v2_inference_soft_context_a_20261029.deployment import deploy
from scripts.race_v2_inference_innovation_a_20261029.io_helpers import record_mask,audit_masks
from scripts.race_v2_inference_a_20261029.statistics import summary_ci,paired
from scripts.followup_test270_common import numeric_mode

FREEZE=R/'FORWARD_FREEZE.json'

def freeze():
    assert not FREEZE.exists();selection=read(SELECTION);assert sha(SELECTION)==SELECTION_SHA
    candidate=next(c for c in selection['candidates'] if c['kind']=='single')
    assert candidate['models'][0]['model_id']=='race164' and candidate['single_recipe']['mode']=='race8_local_dependency'
    source=dict(read(R/'EXECUTION_FREEZE.json')['source_sha256']);source.update(read(ROOT/'archive/research_records/2026-10-29_race_v2_inference_a/FINAL270_SOFT_CONTEXT_ENTRY_FREEZE.json')['source_sha256'])
    for name in ('a.py','common.py'):
        p=ROOT/'scripts/race_v2_final270_recovery_20261029'/name;source[str(p.relative_to(ROOT))]=sha(p)
    p=ROOT/'scripts/race_v2_stage_20261029/governance.py';source[str(p.relative_to(ROOT))]=sha(p)
    write(FREEZE,dict(at=now(),source_sha256=source,selection_sha256=SELECTION_SHA,authorization_sha256=sha(BASE/'AUTHORIZATION.json'),
        resource_allocation_sha256=sha(BASE/'RESOURCE_ALLOCATION.json'),historical_forward_reports={str(OLD/f'FORWARD_COMPLETE_{rank}.json'):sha(OLD/f'FORWARD_COMPLETE_{rank}.json') for rank in (0,1)},
        checkpoint_sha256=CHECKPOINT_SHA,batch=8,tail=7,lanes='even135 then odd135, sequential on assigned GPU0',
        reconstruction_forwards_max=2160,selected_single=candidate,scope='reconstruct old atoms and evaluate unchanged final single only',
        no_new_parameter_search=True,no_automatic_neural_retry=True))

def verify_forward():
    cache_freeze=verify();control.guard();f=read(FREEZE)
    assert sha(CANON_INDEX)==cache_freeze['canonical_geometry_index_sha256']
    assert sha(MANIFEST)==cache_freeze['manifest_sha256']
    for p,d in f['source_sha256'].items():assert sha(ROOT/p)==d,p
    assert sha(BASE/'AUTHORIZATION.json')==f['authorization_sha256'] and sha(BASE/'RESOURCE_ALLOCATION.json')==f['resource_allocation_sha256']
    for p,d in f['historical_forward_reports'].items():assert sha(p)==d,p
    assert sha(SELECTION)==f['selection_sha256']==SELECTION_SHA
    return f

def strict_model():
    from sammed3d.race_seg import RaceSeg
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False);assert saved['epoch']==164 and saved['arm']=='race'
    network=RaceSeg(width=32,dropout=.1,enabled=True);network.load_state_dict(saved['model_state_dict'],strict=True)
    assert all(torch.equal(v,saved['model_state_dict'][k]) and bool(v.isfinite().all()) for k,v in network.state_dict().items())
    write(R/'CPU_STRICT_RELOAD.json',dict(at=now(),status='passed',checkpoint_sha256=CHECKPOINT_SHA,architecture='sammed3d.race_seg.RaceSeg',
        construction=dict(width=32,dropout=.1,dim=16,enabled=True),strict_before_cuda=True,optimizer_loaded=False))
    return network.eval().requires_grad_(False).cuda()

def synthetic_gate(network,canvas):
    torch.manual_seed(193);torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    image=torch.randn(8,1,*canvas,device='cuda');g=dict(canvas_offset=[0,0,0],extent=list(canvas),reference_enabled=True,
        reflection=[[-1,0,0,canvas[0]-1],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    full=forward_view(network,image,[g]*8,7);tail=forward_view(network,image[:7],[g]*7,0)
    assert len(full)==8 and len(tail)==7;torch.cuda.synchronize();peak=torch.cuda.max_memory_reserved()/2**30;assert peak<=38
    return dict(at=now(),status='passed',canvas=list(canvas),batch=8,tail=7,finite=True,peak_reserved_gib=peak,
        seconds=time.monotonic()-started,patient_forwards=0,exact_historical_predictor_source=True)

def reconstruct(index,images,cases,cache_ready):
    expected={}
    for rank in (0,1):
        report=read(OLD/f'FORWARD_COMPLETE_{rank}.json');assert report['case_forwards']==1080 and report['neural_batch_calls']==136
        expected.update({(a['slot'],a['view']):a for a in report['atoms']})
    assert len(expected)==2160 and read(OLD/'BATCH_DECISION.json')['batch']==8 and read(OLD/'BATCH_DECISION.json')['tail']==7
    if (R/'RACE_ATOMS_READY.json').exists():
        ready=read(R/'RACE_ATOMS_READY.json');assert ready['all2160_historical_atom_SHA_exact']
        for atom in ready['atoms']:assert sha(atom['path'])==atom['sha256']==expected[atom['slot'],atom['view']]['sha256']
        return ready
    if (R/'MODEL_PREACCESS.json').exists():
        journal=read(R/'FORWARD_JOURNAL.json');atoms=journal['atoms']
        assert journal['actual_new_model_forwards']==2160 and journal['neural_batch_calls']==272 and len(atoms)==2160,'incomplete raw: no automatic neural retry'
        assert {(a['slot'],a['view']) for a in atoms}==set(expected)
        for atom in atoms:
            assert sha(atom['path'])==atom['sha256']==expected[atom['slot'],atom['view']]['sha256']
            value=np.load(atom['path'],mmap_mode='r',allow_pickle=False);assert list(value.shape)==atom['shape'] and np.isfinite(value).all()
        return commit_ready(index,cache_ready,atoms,journal['seconds'],journal.get('peak_reserved_gib'),True)
    network=strict_model();write(R/'GPU_SYNTHETIC_GATE.json',synthetic_gate(network,tuple(index['canvas'])))
    scratch=TMP/'a/atoms';scratch.mkdir(parents=True,exist_ok=True);assert not list(scratch.glob('*/*.npy'))
    write(R/'MODEL_PREACCESS.json',dict(at=now(),worker=1154259,physical_gpu=0,checkpoint_sha256=CHECKPOINT_SHA,
        selection_sha256=SELECTION_SHA,authorization_sha256=sha(BASE/'AUTHORIZATION.json'),forward_freeze_sha256=sha(FREEZE),
        cache_ready_sha256=sha(R/'CACHE_READY.json'),original_worker_payloads_lost=True,exact_reconstruction=True,
        maximum_case_forwards=2160,lanes=[0,1],batch=8,tail=7,no_automatic_neural_retry=True))
    atoms=[];attempted=0;calls=0;started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for lane in (0,1):
        slots=list(range(lane,270,2))
        for first in range(0,135,8):
            verify_forward();current=slots[first:first+8];image=torch.from_numpy(np.array(images[current],np.float32))[:,None].cuda()
            geometry=[clean_geometry(index['geometry'][cases[s]['case_id']]) for s in current]
            for view in range(8):
                values=forward_view(network,image,geometry,view);attempted+=len(current);calls+=1;mismatches=[]
                for slot,value in zip(current,values):
                    original=expected[slot,view];assert list(value.shape)==original['shape']
                    path=scratch/f'{slot:03}'/f'flip{view}.npy';path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.tmp')
                    with temporary.open('xb') as out:np.save(out,value,allow_pickle=False)
                    os.replace(temporary,path);path.chmod(0o444);digest=sha(path)
                    atom=dict(slot=slot,view=view,path=str(path),shape=list(value.shape),sha256=digest,input_kind='historical_input_reconstructed',historical_sha256=original['sha256'])
                    atoms.append(atom)
                    if digest!=original['sha256']:mismatches.append(atom)
                journal=dict(at=now(),lane=lane,actual_new_model_forwards=attempted,maximum=2160,neural_batch_calls=calls,
                    atoms=atoms,seconds=time.monotonic()-started,selection_sha256=SELECTION_SHA,checkpoint_sha256=CHECKPOINT_SHA,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                write(R/'FORWARD_JOURNAL.json',journal);write(R/'HEARTBEAT.json',dict(at=now(),stage='exact_RACE_reconstruction',completed=attempted,total=2160,lane=lane))
                if mismatches:
                    write(R/'ATOM_SHA_MISMATCH.json',dict(journal,mismatches=mismatches,stop_without_retry=True))
                    raise RuntimeError('reconstructed historical atom SHA mismatch; stopped immediately without repeated inference')
            del image,values
    assert attempted==2160 and calls==272 and len(atoms)==2160
    ready=commit_ready(index,cache_ready,atoms,time.monotonic()-started,torch.cuda.max_memory_reserved()/2**30,False)
    del network;torch.cuda.empty_cache();return ready

def commit_ready(index,cache_ready,atoms,seconds,peak,statistics_only_reseal):
    ready=dict(at=now(),status='passed',worker=1154259,physical_gpu=0,model_id='race164',checkpoint_path=str(CHECKPOINT),checkpoint_sha256=CHECKPOINT_SHA,
        architecture='sammed3d.race_seg.RaceSeg',construction=dict(width=32,dropout=.1,dim=16,enabled=True),
        manifest_sha256=sha(MANIFEST),index_sha256=cache_ready['index_sha256'],cache_path=str(CACHE),canvas=index['canvas'],
        atoms=atoms,case_forwards=2160,new_model_forwards=2160,actual_new_forwards=2160,
        bounded_reconstruction_after_worker_loss=True,original_payload_reuse=False,all2160_historical_atom_SHA_exact=True,
        selection_sha256=SELECTION_SHA,recovery_authorization_sha256=sha(BASE/'AUTHORIZATION.json'),
        source_freeze_sha256=sha(FREEZE),journal_sha256=sha(R/'FORWARD_JOURNAL.json'),
        seconds=seconds,peak_reserved_gib=peak,raw_complete_statistics_only_reseal=statistics_only_reseal)
    write(R/'RACE_ATOMS_READY.json',ready);return ready

def score(index,images,cases,ready):
    selection=read(SELECTION);candidate=next(c for c in selection['candidates'] if c['kind']=='single');entry=candidate['single_recipe']
    assert candidate['models'][0]['checkpoint_sha256']==CHECKPOINT_SHA
    labels=np.memmap(CACHE/'labels.u8',np.uint8,'r',shape=(270,*index['canvas']));lookup={(a['slot'],a['view']):a for a in ready['atoms']};records=[];started=time.monotonic()
    for slot,row in enumerate(cases):
        verify_forward();g=index['geometry'][row['case_id']];zs=[]
        for view in range(8):
            atom=lookup[slot,view];assert sha(atom['path'])==atom['sha256'];zs.append(torch.from_numpy(np.load(atom['path'],allow_pickle=False)).cuda())
        output=deploy(zs,np.array(images[slot][slices(g)],np.float32),entry);target=np.array(labels[slot][slices(g)]>0)
        records.append(record_mask(row,slot,output['mask'].cpu().numpy(),output['automatic_point'],output['base_automatic_point'],target,
            TMP/'a/predictions'/f'{slot:03}.npz',output['diagnostic']))
        write(R/'DECODE_JOURNAL.json',dict(at=now(),completed=slot+1,total=270,selection_sha256=SELECTION_SHA))
        del zs,output
    raw=R/'candidates/single/RAW_TEST270.json';write(raw,dict(at=now(),candidate_id='single',selection_sha256=SELECTION_SHA,
        per_case=records,recipe=entry,checkpoint_sha256=CHECKPOINT_SHA,recovery_authorization_sha256=sha(BASE/'AUTHORIZATION.json'),
        actual_new_model_forwards=2160,historical_prediction_SHA_reconstruction=True,new_test_parameter_search=False))
    audit=R/'MASKS_AUDIT_RECEIPT_SINGLE.json';audit_masks(raw,audit,cases,index,labels,SELECTION_SHA)
    public=R/'RAW_SINGLE.json';write(public,dict(read(raw),canonical_raw_path=str(raw),canonical_raw_sha256=sha(raw),
        masks_audit_receipt_path=str(audit),masks_audit_receipt_sha256=sha(audit)))
    metric=summary_ci(records);comparisons={};worse={}
    for name in ('raw','enhanced'):
        reference=read(OLD/f'RAW_TEST270_{name}.json')['per_case'];comparisons[name]=paired(records,reference);right={r['case_id']:r for r in reference}
        worse[name]=sorted([dict(case_id=r['case_id'],subject_id=r['subject_id'],delta=r['locator_dsc']-right[r['case_id']]['locator_dsc']) for r in records if r['locator_dsc']<right[r['case_id']]['locator_dsc']],key=lambda r:r['delta'])
    write(R/'RESULT_SINGLE.json',dict(at=now(),state='complete',worker=1154259,physical_gpu=0,candidate_id='single',kind='single',model_count=1,metrics=metric,Point_refined=metric['point_hits'],
        Point_base=sum(r['base_automatic_point_hits_gt'] for r in records),Point_auxiliary_rule='same-model ordinary equal8 argmax',
        recipe=entry,checkpoint_sha256=CHECKPOINT_SHA,selection_sha256=SELECTION_SHA,recovery_authorization_sha256=sha(BASE/'AUTHORIZATION.json'),
        aggregate_raw_path=str(public),aggregate_raw_sha256=sha(public),raw_sha256=sha(raw),
        masks_audit_receipt_path=str(audit),masks_audit_receipt_sha256=sha(audit),comparisons_vs_historical_RACE164=comparisons,worse_cases=worse,
        forward_cost_per_case=8,actual_new_model_forwards=2160,reconstruction_cost_case_forwards=2160,
        source_freeze_sha256=sha(FREEZE),all2160_historical_atom_SHA_exact=True,decode_seconds=time.monotonic()-started,
        test_tuned_history=True,independent_confirmation=False,current_test_parameter_search=False))

def main(watch):
    verify_forward();numeric_mode();torch.set_num_threads(2)
    write(R/'NUMERIC_MODE_RECEIPT.json',dict(at=now(),pid=os.getpid(),same_process=True,
        backend_mode=dict(cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)))
    if watch:
        while not (R/'CACHE_READY.json').exists():
            control.guard();assert not list(R.glob('CACHE_FAILURE*')),'CPU cache builder failed';time.sleep(10)
    cache_ready=read(R/'CACHE_READY.json');assert cache_ready['all_four_payloads_exact']
    assert sha(CACHE/'index.json')==cache_ready['index_sha256'];index=read(CACHE/'index.json');cases=cohort()
    assert index['case_ids']==[r['case_id'] for r in cases] and index['canvas']==[160,208,168]
    assert sha(CACHE/'images.f32')==cache_ready['payload_sha256']['native_image']
    assert sha(CACHE/'labels.u8')==cache_ready['payload_sha256']['native_label']
    images=np.memmap(CACHE/'images.f32',np.float32,'r',shape=(270,*index['canvas']))
    with control.gpu_queue_lock('a'):
        torch.cuda.set_per_process_memory_fraction(.95)
        ready=reconstruct(index,images,cases,cache_ready);score(index,images,cases,ready)
    write(R/'GPU_RELEASE.json',dict(at=now(),pid=os.getpid(),released=True,new_model_forwards=2160,worker=1154259,gpu=0))
    write(R/'DONE.json',dict(at=now(),state='complete',selection_sha256=SELECTION_SHA,scans=270,new_model_forwards=2160,result_sha256=sha(R/'RESULT_SINGLE.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--watch',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:
        try:main(a.watch)
        except BaseException as error:
            write(R/f'INFERENCE_FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc(),automatic_neural_retry=False));raise
