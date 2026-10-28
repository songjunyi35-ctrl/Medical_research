"""Complete8^3 integer stride-phase ensemble, fixed flip0 and frozen checkpoint."""
import os
import numpy as np
import torch
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_v2 import stable_size
from scripts.segresnet_auto_inference12h_20261028.render_v1 import forward_view

SNAPSHOTS=[1,64,128,256,512]


def phase_order():
    # Each64-view block covers a complete even-phase grid. Low-bit cosets are
    # ordered in complementary pairs so128/256 prefixes have balanced marginals.
    cosets=[(0,0,0),(1,1,1),(1,0,0),(0,1,1),(0,1,0),(1,0,1),(0,0,1),(1,1,0)]
    base=[[4*((i>>d)&1)+2*((i>>(d+3))&1) for d in range(3)] for i in range(64)]
    result=[[a+b for a,b in zip(p,c)] for c in cosets for p in base]
    assert len(result)==len({tuple(p) for p in result})==512
    assert all(all(0<=a<8 for a in p) for p in result)
    return result


def save_state(path,array,views):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:np.savez(f,sum_probability=array,views_done=np.array(views,np.int64))
    os.replace(tmp,path)


def state_count(path):
    if not path.exists():return 0
    with np.load(path) as value:return int(value['views_done'])


def reserve_bytes(required):
    main=read(R/'MAIN_SCRATCH_BUDGET.json')['bytes']
    while True:
        guard();used=stable_size(TMP)+stable_size(CACHE)+main
        if used+required+256*2**20<=300*2**30:return
        status('phase512_waiting_for_scratch',used_gib=used/2**30,reserve_gib=required/2**30);time.sleep(10)


@torch.inference_mode()
def gate(model,flip=0):
    class Identity:
        def __call__(self,x):
            class Output:pass
            out=Output();out.segmentation_logits=x;return out
    x=torch.arange(8*16*24,device='cuda').reshape(1,1,8,16,24).float()
    for phase in phase_order():assert torch.equal(forward_view(Identity(),x,dict(padding_extra=[16]*3,phase=phase),flip),x)
    samples=torch.rand(512,7,9,11,device='cuda');direct=torch.zeros_like(samples[0]);resumed=torch.zeros_like(direct)
    path=TMP/'phase_state_gate_state.npz';TMP.mkdir(exist_ok=True)
    for i in range(512):
        direct.add_(samples[i]);resumed.add_(samples[i])
        if i+1 in SNAPSHOTS:
            save_state(path,resumed.cpu().numpy(),i+1)
            with np.load(path) as state:
                assert int(state['views_done'])==i+1
                resumed=torch.from_numpy(state['sum_probability']).cuda()
    assert torch.equal(direct,resumed);path.unlink()
    model.eval();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();image=torch.zeros((8,1,*CANVAS),device='cuda')
    for phase in [[0,0,0],[7,7,7]]:
        z=forward_view(model,image,dict(padding_extra=[16]*3,phase=phase),flip)
        assert z.shape==image.shape and torch.isfinite(z).all()
    result=dict(at=now(),status='passed',integer_roundtrips=512,checkpoint_roundtrip_exact=True,full_batch=8,extreme_phases_finite=True,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,patient_model_forward=False)
    assert result['peak_reserved_gib']<50
    return result


@torch.inference_mode()
def execute(model,job):
    verify_job(job);phases=phase_order();assert phases==job['phases'];flip=int(job['spec'].get('flip',0));model.eval()
    jobdir=R/'jobs'/job['id'];jobdir.mkdir(parents=True,exist_ok=True)
    if not (jobdir/'GATE.json').exists():write(jobdir/'GATE.json',gate(model,flip))
    rows,geometry,images,_=data();state_root=TMP/'phase_state'/job['id'];state_root.mkdir(parents=True,exist_ok=True)
    one_map=sum(int(np.prod(geometry[r['case_id']]['extent'])) for r in rows)*4
    # Reserve five probability snapshots (including padded single-phase control) and one atomic current state.
    allocated=stable_size(state_root)+sum(stable_size(TMP/'groups'/(job['id']+f'_n{n}')) for n in SNAPSHOTS if (TMP/'groups'/(job['id']+f'_n{n}')).exists())
    reserve_bytes(max(0,(len(SNAPSHOTS)+1)*one_map+270*4096-allocated))
    append_log('ACCESS_LEDGER.jsonl',dict(at=now(),job=job['id'],role='complete512 integer phase average',
          GT_input=False,model_gradients=False,checkpoint_sha256=CHECKPOINT_SHA,source_sha256=job['source_sha256']))
    start=time.monotonic();all_new=0
    for previous,limit in zip([0]+SNAPSHOTS[:-1],SNAPSHOTS):
        identifier=job['id']+f'_n{limit}';outputs=TMP/'groups'/identifier/'mean_probability';outputs.mkdir(parents=True,exist_ok=True)
        child=R/'jobs'/identifier;child.mkdir(parents=True,exist_ok=True)
        source=dict(id=identifier,parent=job['id'],handler='phase512_snapshot',views=[flip]*limit,
                    spec=dict(padding_extra=[16]*3,flip=flip,phase_prefix=limit,phase_order_sha256=sha(R/job.get('source_queue','queue_aux_v1')/(job['id']+'.json'))))
        write(child/'SOURCE.json',source)
        if (child/'DONE.json').exists():continue
        added=0;repeated=0
        for first in range(0,270,8):
            guard();slots=list(range(first,min(first+8,270)))
            counts=[state_count(state_root/(rows[i]['case_id']+'.npz')) for i in slots]
            assert all(n in [previous,limit] for n in counts),(counts,previous,limit)
            pending=[n<limit for n in counts]
            if not any(pending):
                assert all((outputs/(rows[i]['case_id']+'.npy')).exists() for i in slots);continue
            image=torch.from_numpy(np.array(images[slots],np.float32))[:,None].cuda();total=torch.zeros_like(image)
            for k,i in enumerate(slots):
                if pending[k] and previous:
                    with np.load(state_root/(rows[i]['case_id']+'.npz')) as state:
                        total[(k,0)+slices(geometry[rows[i]['case_id']])]=torch.from_numpy(state['sum_probability']).cuda()
            for vi in range(previous,limit):
                if (vi-previous)%8==0:guard()
                z=forward_view(model,image,dict(padding_extra=[16]*3,phase=phases[vi]),flip);assert torch.isfinite(z).all()
                p=z.sigmoid();total.add_(p);added+=len(slots);all_new+=len(slots)
                repeated+=sum(not a for a in pending)
                if (vi-previous)%32==31:status('phase512_forward',job=job['id'],prefix=limit,case_start=first,view_in_prefix=vi+1,new_case_forwards=all_new,seconds=time.monotonic()-start)
            for k,i in enumerate(slots):
                if not pending[k]:continue
                cid=rows[i]['case_id'];value=total[(k,0)+slices(geometry[cid])].cpu().numpy()
                save_array(outputs/(cid+'.npy'),value/np.float32(limit))
                save_state(state_root/(cid+'.npz'),value,limit)
            status('phase512_snapshot',job=job['id'],prefix=limit,completed=min(first+8,270),total=270,new_case_forwards=all_new,seconds=time.monotonic()-start)
            del image,total,p,z
        files=[dict(path=str(p.relative_to(TMP)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(outputs.glob('*.npy'))]
        assert len(files)==270
        write(child/'DONE.json',dict(at=now(),state='complete',job=identifier,parent=job['id'],scans=270,views=[flip]*limit,
              deployment_mean_case_forwards=limit,new_case_forwards_this_stage=added,repeated_case_forwards_this_stage=repeated,
              probability_only=True,files=files,scratch_bytes=sum(x['bytes'] for x in files),test_tuned=True))
    write(jobdir/'DONE.json',dict(at=now(),state='complete',job=job['id'],scans=270,phases=512,flip=flip,
          new_case_forwards=all_new,seconds=time.monotonic()-start,snapshot_ids=[job['id']+f'_n{n}' for n in SNAPSHOTS],test_tuned=True))
    model.eval();torch.cuda.empty_cache()
