"""Generate w32e171 eight-view cropped logits exactly once on physical GPU1."""
import argparse
import numpy as np
import torch
from scripts.race_v2_inference_b_20261029.common import *

def load_model():
    from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
    saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    assert saved['epoch']==171
    model=AutomaticPromptLocator(center_aware=False,init_filters=32,dropout=.1)
    model.load_state_dict(saved['model_state_dict'],strict=True)
    assert sum(v.numel() for v in model.parameters())==18795969
    assert all(torch.equal(v,saved['model_state_dict'][k]) and bool(v.isfinite().all()) for k,v in model.state_dict().items())
    return model.eval().requires_grad_(False).cuda()

@torch.inference_mode()
def main(cache,synthetic_only=False):
    lock=gpu_lock(); verify(); numeric_mode(); assert torch.cuda.device_count()==1
    model=load_model()
    # Full geometry synthetic gate before any validation-image model access.
    torch.cuda.set_per_process_memory_fraction(.94); torch.cuda.reset_peak_memory_stats()
    probes=[]
    for batch in (2,4,8,1):
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        try:
            x=torch.zeros((batch,1,*CANVAS),device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16): z=model(x).segmentation_logits
            assert z.shape==x.shape and bool(z.isfinite().all())
            for flip in range(8):
                dims=tuple(a+2 for a in range(3) if flip&(1<<a))
                if dims: assert torch.equal(x,torch.flip(torch.flip(x,dims),dims))
            torch.cuda.synchronize();started=time.monotonic()
            for _ in range(3):
                with torch.autocast('cuda',dtype=torch.bfloat16): z=model(x).segmentation_logits
            torch.cuda.synchronize();seconds=time.monotonic()-started
            reserved=torch.cuda.max_memory_reserved()/2**30
            probes.append(dict(batch=batch,passed=reserved<=38,peak_reserved_gib=reserved,
                               cases_per_second=3*batch/seconds))
            del x,z
        except torch.cuda.OutOfMemoryError as error:
            probes.append(dict(batch=batch,passed=False,error=repr(error),patient_forwards=0))
            if 'x' in locals():del x
            if 'z' in locals():del z
            torch.cuda.empty_cache()
        write(R/'SYNTHETIC_PROBE_JOURNAL.json',dict(at=now(),probes=probes,patient_forwards=0))
    assert any(p['batch']==1 and p['passed'] for p in probes)
    eligible=[p for p in probes if p['batch']!=1 and p['passed']];fastest=max(p['cases_per_second'] for p in eligible)
    batch=min(p['batch'] for p in eligible if p['cases_per_second']>=.97*fastest)
    write(R/'NUMERIC_GATE.json',dict(at=now(),passed=True,probes=probes,strict_cpu_checkpoint_reload=True,
         geometry=CANVAS,bf16=True,patient_forwards=0,tail_batch=1,batch=batch,
         selection='hardware-only max throughput; smallest batch within97pct'))
    if synthetic_only: return
    index,images,_=data(cache); cohort=rows()
    exclusive(R/'MODEL_PREACCESS.json',dict(at=now(),checkpoint_sha256=CHECKPOINT_SHA,split='validation',
         count=108,subjects=50,views=list(range(8)),maximum_case_forwards=864,cache=str(cache),
         index_sha256=sha(Path(cache)/'index.json'),manifest_sha256=sha(MANIFEST),automatic_neural_retry=False))
    entries=[]; started=time.monotonic(); calls=0
    for first in range(0,108,batch):
        guard(); current=list(range(first,min(first+batch,108)))
        x=torch.from_numpy(np.array(images[current],np.float32))[:,None].cuda()
        for flip in range(8):
            guard(); dims=tuple(a+2 for a in range(3) if flip&(1<<a)); view=torch.flip(x,dims) if dims else x
            with torch.autocast('cuda',dtype=torch.bfloat16): z=model(view).segmentation_logits
            z=z.float(); z=torch.flip(z,dims) if dims else z; calls+=1
            assert bool(z.isfinite().all())
            for k,slot in enumerate(current):
                cid=cohort[slot]['case_id']; value=z[(k,0)+slices(index['geometry'][cid])].contiguous().cpu().numpy()
                path=OUT/'atoms'/f'{slot:03}'/f'flip{flip}.npy'; path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as handle: np.save(handle,value,allow_pickle=False)
                entries.append(dict(slot=slot,case_id=cid,view=flip,path=str(path),shape=list(value.shape),sha256=sha(path)))
            write(R/'FORWARD_JOURNAL.json',dict(at=now(),case_forwards=len(entries),batch_calls=calls,atoms=entries))
            write(R/'HEARTBEAT.json',dict(at=now(),stage='w32_validation_eight_views',case_forwards=len(entries),maximum=864))
        del x,z,view
    assert len(entries)==864 and len({(a['slot'],a['view']) for a in entries})==864
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    complete=dict(at=now(),status='complete',run_id='race_v2_inference_b_20261029',model='w32e171',checkpoint_sha256=CHECKPOINT_SHA,
        checkpoint_path=str(CHECKPOINT),architecture='AutomaticPromptLocator',constructor=dict(center_aware=False,init_filters=32,dropout=.1),
        new_training_epochs=0,new_training_updates=0,reference_matrix='plain SegResNet ignores RACE reference; shared native geometry/crop exact',
        manifest_sha256=sha(MANIFEST),index_sha256=sha(Path(cache)/'index.json'),cache_path=str(cache),
        canvas=list(CANVAS),geometry_contract='aligned native crop excludes centered padding',dtype='float32',
        case_forwards=864,batch_calls=calls,seconds=time.monotonic()-started,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,atoms=entries)
    write(OUT/'ATOMS_READY.json',complete); write(R/'ATOMS_READY.json',complete)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache');p.add_argument('--synthetic-only',action='store_true');a=p.parse_args()
    assert a.synthetic_only or a.cache
    try: main(a.cache,a.synthetic_only)
    except BaseException as error:
        import traceback
        write(R/f'FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc(),
            model_access=(R/'MODEL_PREACCESS.json').exists(),no_automatic_neural_retry=True));raise
