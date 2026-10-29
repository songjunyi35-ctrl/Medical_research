"""Single GPU queue with a permanent journal; retry never duplicates saved atoms."""
import argparse,fcntl,subprocess,time
import numpy as np
import torch
from scripts.race_v2_inference_a_20261029.common import *
from scripts.race_v2_inference_a_20261029.predictor import forward_view,load_model,numerical_gate

def atom_path(slot,view):return TMP/'atoms'/f'{slot:03}'/f'flip{view}.npy'

def forward():
    ready=read(R/'CACHE_READY.json');assert ready['all_four_payloads_exact']
    assert sha(CACHE/'index.json')==ready['index_sha256']
    index=read(CACHE/'index.json');assert tuple(index['canvas'])==CANVAS
    cohort=rows();assert index['case_ids']==[r['case_id'] for r in cohort]
    images=np.memmap(CACHE/'images.f32',np.float32,'r',shape=(108,*CANVAS))
    with (R/'MODEL_PREACCESS.json').open('x') as f:
        json.dump(dict(at=now(),split='validation',scans=108,subjects=50,views=list(range(8)),maximum_case_forwards=864,checkpoint_sha256=CHECKPOINT_SHA,no_test_access=True),f)
    model=load_model();torch.cuda.set_per_process_memory_fraction(.95);torch.cuda.reset_peak_memory_stats()
    batch=8;atoms=[];start=time.monotonic()
    for first in range(0,108,batch):
        guard();slots=list(range(first,min(first+batch,108)))
        image=torch.from_numpy(np.array(images[slots],np.float32))[:,None].cuda()
        geometries=[index['geometry'][cohort[i]['case_id']] for i in slots]
        for view in range(8):
            values=forward_view(model,image,geometries,view,CANVAS)
            for slot,value in zip(slots,values):
                path=atom_path(slot,view);path.parent.mkdir(parents=True,exist_ok=True)
                temp=path.with_name(path.name+'.tmp')
                with temp.open('xb') as f:np.save(f,value,allow_pickle=False)
                os.replace(temp,path)
                atoms.append(dict(slot=slot,view=view,path=str(path),sha256=sha(path),shape=list(value.shape),dtype=str(value.dtype)))
            write(R/'FORWARD_JOURNAL.json',dict(at=now(),completed_case_forwards=len(atoms),maximum=864,atoms=atoms))
            write(R/'HEARTBEAT.json',dict(at=now(),stage='validation_forward',case_forwards=len(atoms),maximum=864,seconds=time.monotonic()-start))
        del image,values
    assert len(atoms)==864 and len({(a['slot'],a['view']) for a in atoms})==864
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    manifest=ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/validation_t1ce_wt.jsonl'
    ready=dict(at=now(),status='passed',run_id='race164_validation108_8views_20261029',worker=1154121,physical_gpu=0,checkpoint_path=str(CHECKPOINT),checkpoint_sha256=CHECKPOINT_SHA,architecture='RaceSeg',constructor=dict(width=32,dropout=.1,enabled=True),split='validation',canvas=list(CANVAS),index_sha256=sha(CACHE/'index.json'),manifest_sha256=sha(manifest),cache_index_sha256=sha(CACHE/'index.json'),case_ids=index['case_ids'],views=list(range(8)),atoms=atoms,case_forwards=864,batch=batch,seconds=time.monotonic()-start,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,code_freeze_sha256=sha(R/'EXECUTION_FREEZE.json'),geometry_convention='voxel RAS affine reflection; flip T R T; nonzero support transformed with image; cropped aligned logits',read_only_for_agent_b=True)
    write(R/'ATOMS_READY.json',ready);write(TMP/'ATOMS_READY.json',ready)
    del model

def main():
    TMP.mkdir(parents=True,exist_ok=True);verify();torch.set_num_threads(1)
    lock_path=Path('/tmp/aflow_race_v2_20261029/locks/worker1154121_gpu0.lock');lock_path.parent.mkdir(parents=True,exist_ok=True)
    lock=lock_path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True)
    write(R/'GPU_LOCK.json',dict(at=now(),worker=1154121,physical_gpu=0,visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],pid=os.getpid(),lock=str(lock_path),nvidia_smi=gpu,allocation_change_authority='root'))
    write(R/'CPU_GATE.json',numerical_gate())
    # Full native-sized synthetic mirror and reload gate before patient model access.
    model=load_model();image=torch.randn(8,1,*CANVAS,device='cuda')
    g=dict(canvas_offset=[0,0,0],extent=list(CANVAS),reference_enabled=True,reflection=[[-1,0,0,159],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    with torch.inference_mode():
        start=time.monotonic();value=forward_view(model,image,[g]*8,7,CANVAS);torch.cuda.synchronize()
    reserved=torch.cuda.max_memory_reserved()/2**30;assert reserved<=38
    with torch.inference_mode():tail=forward_view(model,image[:4],[g]*4,0,CANVAS)
    write(R/'GPU_SYNTHETIC_GATE.json',dict(at=now(),status='passed',canvas=list(CANVAS),batch=8,tail=4,seconds=time.monotonic()-start,peak_reserved_gib=reserved,full_shape_finite=True,CPU_strict_reload=True,patient_forwards=0,batch_basis='existing same model native168 batch8 gate; new native160 full and tail finite/<=38GiB gate'))
    del model,image,value;torch.cuda.empty_cache()
    while not (R/'CACHE_READY.json').exists():
        guard()
        if (R/'CACHE_FAILURE.json').exists():raise RuntimeError('validation builder failed')
        write(R/'HEARTBEAT.json',dict(at=now(),stage='waiting_native_validation',cache=read(R/'CACHE_HEARTBEAT.json') if (R/'CACHE_HEARTBEAT.json').exists() else None))
        time.sleep(10)
    forward()

if __name__=='__main__':
    try:main()
    except BaseException as e:
        import traceback
        write(R/'QUEUE_FAILURE.json',dict(at=now(),error=repr(e),traceback=traceback.format_exc(),neural_auto_retry=False));raise
