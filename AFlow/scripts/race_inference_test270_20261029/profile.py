"""Isolated full-shape synthetic batch/8-view probes; never reads patient arrays."""
import argparse
import sys
import traceback
import numpy as np
import torch
from scripts.race_inference_test270_20261029.common import *
from scripts.race_inference_test270_20261029.predictor import load_model,forward_view,decode_outputs


def main(rank,batch):
    verify('PREDICTOR_FREEZE.json');TMP.mkdir(parents=True,exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(.95)
    model=load_model();image=torch.ones((batch,1,*CANVAS),device='cuda')
    matrix=np.eye(4,dtype=np.float32);matrix[0,0]=-1;matrix[0,3]=159
    g=dict(canvas_offset=[0,0,0],extent=list(CANVAS),reflection=matrix.tolist(),reference_enabled=True)
    geometry=[dict(g) for _ in range(batch)]
    if batch>1:geometry[-1]=dict(g,reference_enabled=False)
    forward_view(model,image,geometry,0)
    torch.cuda.synchronize();started=time.monotonic()
    outputs=[forward_view(model,image,geometry,flip) for flip in range(8)]
    torch.cuda.synchronize();seconds=time.monotonic()-started
    decoded=decode_outputs([view[0] for view in outputs],np.ones(CANVAS,np.float32),read(R/'SELECTED_RECIPE.json'))
    assert all(v['mask'].dtype==bool and v['mask'].shape==CANVAS for v in decoded.values())
    tail=135%batch or batch
    if tail!=batch:
        for flip in range(8):assert len(forward_view(model,image[:tail],geometry[:tail],flip))==tail
    peak=torch.cuda.max_memory_reserved()/2**30
    result=dict(at=now(),status='passed' if peak<=38 else 'over_memory',rank=rank,batch=batch,
                tail=tail,tail_checked=True,seconds_8views_and_host_copy=seconds,
                scans_per_second=batch/seconds,peak_reserved_gib=peak,decoder_output_checked=True,
                patient_model_forwards=0,checkpoint_sha256=CHECKPOINT_SHA)
    write(R/f'PROFILE_rank{rank}_batch{batch}.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);p.add_argument('--batch',type=int,required=True);a=p.parse_args()
    try:main(a.rank,a.batch)
    except torch.cuda.OutOfMemoryError:
        write(R/f'PROFILE_rank{a.rank}_batch{a.batch}.json',dict(at=now(),status='oom',rank=a.rank,batch=a.batch,patient_model_forwards=0))
    except BaseException as error:
        traceback.print_exc();write(R/f'PROFILE_rank{a.rank}_batch{a.batch}.json',dict(at=now(),status='failed',error=repr(error),patient_model_forwards=0));sys.stdout.flush();sys.stderr.flush();os._exit(1)
    sys.stdout.flush();sys.stderr.flush();os._exit(0)
