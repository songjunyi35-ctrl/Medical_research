"""Synthetic native eight-view throughput; no patient pixels or Dice selection."""
import numpy as np
import torch
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_cached_v1 import guard
from scripts.segresnet_auto_inference12h_20261028.render_v1 import forward_view


@torch.inference_mode()
def main():
    guard();numeric_mode();torch.cuda.set_per_process_memory_fraction(.85);model=load_model('cuda');measurements=[]
    # Two opposite orders reduce timing drift from the independent scoring lane.
    for repeat,order in enumerate([[4,8,12,16],[16,12,8,4]]):
        for batch in order:
            guard();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();x=torch.randn((batch,1,*CANVAS),device='cuda');values=[]
            if repeat==0:
                warm=forward_view(model,x,{},0);assert torch.isfinite(warm).all();del warm
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);torch.cuda.synchronize();start.record()
            for flip in range(8):values.append(forward_view(model,x,{},flip))
            end.record();torch.cuda.synchronize();seconds=start.elapsed_time(end)/1000
            assert all(torch.isfinite(z).all() for z in values)
            measurements.append(dict(batch=batch,repeat=repeat,neural_case_forwards=batch*8,seconds=seconds,case_forwards_per_second=batch*8/seconds,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
            write(R/'HARDWARE_PROFILE_PROGRESS.json',dict(at=now(),measurements=measurements));del x,values
    grouped=[]
    for batch in [4,8,12,16]:
        rows=[m for m in measurements if m['batch']==batch];grouped.append(dict(batch=batch,median_case_forwards_per_second=float(np.median([m['case_forwards_per_second'] for m in rows])),maximum_reserved_gib=max(m['peak_reserved_gib'] for m in rows)))
    best=max(grouped,key=lambda r:r['median_case_forwards_per_second'])
    write(R/'HARDWARE_PROFILE.json',dict(at=now(),state='complete',measurements=measurements,summary=grouped,fastest_observed_batch=best['batch'],checkpoint_sha256=CHECKPOINT_SHA,checkpoint_final_sha256=sha(CHECKPOINT),source_sha256=sha(Path(__file__)),patient_access=False,GT_access=False,training=False,production_batch_changed=False,limitations='Synthetic neural-forward timing including inverse flips; excludes calibration/components/cavities/I/O. Concurrent scoring may affect timing; not exclusive-card or end-to-end latency. Verified Dice recipe remains batch8/tail6.',synthetic_neural_case_forwards=8*2*sum([4,8,12,16])+sum([4,8,12,16])))
    print(dict(state='complete',fastest_observed_batch=best['batch'],summary=grouped,production_batch_changed=False))

if __name__=='__main__':main()
