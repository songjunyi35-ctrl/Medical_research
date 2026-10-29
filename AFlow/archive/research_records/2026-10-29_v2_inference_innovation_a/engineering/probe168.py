"""One fixed-recipe dynamic270-canvas synthetic gate; no patient data/model."""
import fcntl,time
import numpy as np
import torch
from scripts.race_v2_inference_innovation_a_20261029.common import *
from scripts.race_v2_inference_innovation_a_20261029.algorithm import expand_weights
from scripts.race_v2_inference_innovation_a_20261029.deployment import deploy

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0';guard();torch.set_num_threads(2)
    lock=Path('/tmp/aflow_race_v2_20261029/locks/worker1154121_gpu0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    d,h,w=torch.meshgrid(torch.arange(3.),torch.arange(4.),torch.arange(5.),indexing='ij')
    change=.001*d+.002*h+.003*w;weights=torch.full((8,3,4,5),.125);weights[0]+=change;weights[1]-=change
    expanded=expand_weights(weights,(13,17,21),4)
    a,b,c=torch.meshgrid(torch.arange(13.).div(4).clamp_max(2),torch.arange(17.).div(4).clamp_max(3),torch.arange(21.).div(4).clamp_max(4),indexing='ij')
    error=float((expanded[0]-(.125+.001*a+.002*b+.003*c)).abs().max());assert error<1e-6
    entry=read(R/'DEPLOYMENT_RECIPE.json')['innovation_entry'];torch.cuda.reset_peak_memory_stats();torch.manual_seed(44)
    logits=[torch.randn(160,208,168,device='cuda') for _ in range(8)];first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True)
    first.record();out=deploy(logits,np.ones((160,208,168),np.float32),entry);last.record();last.synchronize()
    peak=torch.cuda.max_memory_reserved()/2**30;assert peak<=16 and bool(out['logits'].isfinite().all())
    write(R/'FINAL270_GEOMETRY_SYNTHETIC_GATE.json',dict(at=now(),status='passed',canvas=[160,208,168],fixed_recipe_id=entry['id'],
        deployment_recipe_sha256=sha(R/'DEPLOYMENT_RECIPE.json'),algorithm_sha256=sha(ROOT/'scripts/race_v2_inference_innovation_a_20261029/algorithm.py'),
        physical_lattice_linear_field_max_error=error,GPU_event_seconds=first.elapsed_time(last)/1000,peak_reserved_gib=peak,
        probe_source_sha256=sha(Path(__file__)),clinical_arrays_loaded=False,network_forwards=0,new270_scoring=False))

if __name__=='__main__':main()
