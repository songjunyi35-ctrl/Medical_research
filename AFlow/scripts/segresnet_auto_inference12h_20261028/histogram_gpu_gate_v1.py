import torch,time,numpy as np
from scripts.segresnet_auto_inference12h_20261028.histograms_triton_v1 import histogram as triton_hist
from scripts.segresnet_auto_inference12h_20261028.core_fusions_v1 import histogram
from scripts.segresnet_auto_inference12h_20261028.common import R,write,now

def main():
 probes=[]
 for size in [1,2047,2048,2049,4500000]:
  for kind in ['uniform','tails','edges']:
   p=torch.rand(size,device='cuda')
   if kind=='tails':p=torch.sigmoid(p*30-25)
   if kind=='edges':p=(torch.randint(0,65536,(size,),device='cuda').float()-.5)/65535
   y=torch.rand_like(p)<.1
   ref=histogram(p,y,65536);actual=triton_hist(p,y)
   assert np.array_equal(ref,actual),(size,kind,np.abs(ref-actual).sum())
   ts=[]
   for f in [histogram,triton_hist]:
    torch.cuda.synchronize();t=time.monotonic()
    for i in range(3):f(p,y,65536)
    torch.cuda.synchronize();ts.append((time.monotonic()-t)/3)
   probes.append(dict(size=size,kind=kind,exact=True,reference_seconds=ts[0],triton_seconds=ts[1]));print(probes[-1],flush=True)
 write(R/'HISTOGRAM_TRITON_GPU_GATE.json',dict(at=now(),status='passed',probes=probes,model_forwards=0,preclinical_failure='negative sentinel wraps under installed Triton3.3; fixed by invalid-bin count subtraction before any clinical use'))
if __name__=='__main__':main()
