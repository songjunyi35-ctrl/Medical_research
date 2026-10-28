import torch,time,numpy as np
from scripts.segresnet_auto_inference12h_20261028.common import TMP,CACHE,read,R,write,now
from scripts.segresnet_auto_inference12h_20261028.histograms_triton_v1 import histogram as original
from scripts.segresnet_auto_inference12h_20261028.histograms_adaptive_v1 import histogram as adaptive
from scripts.segresnet_auto_inference12h_20261028.core_fusions_v1 import histogram as reference

def main():
 cid=read(CACHE/'index.json')['case_ids'][0];zs=[torch.from_numpy(np.load(TMP/'groups/r1_0000_core_mirror8/atoms'/f'flip{i}'/(cid+'.npy'))).cuda() for i in range(8)]
 probes=[]
 for kind,p in [('T32mean',torch.stack([(z/32).sigmoid() for z in zs]).mean(0)),('T1mean',torch.stack([z.sigmoid() for z in zs]).mean(0)),('constant_mid',torch.full_like(zs[0],.2)),('uniform',torch.rand_like(zs[0]))]:
  y=torch.rand_like(p)<.05;ref=reference(p,y,65536);times={}
  for name,f in [('original',original),('adaptive',adaptive)]:
   actual=f(p,y);assert np.array_equal(ref,actual),(kind,name)
   torch.cuda.synchronize();t=time.monotonic()
   for _ in range(10):f(p,y)
   torch.cuda.synchronize();times[name]=(time.monotonic()-t)/10
  d=dict(kind=kind,counts_exact=True,**times);probes.append(d);print(d,flush=True)
 write(R/'ADAPTIVE_HISTOGRAM_GPU_GATE.json',dict(at=now(),status='passed',probes=probes,model_forwards=0,wholecode_probabilities_unchanged=True))
if __name__=='__main__':main()
