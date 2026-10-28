import torch,time,numpy as np
from scripts.segresnet_auto_inference12h_20261028.threshold_histograms_v1 import histogram,scores
from scripts.segresnet_auto_inference12h_20261028.core_fusions_v1 import threshold_scores,histogram as reference
from scripts.segresnet_auto_inference12h_20261028.common import R,write,now

def main():
 thresholds=np.round(np.arange(.025,.9751,.025),6).tolist();probes=[]
 for size in [1,2047,2048,2049,65536,4500000]:
  for kind in ['uniform','tails','all_codes']:
   if kind=='all_codes':p=(torch.arange(size,device='cuda')%65536).float()/65535
   else:p=torch.rand(size,device='cuda')
   if kind=='tails':p=torch.sigmoid(p*35-30)
   y=torch.rand_like(p)<.1
   full=reference(p,y,65536);short=histogram(p,y)
   assert np.array_equal(scores(short),threshold_scores(full,thresholds)[0]),(size,kind)
   assert np.array_equal(short.sum(1),full.sum(1))
   probes.append(dict(size=size,kind=kind,exact=True))
 p=torch.rand(4500000,device='cuda');y=torch.rand_like(p)<.1
 torch.cuda.synchronize();start=time.monotonic()
 for _ in range(10):scores(histogram(p,y))
 torch.cuda.synchronize();seconds=(time.monotonic()-start)/10
 write(R/'THRESHOLD_HISTOGRAM_GPU_GATE.json',dict(at=now(),status='passed',probes=probes,seconds=seconds,quantized_mask_definitions_unchanged=True,model_forwards=0))
 print(dict(probes=len(probes),seconds=seconds,status='passed'))
if __name__=='__main__':main()
