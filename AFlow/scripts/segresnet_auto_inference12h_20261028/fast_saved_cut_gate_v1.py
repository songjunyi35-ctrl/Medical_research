import numpy as np
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.fast_volume_cuts_v1 import fast_cuts
from scripts.segresnet_auto_inference12h_20261028 import calibration_parallel_extended_v1 as old

def main():
 source='b1_0000_nonlinear_fusions';destination=TMP/'analyses/b1_1000_nonlinear_calibration/cases'
 files=sorted(destination.glob('*/*.npz'));assert len(files)>=10
 chosen=[files[int(i)] for i in np.linspace(0,len(files)-1,10,dtype=int)];probes=[]
 for p in chosen:
  ri=p.parent.name;hist=TMP/'analyses'/source/'histograms'/ri/(p.stem+'.npy');counts=np.load(hist).sum(0,dtype=np.int64)
  expected=np.load(p)['cuts'];actual=fast_cuts(counts,old.calibrated_bins(len(counts)),old.BETAS,old.VOLUME_SCALES)
  assert np.array_equal(expected,actual),(ri,p.stem,np.flatnonzero(expected!=actual)[:10])
  probes.append(dict(recipe_index=int(ri),case_id=p.stem,parameters=len(expected),exact=True))
 write(R/'FAST_CUTS_EXISTING_WORKER_GATE.json',dict(at=now(),status='passed',probes=probes,cuts_compared=sum(x['parameters'] for x in probes),same_worker_CPU_gold=True,model_forwards=0));print(probes)
if __name__=='__main__':main()
