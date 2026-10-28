import numpy as np
from scripts.segresnet_auto_inference12h_20261028.common import *

def main():
 source='b1_0000_nonlinear_fusions';shortlist=read(R/'analyses'/source/'RANKING.json')['calibration_shortlist'];ids=read(CACHE/'index.json')['case_ids'];counts=[];records=[]
 for ri in [shortlist[i]['recipe_index'] for i in [0,len(shortlist)//2,len(shortlist)-1]]:
  for slot in [0,64,128,192,269]:
   p=TMP/'analyses'/source/'histograms'/str(ri)/(ids[slot]+'.npy');h=np.load(p);counts.append(h.sum(0,dtype=np.int64));records.append(dict(recipe_index=ri,case_id=ids[slot],histogram_sha256=sha(p)))
 path=ROOT/'archive/research_records/2026-10-28_segresnet_auto_inference12h/fast_calibration_gate_counts.npy'
 # Temporary derived unlabeled histograms only; publication excludes this payload.
 np.save(path,np.stack(counts));write(R/'FAST_CALIBRATION_REAL_GATE_INPUTS.json',dict(at=now(),counts_sha256=sha(path),records=records,GT_labels_removed=True))
 print(path)
if __name__=='__main__':main()
