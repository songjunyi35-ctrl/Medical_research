"""Cache only frozen prediction masks/reference masks for CPU shape fusion."""
import os
import numpy as np
import torch
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_cached_v1 import guard


@torch.inference_mode()
def main():
    rows,geometry,_,_=data();out=TMP/'shape_fusion_inputs';out.mkdir(exist_ok=True);records=[]
    for i,row in enumerate(rows):
        guard();cid=row['case_id'];logits=[torch.from_numpy(np.load(TMP/'groups/r1_0000_core_mirror8/atoms'/f'flip{j}'/(cid+'.npy'))).cuda() for j in range(8)]
        p=torch.stack([z.sigmoid() for z in logits]);mean=p.mean(0);payload={}
        for name,threshold in [('t20',.2),('t50',.5),('t80',.8)]:
            payload[name]=np.packbits((p>threshold).cpu().numpy().reshape(8,-1),axis=1)
            payload[name+'_reference']=np.packbits((mean>threshold).cpu().numpy().ravel())
        point=list(map(int,np.unravel_index(int(mean.argmax()),mean.shape)));path=out/(cid+'.npz');temp=path.with_suffix('.tmp')
        with temp.open('wb') as f:np.savez_compressed(f,**payload,shape=np.array(mean.shape),point=np.array(point))
        os.replace(temp,path);records.append(dict(case_id=cid,subject_id=row['subject_id'],path=str(path.relative_to(TMP)),sha256=sha(path),point=point))
    write(R/'SHAPE_FUSION_INPUT_READY.json',dict(at=now(),state='complete',records=records,GT_input=False,model_forwards_added=0,source_render_done_sha256=sha(R/'jobs/r1_0000_core_mirror8/DONE.json')))
if __name__=='__main__':main()
