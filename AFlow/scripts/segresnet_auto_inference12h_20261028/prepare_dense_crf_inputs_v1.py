"""Prepare calibrated unary maps from fixed cached model predictions, never GT."""
import numpy as np
import torch
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_cached_v1 import guard
from scripts.segresnet_auto_inference12h_20261028.deploy_core_v1 import aggregate


@torch.inference_mode()
def main():
    config=read(R/'PORTABLE_RECIPE_INTERIM.json');assert config['aggregation']['kind']=='consensus'
    rows,geometry,_,_=data();out=TMP/'dense_crf_inputs';out.mkdir(parents=True,exist_ok=True);records=[]
    params=config['decoder'];assert set(['post_temperature','bias','beta','volume_scale'])<=set(params)
    for i,row in enumerate(rows):
        guard();cid=row['case_id'];path=out/(cid+'.npy')
        logits={j:torch.from_numpy(np.load(TMP/'groups/r1_0000_core_mirror8/atoms'/f'flip{j}'/(cid+'.npy'))).cuda() for j in range(8)}
        prototype=logits[0];p,point_probability=aggregate(logits,config['aggregation'],torch.zeros_like(prototype))
        q=torch.sigmoid((torch.logit(p.double().clamp(1e-7,1-1e-7))+params['bias'])/params['post_temperature']).float()
        assert torch.isfinite(q).all();save_array(path,q.cpu().numpy());point=list(map(int,np.unravel_index(int(point_probability.argmax()),point_probability.shape)))
        records.append(dict(case_id=cid,subject_id=row['subject_id'],point=point,path=str(path.relative_to(TMP)),sha256=sha(path)))
        if i%25==0:write(R/'DENSE_CRF_INPUT_STATUS.json',dict(at=now(),completed=i+1,total=270))
    write(R/'DENSE_CRF_INPUT_READY.json',dict(at=now(),state='complete',source_portable_recipe_sha256=sha(R/'PORTABLE_RECIPE_INTERIM.json'),source_candidate='0008_consensus_cc',
          unary='sigmoid((logit(clamp(weighted raw-logit probability))+bias)/temperature), float64 then float32',point_rule='same original mean8 probability argmax',GT_input=False,model_forwards_added=0,records=records))
if __name__=='__main__':main()
