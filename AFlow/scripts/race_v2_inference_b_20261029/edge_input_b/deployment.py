"""Cache decoder after exactly one chosen guided partner input; no GT or IDs."""
import numpy as np
import torch
from scripts.race_v2_inference_b_20261029.innovation_b.core import prepare,fuse

def deploy(logits_primary,logits_partner,image,recipe):
    assert recipe['kind']=='guided_input_fusion' and recipe['models']==2 and recipe['forwards_per_scan']==16
    assert len(logits_primary)==len(logits_partner)==8
    left=[torch.as_tensor(v) for v in logits_primary];right=[torch.as_tensor(v) for v in logits_partner]
    assert all(tuple(v.shape)==tuple(image.shape) and bool(v.isfinite().all()) for v in [*left,*right])
    if recipe['decoder']=='fixed075':
        score=(sum(left)/8)*.75+(sum(right)/8)*.25;alpha=torch.full_like(score,.75);gate=torch.ones_like(score)
    else:
        assert recipe['decoder']=='covariance'
        score,alpha,gate=fuse(prepare(left,right),recipe['covariance_recipe'])
    point=list(map(int,np.unravel_index(int(score.argmax()),tuple(score.shape))))
    return dict(mask=(score>0).cpu().numpy(),automatic_point=point,score=score,alpha=alpha,conflict_gate=gate)
