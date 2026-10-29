"""Frozen RACE predictor and fixed transferred decoder; no GT or case-ID input."""
import numpy as np
import torch
from sammed3d.race_seg import RaceSeg
from scripts.race_inference_test270_20261029.common import *


def clean_geometry(g):
    return {k:g[k] for k in ('canvas_offset','extent','reflection','reference_enabled')}


def flip_inputs(image,reflection,valid,flip):
    dims=tuple(d+2 for d in range(3) if flip&(1<<d))
    transform=torch.eye(4,device=image.device,dtype=reflection.dtype).repeat(len(image),1,1)
    for axis in range(3):
        if flip&(1<<axis):transform[:,axis,axis]=-1;transform[:,axis,3]=image.shape[axis+2]-1
    matrix=transform@reflection@transform
    return (torch.flip(image,dims) if dims else image),matrix,(torch.flip(valid,dims) if dims else valid),dims


def load_model():
    numeric_mode();torch.set_num_threads(2)
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    assert saved['epoch']==164 and saved['arm']=='race'
    model=RaceSeg(width=32,dropout=.1,enabled=True).cuda()
    model.load_state_dict(saved['model_state_dict'],strict=True)
    assert all(torch.equal(v.detach().cpu(),saved['model_state_dict'][k]) and bool(v.isfinite().all()) for k,v in model.state_dict().items())
    model.eval().requires_grad_(False)
    return model


@torch.inference_mode()
def forward_view(model,image,geometries,flip):
    assert len(image)==len(geometries) and image.ndim==5 and image.shape[1]==1
    assert tuple(image.shape[-3:])==CANVAS and bool(image.isfinite().all())
    reflection=torch.tensor([g['reflection'] for g in geometries],dtype=torch.float32,device=image.device)
    valid=(image!=0).float()
    for i,g in enumerate(geometries):
        assert set(g)=={'canvas_offset','extent','reflection','reference_enabled'}
        if not g['reference_enabled']:valid[i].zero_()
    view,matrix,support,dims=flip_inputs(image,reflection,valid,flip)
    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(view,matrix,support).segmentation_logits
    out=out.float()
    if dims:out=torch.flip(out,dims)
    assert bool(out.isfinite().all())
    return [out[(i,0)+slices(g)].contiguous().cpu().numpy() for i,g in enumerate(geometries)]


@torch.inference_mode()
def decode_outputs(logits,image,config):
    """Only eight aligned predictions and input image determine masks/points."""
    from scripts.segresnet_auto_inference12h_20261028.deploy_core_v3 import aggregate
    from scripts.segresnet_auto_inference12h_20261028.promote_v6 import single_cut
    from scripts.segresnet_auto_inference12h_20261028.brain_domain_v1 import supports
    from scripts.segresnet_auto_inference12h_20261028.component_sweep_v1 import apply
    from scripts.segresnet_auto_inference12h_20261028.postprocess_sweep_v1 import apply_global
    assert len(logits)==8 and all(x.shape==image.shape for x in logits)
    tensors={i:torch.from_numpy(np.asarray(z,np.float32)).cuda() for i,z in enumerate(logits)}
    baseline=np.asarray(logits[0])>0
    baseline_point=list(map(int,np.unravel_index(int(np.argmax(logits[0])),image.shape)))
    p,point_p=aggregate(tensors,config['aggregation'],torch.from_numpy(np.asarray(image,np.float32)).cuda())
    point=list(map(int,np.unravel_index(int(point_p.argmax()),image.shape)))
    valid=supports(image)[config['aggregation']['margin']]
    codes=(p.clamp(0,1)*65535+.5).long().cpu().numpy()
    cut=single_cut(np.bincount(codes[valid],minlength=65536),config['decoder'])
    enhanced=(codes>=cut)&valid
    original=(tensors[0].sigmoid()>.5).cpu().numpy()
    for rule in config['postprocessing']:
        if rule.get('kind') in ('minimum','relative','largest'):enhanced=apply(enhanced,rule)
        else:enhanced=apply_global(enhanced,original,rule,{'selected':enhanced})
    return {'raw':dict(mask=baseline,automatic_point=baseline_point),
            'enhanced':dict(mask=enhanced,automatic_point=point,quantized_cut=int(cut))}
