"""RACE-specific dynamic canvas predictor and image-only reusable decoder."""
import copy,itertools
import numpy as np
import torch
from scripts.race_v2_inference_a_20261029.common import *
from scripts.race_inference_test270_20261029.predictor import flip_inputs
from scripts.segresnet_auto_inference12h_20261028.promote_v6 import single_cut
from scripts.segresnet_auto_inference12h_20261028.brain_domain_v1 import supports
from scripts.segresnet_auto_inference12h_20261028.component_sweep_v1 import apply
from scripts.segresnet_auto_inference12h_20261028.postprocess_sweep_v1 import apply_global

def definitions():
    legacy=read(OLD/'SELECTED_RECIPE.json')
    def entry(name,config,mode='volume'):
        return dict(id=name,mode=mode,config=config,neural_case_forwards=1 if mode=='raw' else 8)
    out=[entry('raw_identity_t05',None,'raw'),entry('equal8_logit_t05',None,'equal'),entry('legacy_full',copy.deepcopy(legacy))]
    for vote,exponent in itertools.product([.1,.2,.3],[8.,24.]):
        if (vote,exponent)==(.2,24.):continue
        c=copy.deepcopy(legacy);c['aggregation']['base'].update(vote_threshold=vote,exponent=exponent)
        out.append(entry(f'consistency_vote{vote}_exp{exponent}',c))
    for temperature,bias in itertools.product([.75,1.125,1.5],[0.,.375,.75]):
        if (temperature,bias)==(1.125,.375):continue
        c=copy.deepcopy(legacy);c['decoder'].update(post_temperature=temperature,bias=bias)
        out.append(entry(f'calibration_temp{temperature}_bias{bias}',c))
    for beta,scale in itertools.product([.4,.5,.65],[.9,1.,1.1,1.2]):
        if (beta,scale)==(.5,1.):continue
        c=copy.deepcopy(legacy);c['decoder'].update(beta=beta,volume_scale=scale)
        out.append(entry(f'volume_beta{beta}_scale{scale}',c))
    for minimum,hole in itertools.product([0,128,512],[0,1000]):
        if (minimum,hole)==(512,1000):continue
        c=copy.deepcopy(legacy);c['postprocessing']=[]
        if minimum:c['postprocessing'].append(dict(kind='minimum',connectivity=3,value=minimum,protect_largest=False))
        if hole:c['postprocessing'].append(dict(operation='fill_holes',source='selected',value=hole))
        out.append(entry(f'post_min{minimum}_hole{hole}',c))
    assert len(out)==32 and len({r['id'] for r in out})==32
    return out

def load_model(device='cuda'):
    from sammed3d.race_seg import RaceSeg
    from scripts.followup_test270_common import numeric_mode
    numeric_mode();assert sha(CHECKPOINT)==CHECKPOINT_SHA
    state=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    assert state['epoch']==164 and state['arm']=='race'
    model=RaceSeg(width=32,dropout=.1,enabled=True)
    model.load_state_dict(state['model_state_dict'],strict=True)
    assert all(torch.equal(v,state['model_state_dict'][k]) and bool(v.isfinite().all()) for k,v in model.state_dict().items())
    return model.to(device).eval().requires_grad_(False)

@torch.inference_mode()
def forward_view(model,image,geometry,flip,canvas):
    assert image.ndim==5 and image.shape[1]==1 and len(image)==len(geometry)
    assert tuple(image.shape[-3:])==tuple(canvas) and all(n%8==0 for n in canvas)
    assert bool(image.isfinite().all())
    for g in geometry:
        assert all(0<=o and 0<n and o+n<=s for o,n,s in zip(g['canvas_offset'],g['extent'],canvas))
    reflection=torch.tensor([g['reflection'] for g in geometry],device=image.device,dtype=torch.float32)
    valid=(image!=0).float()
    for i,g in enumerate(geometry):
        if not g['reference_enabled']:valid[i].zero_()
    view,matrix,support,dims=flip_inputs(image,reflection,valid,flip)
    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(view,matrix,support).segmentation_logits
    out=out.float()
    if dims:out=torch.flip(out,dims)
    assert bool(out.isfinite().all())
    return [out[(i,0)+slices(g)].contiguous().cpu().numpy() for i,g in enumerate(geometry)]

def probability(logits,recipe,cache):
    """Same arithmetic as historical case_consistency, without unused temperatures."""
    base=recipe['aggregation']['base'];key=(base['vote_threshold'],base['exponent'],base['view_temperature'])
    if key in cache:return cache[key]
    tensors=cache['tensors'];stack=cache['stack'];mean=cache['mean']
    votes=stack>base['vote_threshold'];reference=mean>base['vote_threshold']
    tp=(votes&reference[None]).flatten(1).sum(1).double()
    den=votes.flatten(1).sum(1).double()+reference.sum()
    weights=torch.softmax((2*tp/den.clamp_min(1))*base['exponent'],0).tolist()
    z=torch.zeros_like(tensors[0])
    for i in range(8):z.add_(tensors[i],alpha=weights[i])
    p=(z/base['view_temperature']).sigmoid().numpy()
    cache[key]=(p,weights)
    return p,weights

def decode_all(logits,image,recipes):
    """No labels or case identities accepted; GT enters subsequent scoring only."""
    assert len(logits)==8 and all(z.shape==image.shape for z in logits)
    tensors=[torch.from_numpy(np.asarray(z,np.float32)) for z in logits]
    stack=torch.stack([z.sigmoid() for z in tensors]);cache=dict(tensors=tensors,stack=stack,mean=stack.mean(0))
    raw=logits[0]>0;rawpoint=list(map(int,np.unravel_index(int(logits[0].argmax()),image.shape)))
    equal=torch.zeros_like(tensors[0])
    for z in tensors:equal.add_(z,alpha=.125)
    valid=supports(image)[0];results=[];masks={}
    for recipe in recipes:
        if recipe['mode']=='raw':mask=raw;point=rawpoint;cut=None;weights=[1.]+[0.]*7
        elif recipe['mode']=='equal':
            mask=equal.numpy()>0;point=list(map(int,np.unravel_index(int(equal.argmax()),image.shape)));cut=None;weights=[.125]*8
        else:
            config=recipe['config'];p,weights=probability(logits,config,cache)
            point=list(map(int,np.unravel_index(int(p.argmax()),image.shape)))
            codes=(p.clip(0,1)*np.float32(65535)+np.float32(.5)).astype(np.int64)
            cut=single_cut(np.bincount(codes[valid],minlength=65536),config['decoder'])
            mask=(codes>=cut)&valid
            for rule in config['postprocessing']:
                if rule.get('kind') in ('minimum','relative','largest'):mask=apply(mask,rule)
                else:mask=apply_global(mask,raw,rule,{'selected':mask})
        results.append(dict(mask=mask,automatic_point=point,quantized_cut=cut,view_weights=weights))
    return results

def numerical_gate():
    from scripts.segresnet_auto_inference12h_20261028.deploy_core_v3 import aggregate
    torch.manual_seed(41);zs=[torch.randn(8,9,10) for _ in range(8)]
    stack=torch.stack([z.sigmoid() for z in zs]);cache=dict(tensors=zs,stack=stack,mean=stack.mean(0))
    c=read(OLD/'SELECTED_RECIPE.json');p,w=probability([z.numpy() for z in zs],c,cache)
    exact,_=aggregate(dict(enumerate(zs)),c['aggregation'],torch.ones_like(zs[0]))
    assert np.array_equal(p,exact.numpy()),float(np.max(abs(p-exact.numpy())))
    image=torch.arange(2*1*8*16*24).reshape(2,1,8,16,24).float()
    r=torch.eye(4).repeat(2,1,1);r[:,0,0]=-1;r[:,0,3]=6
    support=(image%3==0).float()
    for flip in range(8):
        x,m,v,dims=flip_inputs(image,r,support,flip)
        assert torch.equal(torch.flip(x,dims),image) and torch.equal(torch.flip(v,dims),support)
        transform=torch.eye(4).repeat(2,1,1)
        for axis in range(3):
            if flip&(1<<axis):transform[:,axis,axis]=-1;transform[:,axis,3]=image.shape[axis+2]-1
        assert torch.equal(m,transform@r@transform)
    return dict(at=now(),status='passed',legacy_probability_CPU_bit_exact=True,mirror_conjugation_and_support_all8=True,dynamic_canvas=True,GT_to_deployment=False,recipes=len(definitions()))
