"""Bounded, image-only contralateral evidence for two fixed experts; no GT input."""
import numpy as np
import torch
import torch.nn.functional as F
from scripts.race_seg_20261028.prepare import fit_plane

def definitions():
    return [dict(id=f'contra_{evidence}_k{kernel}_b{beta:g}',kind='contralateral_image_evidence',
        evidence=evidence,kernel=kernel,beta=beta,anchor=.75,alpha_min=.5,alpha_max=.95,
        image_center=0.,image_scale=1.,standard_deviation_floor=.25,
        reference_probability_ceiling=.2,plane_minimum_dice=.8,direction_temperature=4.,
        threshold_logit=0.,models=2,forwards_per_scan=16)
        for evidence in ('absolute','positive_difference') for kernel in (5,11) for beta in (.1,.2)]

def box(value,kernel):
    """Separable uniform box including zero padding; ratios remove missing support."""
    value=value[None,None]
    for axis in range(3):
        sizes=[1,1,1];sizes[axis]=kernel
        pads=[0,0,0];pads[axis]=kernel//2
        value=F.avg_pool3d(value,kernel_size=sizes,stride=1,padding=pads,count_include_pad=True)
    return value[0,0]

def reflect(value,center):
    coordinates=2*int(center)-torch.arange(value.shape[0],device=value.device)
    inside=(coordinates>=0)&(coordinates<value.shape[0])
    result=value.index_select(0,coordinates.clamp(0,value.shape[0]-1))
    result=result*inside[:,None,None]
    return result,inside[:,None,None].expand_as(value)

def prepare(left,right,image,plane=None):
    assert len(left)==len(right)==8 and image.ndim==3
    assert all(z.shape==image.shape and bool(z.isfinite().all()) for z in [image,*left,*right])
    if plane is None:plane=fit_plane(image.detach().cpu().numpy())
    center,quality=plane
    reflected,inside=reflect(image,center)
    support=(image!=0)&(reflected!=0)&inside
    lm=sum(left)/8;rm=sum(right)/8;p=lm.sigmoid();q=rm.sigmoid()
    pr,_=reflect(p,center);qr,_=reflect(q,center)
    healthy=(1-torch.maximum(pr,qr)/.2).clamp(0,1)
    healthy=healthy*support*(quality>=.8)
    conflict=4*(-(p-.5)*(q-.5)).clamp_min(0)
    direction=((lm-rm)/4).tanh()
    return dict(left_mean=lm,right_mean=rm,image=image,reflected=reflected,support=support,
        healthy=healthy,conflict=conflict,direction=direction,center=float(center),plane_dice=float(quality),pooled={})

def evidence(cache,kernel,mode):
    if kernel not in cache['pooled']:
        weight=cache['support'].to(cache['image'].dtype);mass=box(weight,kernel).clamp_min(1e-8)
        a,b=cache['image'],cache['reflected']
        mu=box(a*weight,kernel)/mass;ref_mu=box(b*weight,kernel)/mass
        var=(box(a.square()*weight,kernel)/mass-mu.square()).clamp_min(0)
        ref_var=(box(b.square()*weight,kernel)/mass-ref_mu.square()).clamp_min(0)
        delta=mu-ref_mu;scale=((var+ref_var)/2+.25**2).sqrt()
        cache['pooled'][kernel]=dict(delta=delta,scale=scale)
    value=cache['pooled'][kernel];delta=value['delta']
    return ((delta.abs() if mode=='absolute' else delta)/value['scale']).tanh()

def fuse(cache,recipe):
    assert recipe['kind']=='contralateral_image_evidence'
    ev=evidence(cache,recipe['kernel'],recipe['evidence'])
    shift=recipe['beta']*cache['conflict']*cache['direction']*ev*cache['healthy']
    alpha=(recipe['anchor']+shift).clamp(recipe['alpha_min'],recipe['alpha_max'])
    score=alpha*cache['left_mean']+(1-alpha)*cache['right_mean']
    assert bool(score.isfinite().all()) and bool(alpha.isfinite().all())
    return score,alpha,cache['conflict']

def deploy(logits_primary,logits_partner,image,recipe):
    left=[torch.as_tensor(z) for z in logits_primary];right=[torch.as_tensor(z) for z in logits_partner]
    image=torch.as_tensor(image,device=left[0].device)
    cache=prepare(left,right,image);score,alpha,conflict=fuse(cache,recipe)
    point=list(map(int,np.unravel_index(int(score.argmax()),tuple(score.shape))))
    return dict(mask=(score>0).cpu().numpy(),automatic_point=point,score=score,alpha=alpha,conflict_gate=conflict)

def cpu_gate():
    shape=(24,17,19);x=torch.arange(24.)[:,None,None];image=(1+(x-11).square()/100).expand(shape).clone()
    plane=fit_plane(image.numpy());center=plane[0]
    reflected,inside=reflect(image,center)
    matrix=np.eye(4);matrix[0,0]=-1;matrix[0,3]=2*center
    points=np.array([[3,6,8,1],[7,4,9,1]],dtype=float).T
    assert np.array_equal((matrix@points)[0],2*center-points[0])
    offset=13.;translate=np.eye(4);translate[0,3]=offset
    canvas_reflection=translate@matrix@np.linalg.inv(translate)
    assert np.allclose(np.linalg.inv(translate)@canvas_reflection@translate,matrix)
    assert not np.allclose(canvas_reflection,matrix)
    twice,_=reflect(reflected,center);assert torch.equal(twice[inside],image[inside])
    # Make the mirror site healthy while the inspected side has conflicting experts.
    l=torch.full(shape,-5.);r=l.clone();l[4:9]=2.;r[4:9]=-2.
    left=[l]*8;right=[r]*8
    symmetric=torch.ones(shape);cache=prepare(left,right,symmetric)
    assert all(torch.equal(fuse(cache,c)[1],torch.full(shape,.75)) for c in definitions())
    asymmetric=symmetric.clone();asymmetric[4:9]=2.
    positive=prepare(left,right,asymmetric);negative=prepare(left,right,3-asymmetric)
    tested=[]
    for recipe in definitions():
        score,alpha,_=fuse(positive,recipe);assert bool(((alpha>=.5)&(alpha<=.95)).all())
        assert bool((alpha[4:9]>.75).any())
        swapped=prepare(right,left,asymmetric);sa=fuse(swapped,recipe)[1]
        assert torch.allclose(alpha-.75,-(sa-.75),atol=1e-7)
        na=fuse(negative,recipe)[1]
        if recipe['evidence']=='positive_difference':assert torch.allclose(alpha-.75,-(na-.75),atol=2e-6)
        else:assert torch.allclose(alpha,na,atol=2e-6)
        bad=prepare(left,right,asymmetric,plane=(positive['center'],.79))
        assert torch.equal(fuse(bad,recipe)[1],torch.full(shape,.75))
        lesion_ref=prepare([torch.full(shape,2.)]*8,right,asymmetric)
        assert torch.equal(fuse(lesion_ref,recipe)[1],torch.full(shape,.75))
        zero_support=asymmetric.clone();zero_support[14:20]=0
        absent=prepare(left,right,zero_support,plane=(positive['center'],1.))
        assert bool((fuse(absent,recipe)[1][~absent['support']]==.75).all())
        exported=deploy(left,right,asymmetric,recipe);assert torch.equal(exported['score'],score)
        assert exported['automatic_point']==list(np.unravel_index(int(score.argmax()),shape))
        tested.append(recipe['id'])
    return dict(status='passed',configurations=tested,crop_matrix_and_canvas_conjugation=True,
        support_reflection_roundtrip=True,symmetric_evidence_anchor=True,swapped_model_shift_sign_reversed=True,
        signed_negative_difference_reverses_shift=True,absolute_difference_even=True,
        unhealthy_or_missing_reference_fallback=True,finite_alpha_bounds=True,actual_score_argmax=True,
        patient_arrays_loaded=False,new_neural_forwards=0)
