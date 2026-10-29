"""Image/seed-only conservative normalization; one seed plus seven new views."""
import itertools,math
import numpy as np
import torch
from scipy.ndimage import maximum_filter
from scripts.race_inference_test270_20261029.predictor import flip_inputs

def groups():
    return [dict(id=f'healthy_q{q}_alpha{alpha}',exclude_probability=q,alpha=alpha,
        dilation_voxels=3,dilation_metric='Chebyshev',minimum_healthy_fraction=.2,
        minimum_standard_deviation=.1,gain_bounds=[.85,1.15],offset_bounds=[-.15,.15])
        for q,alpha in itertools.product([.1,.5],[.5,1.])]

def definitions():
    return [dict(id=f"{g['id']}_seed{weight}",kind='race_seed_healthy_normalization',
        input_group=g,seed_weight=weight,other_views=list(range(1,8)),
        output_logit_threshold=0.,point_rule='actual_final_score_argmax',neural_case_forwards=8)
        for g,weight in itertools.product(groups(),[.125,.5])]

def coefficients(image,seed,recipe):
    """No target, case ID, cohort template, or learned statistics are accepted."""
    assert image.shape==seed.shape and image.ndim==3
    assert np.isfinite(image).all() and np.isfinite(seed).all()
    support=image!=0
    cut=math.log(recipe['exclude_probability']/(1-recipe['exclude_probability']))
    suspected=maximum_filter((seed>=cut).astype(np.uint8),size=7,mode='constant',cval=0)>0
    healthy=support&~suspected
    n=int(support.sum());count=int(healthy.sum());fraction=count/max(n,1)
    x=np.asarray(image[healthy],np.float64)
    mean=float(x.mean()) if len(x) else 0.;sd=float(x.std()) if len(x) else 0.
    fallback=fraction<recipe['minimum_healthy_fraction'] or sd<recipe['minimum_standard_deviation'] or n==0
    gain=1. if fallback else float(np.clip(1./sd,*recipe['gain_bounds']))
    offset=0. if fallback else float(np.clip(-mean/sd,*recipe['offset_bounds']))
    alpha=recipe['alpha'];a=1.+alpha*(gain-1.);b=alpha*offset
    return dict(gain=a,offset=b,unblended_gain=gain,unblended_offset=offset,
        healthy_mean=mean,healthy_standard_deviation=sd,healthy_fraction=fraction,
        support_voxels=n,healthy_voxels=count,suspected_support_voxels=int((support&suspected).sum()),
        identity_fallback=bool(fallback),alpha=alpha)

def transform(image,support,coefficient):
    """Original support is carried separately, including affine zero crossings."""
    return torch.where(support,image*coefficient['gain']+coefficient['offset'],torch.zeros_like(image))

@torch.inference_mode()
def forward_view(model,image,original_support,geometry,flip,canvas):
    assert image.ndim==5 and tuple(image.shape[-3:])==tuple(canvas)
    assert original_support.shape==image.shape and bool(image.isfinite().all())
    reflection=torch.tensor([g['reflection'] for g in geometry],device=image.device,dtype=torch.float32)
    valid=original_support.float().clone()
    for i,g in enumerate(geometry):
        assert all(0<=o and 0<n and o+n<=s for o,n,s in zip(g['canvas_offset'],g['extent'],canvas))
        if not g['reference_enabled']:valid[i].zero_()
    view,matrix,support,dims=flip_inputs(image,reflection,valid,flip)
    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(view,matrix,support).segmentation_logits
    out=out.float()
    if dims:out=torch.flip(out,dims)
    assert bool(out.isfinite().all())
    return [out[(i,0)+tuple(slice(int(a),int(a)+int(n)) for a,n in zip(g['canvas_offset'],g['extent']))].contiguous().cpu().numpy() for i,g in enumerate(geometry)]

def decode(seed,adapted,recipe):
    """Seven aligned new logits and one original seed; GT/ID-free."""
    assert len(adapted)==7 and all(z.shape==seed.shape for z in adapted)
    out=seed*recipe['seed_weight']
    for z in adapted:out=out+z*((1-recipe['seed_weight'])/7.)
    point=list(map(int,np.unravel_index(int(out.argmax()),tuple(out.shape))))
    return out,point

def numerical_gate():
    rng=np.random.default_rng(20261029);image=rng.normal(size=(19,21,23)).astype(np.float32);image[:2]=0
    seed=np.full_like(image,-5.);seed[8:10,9:11,10:12]=3.
    checks=[]
    for g in groups():
        c=coefficients(image,seed,g);value=transform(torch.from_numpy(image),torch.from_numpy(image!=0),c).numpy()
        expected=np.where(image!=0,image*np.float32(c['gain'])+np.float32(c['offset']),0)
        assert np.array_equal(value,expected) and not value[:2].any() and c['gain']>0
        checks.append(dict(group=g['id'],coefficient=c,finite_and_preserves_original_support=True))
    empty=coefficients(image,np.full_like(seed,99),groups()[0]);assert empty['identity_fallback'] and empty['gain']==1 and empty['offset']==0
    zs=[torch.from_numpy(rng.normal(size=image.shape).astype(np.float32)) for _ in range(8)]
    for recipe in definitions():
        out,point=decode(zs[0],zs[1:],recipe)
        assert bool(out.isfinite().all()) and len(point)==3
    z=torch.zeros(3,4,5);z[0,0,0]=0
    support=torch.ones_like(z,dtype=torch.bool)
    crossed=transform(z,support,dict(gain=1,offset=0));assert support[0,0,0] and crossed[0,0,0]==0
    return dict(status='passed',checks=checks,all_excluded_identity_fallback=True,
        separate_original_reference_support=True,deploy_target_or_ID_arguments=False,
        deployment_total_neural_forwards=8,configuration_count=8)
