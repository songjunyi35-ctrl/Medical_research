"""Paired perturbation covariance shrinkage; accepts only two aligned view sets."""
import torch
import torch.nn.functional as F

ANCHOR=.75
EPSILON=1e-4
VIEW_TEMPERATURE=4.

def definitions():
    return [dict(id=f'paired_cov_k{kernel}_ridge{ridge:g}_conflict{int(conflict)}',
        kind='paired_covariance_shrinkage',
        kernel=kernel,ridge=ridge,conflict_gate=conflict,anchor=ANCHOR,
        alpha_min=.5,alpha_max=.95,statistics_temperature=VIEW_TEMPERATURE,
        epsilon=EPSILON,singular_relative_tolerance=1e-6,zero_variance_floor=1e-8,
        threshold_logit=0.,models=2,forwards_per_scan=16)
        for kernel in (1,5) for ridge in (.25,1.,4.) for conflict in (False,True)]

def prepare(left,right):
    assert len(left)==len(right)==8
    assert all(z.shape==left[0].shape and bool(z.isfinite().all()) for z in [*left,*right])
    p=torch.stack([(z/VIEW_TEMPERATURE).sigmoid() for z in left])
    q=torch.stack([(z/VIEW_TEMPERATURE).sigmoid() for z in right])
    p_mean=p.mean(0);q_mean=q.mean(0);a=p-p_mean;b=q-q_mean
    moments=torch.stack([a.square().mean(0),b.square().mean(0),(a*b).mean(0)])
    return dict(left_mean=sum(left).div(8),right_mean=sum(right).div(8),
        probability_difference_squared=(p_mean-q_mean).square(),moments=moments,pooled={1:moments})

def statistics(cache,kernel):
    if kernel not in cache['pooled']:
        cache['pooled'][kernel]=F.avg_pool3d(cache['moments'][None],kernel_size=kernel,stride=1,
            padding=kernel//2,count_include_pad=False)[0]
    return cache['pooled'][kernel]

def fuse(cache,recipe):
    vr,vw,cov=statistics(cache,recipe['kernel'])
    difference_variance=(vr+vw-2*cov).clamp_min(0)
    ridge=recipe['ridge']*(vr+vw)+recipe['epsilon']
    alpha=((vw-cov+ridge*recipe['anchor'])/(difference_variance+ridge)).clamp(recipe['alpha_min'],recipe['alpha_max'])
    determinant=(vr*vw-cov.square()).clamp_min(0)
    singular=((vr+vw)<=recipe['zero_variance_floor']) | (determinant<=recipe['singular_relative_tolerance']*(vr*vw+recipe['epsilon']**2))
    alpha=torch.where(singular,torch.full_like(alpha,recipe['anchor']),alpha)
    if recipe['conflict_gate']:
        signal=cache['probability_difference_squared']
        gate=signal/(signal+difference_variance/8+recipe['epsilon'])
        alpha=recipe['anchor']+gate*(alpha-recipe['anchor'])
    else:gate=torch.ones_like(alpha)
    score=alpha*cache['left_mean']+(1-alpha)*cache['right_mean']
    assert bool(score.isfinite().all()) and bool(alpha.isfinite().all())
    return score,alpha,gate

def deploy(logits_primary,logits_partner,image,recipe):
    """One dynamic crop; no labels, case identifiers, or lookup tables accepted."""
    import numpy as np
    left=[torch.as_tensor(z) for z in logits_primary];right=[torch.as_tensor(z) for z in logits_partner]
    assert tuple(left[0].shape)==tuple(image.shape) and left[0].ndim==3
    score,alpha,gate=fuse(prepare(left,right),recipe)
    mask=(score>recipe['threshold_logit']).cpu().numpy()
    point=list(map(int,np.unravel_index(int(score.argmax()),score.shape)))
    return dict(mask=mask,automatic_point=point,score=score,alpha=alpha,conflict_gate=gate)

def cpu_gate():
    generator=torch.Generator().manual_seed(20261029)
    left=[torch.randn((7,9,11),generator=generator,dtype=torch.float64)+.6 for _ in range(8)]
    right=[1.7*left[i]+torch.randn((7,9,11),generator=generator,dtype=torch.float64)*.3-.9 for i in range(8)]
    cache=prepare(left,right);checks=[]
    for recipe in definitions():
        vr,vw,cov=statistics(cache,recipe['kernel'])
        assert bool((vr>=0).all()) and bool((vw>=0).all())
        assert bool((vr*vw-cov.square()>=-1e-12).all())
        assert bool((vr+vw-2*cov>=-1e-12).all())
        assert bool((recipe['ridge']*(vr+vw)+recipe['epsilon']>=recipe['epsilon']).all())
        score,alpha,gate=fuse(cache,recipe)
        assert bool(((alpha>=.5)&(alpha<=.95)).all()) and bool(((gate>=0)&(gate<=1)).all())
        assert torch.equal(score,fuse(cache,recipe)[0])
        same=prepare(left,left);assert torch.allclose(fuse(same,recipe)[0],same['left_mean'],atol=1e-12,rtol=0)
        empty=prepare([torch.zeros_like(left[0])]*8,[torch.zeros_like(left[0])]*8)
        zero,zero_alpha,_=fuse(empty,recipe);assert bool((zero==0).all()) and torch.allclose(zero_alpha,torch.full_like(zero_alpha,.75))
        singular=prepare(left,[z+.9 for z in left])
        # In probability space an exactly shared perturbation gives a rank-one covariance.
        singular['moments'][1]=singular['moments'][0];singular['moments'][2]=singular['moments'][0]
        singular['pooled']={1:singular['moments']}
        assert torch.equal(fuse(singular,recipe)[1],torch.full_like(zero_alpha,.75))
        exported=deploy(left,right,torch.ones_like(left[0]),recipe)
        assert torch.equal(exported['score'],score)
        checks.append(recipe['id'])
    # Verify the bounded closed-form optimum of the declared quadratic objective.
    recipe=definitions()[2];recipe=dict(recipe,conflict_gate=False)
    _,alpha,_=fuse(cache,recipe);vr,vw,cov=statistics(cache,1);lam=recipe['ridge']*(vr+vw)+recipe['epsilon']
    objective=lambda value:value.square()*vr+(1-value).square()*vw+2*value*(1-value)*cov+lam*(value-.75).square()
    optimum=objective(alpha)
    for alternative in torch.linspace(.5,.95,101,dtype=torch.float64):assert bool((optimum<=objective(alternative)+1e-12).all())
    # Each model mean is permutation-invariant; paired covariance is deliberately not.
    shuffled=prepare(left,list(reversed(right)))
    assert torch.allclose(cache['left_mean'],shuffled['left_mean']) and torch.allclose(cache['right_mean'],shuffled['right_mean'])
    delta=float((fuse(cache,recipe)[1]-fuse(shuffled,recipe)[1]).abs().max())
    assert delta>1e-4,'new method must use cross-model paired-view information'
    return dict(status='passed',configurations=len(checks),all12_finite_bounded_and_repeatable=True,
        identical_experts_fixed_point=True,zero_variance_anchor_fixed_point=True,
        quadratic_optimum_checked_against101_alphas=True,
        pairing_shuffle_max_alpha_change=delta,unpaired_means_unchanged=True,
        PSD_and_positive_ridge_checked=True,singular_covariance_falls_back_to075=True,
        standalone_dynamic_deploy_matches_cached=True,
        labels_or_case_ids_accepted_by_predictor=False,patient_arrays_loaded=False,model_forwards=0)
