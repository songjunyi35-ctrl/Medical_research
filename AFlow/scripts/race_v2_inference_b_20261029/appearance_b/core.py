"""Same-image appearance unary heuristic; deterministic histograms, no GT/IDs."""
import math
import numpy as np
import torch
from scripts.race_v2_inference_b_20261029.innovation_b.core import prepare as covariance_prepare,fuse as covariance_fuse

def definitions(base_recipe):
    return [dict(id=f'appearance_k{components}_lambda{strength:g}',kind='image_appearance_unary',components=components,strength=strength,
        base_covariance_recipe=base_recipe,foreground_probability_minimum=.95,background_probability_maximum=.01,
        foreground_seed_minimum=64,background_seed_minimum=1024,histogram_bins=256,histogram_domain='full_image_nonzero_min_max',
        sampling='all_seed_voxels_no_subsampling',maximum_crop_voxels=160*208*168,initialization='equal_weights_weighted_hist_midquantiles_global_variance',
        EM_iterations=5,variance_floor=.01,mixture_weight_floor=1e-6,range_floor=1e-6,log_likelihood_ratio_clip=6.,
        trust='one_minus_Bhattacharyya_coefficient_of_normalized_densities_on256bin_centers',
        threshold_logit=0.,models=2,forwards_per_scan=16,new_neural_forwards=0)
        for components in (1,3) for strength in (.25,.5)]

def log_density(values,weights,means,variance):
    logits=weights.log()[:,None]-.5*((values[None]-means[:,None]).square()/variance[:,None]+variance.log()[:,None]+math.log(2*math.pi))
    return torch.logsumexp(logits,dim=0)

def fit_histogram(centers,histogram,components):
    mass=histogram.sum();assert float(mass)>0
    probability=histogram/mass;cdf=probability.cumsum(0)
    quantiles=(torch.arange(components,device=centers.device,dtype=centers.dtype)+.5)/components
    means=centers[torch.searchsorted(cdf.contiguous(),quantiles).clamp_max(len(centers)-1)]
    mu=(centers*probability).sum();variance=((centers-mu).square()*probability).sum().clamp_min(.01).expand(components).clone()
    weights=torch.full_like(means,1/components)
    for _ in range(5):
        terms=weights.log()[:,None]-.5*((centers[None]-means[:,None]).square()/variance[:,None]+variance.log()[:,None]+math.log(2*math.pi))
        responsibility=terms.softmax(0)*histogram[None];counts=responsibility.sum(1).clamp_min(1e-12)
        updated_means=(responsibility*centers[None]).sum(1)/counts
        updated_variance=(responsibility*(centers[None]-updated_means[:,None]).square()).sum(1)/counts
        weights=(counts/mass).clamp_min(1e-6);weights=weights/weights.sum();means=updated_means;variance=updated_variance.clamp_min(.01)
    assert bool(weights.isfinite().all()) and bool(means.isfinite().all()) and bool(variance.isfinite().all())
    assert bool((weights>0).all()) and bool((variance>=.01).all())
    return weights,means,variance

@torch.inference_mode()
def appearance_ratio(image,foreground,background,components):
    support=image!=0;zero=torch.zeros_like(image);fg=int(foreground.sum());bg=int(background.sum())
    if fg<64 or bg<1024:return zero,0.,dict(foreground_seeds=fg,background_seeds=bg,fallback='insufficient_seeds')
    values=image[support];low=float(values.min());high=float(values.max())
    if high-low<1e-6:return zero,0.,dict(foreground_seeds=fg,background_seeds=bg,fallback='constant_image')
    assert image.numel()<=160*208*168 and image.numel()<2**24
    centers=torch.linspace(low+(high-low)/512,high-(high-low)/512,256,device=image.device,dtype=image.dtype)
    fh=torch.histc(image[foreground],bins=256,min=low,max=high);bh=torch.histc(image[background],bins=256,min=low,max=high)
    assert int(fh.sum())==fg and int(bh.sum())==bg
    fm=fit_histogram(centers,fh,components);bm=fit_histogram(centers,bh,components)
    f=log_density(centers,*fm).softmax(0);b=log_density(centers,*bm).softmax(0)
    coefficient=(f*b).sqrt().sum().clamp(0,1);trust=float((1-coefficient).clamp(0,1))
    if torch.equal(fh/fg,bh/bg):trust=0.
    ratio=torch.zeros_like(image)
    ratio[support]=(log_density(values,*fm)-log_density(values,*bm)).clamp(-6,6)
    assert bool(ratio.isfinite().all()) and bool((ratio[~support]==0).all())
    return ratio,trust,dict(foreground_seeds=fg,background_seeds=bg,fallback=None,trust=trust,
        foreground_weights=fm[0].cpu().tolist(),foreground_means=fm[1].cpu().tolist(),foreground_variances=fm[2].cpu().tolist(),
        background_weights=bm[0].cpu().tolist(),background_means=bm[1].cpu().tolist(),background_variances=bm[2].cpu().tolist(),
        histogram_range=[low,high],Bhattacharyya_coefficient=float(coefficient))

@torch.inference_mode()
def prepare(left,right,image,base_recipe):
    assert image.ndim==3 and bool(image.isfinite().all())
    assert all(z.shape==image.shape for z in [*left,*right])
    cache=covariance_prepare(left,right);base,alpha,gate=covariance_fuse(cache,base_recipe)
    p=cache['left_mean'].sigmoid();q=cache['right_mean'].sigmoid();support=image!=0
    return dict(base_score=base,left_mean=cache['left_mean'],right_mean=cache['right_mean'],image=image,foreground=support&(p>.95)&(q>.95),background=support&(p<.01)&(q<.01),ratios={})

@torch.inference_mode()
def fuse(cache,recipe):
    components=recipe['components']
    if components not in cache['ratios']:cache['ratios'][components]=appearance_ratio(cache['image'],cache['foreground'],cache['background'],components)
    ratio,trust,diagnostics=cache['ratios'][components]
    shift=recipe['strength']*trust*ratio;score=cache['base_score']+shift
    assert bool(score.isfinite().all()) and float(shift.abs().max())<=recipe['strength']*6+1e-6
    return score,shift,trust,diagnostics

@torch.inference_mode()
def deploy(logits_primary,logits_partner,image,recipe):
    left=[torch.as_tensor(z) for z in logits_primary];right=[torch.as_tensor(z) for z in logits_partner];image=torch.as_tensor(image,device=left[0].device)
    score,shift,trust,diagnostics=fuse(prepare(left,right,image,recipe['base_covariance_recipe']),recipe)
    point=list(map(int,np.unravel_index(int(score.argmax()),tuple(score.shape))))
    return dict(mask=(score>0).cpu().numpy(),automatic_point=point,score=score,appearance_shift=shift,trust=trust,diagnostics=diagnostics)

def cpu_gate(base_recipe):
    shape=(24,31,33);image=torch.ones(shape);image[:5]=2.;fg=torch.zeros(shape,dtype=torch.bool);fg[:5]=True;bg=~fg
    for components in (1,3):
        ratio,trust,info=appearance_ratio(image,fg,bg,components);other,other_trust,_=appearance_ratio(image,bg,fg,components)
        assert trust==other_trust and torch.allclose(ratio,-other,atol=1e-6)
        same,same_trust,_=appearance_ratio(image,torch.ones_like(fg),torch.ones_like(fg),components)
        assert same_trust==0.
        constant=appearance_ratio(torch.ones(shape),fg,bg,components);assert constant[1]==0. and bool((constant[0]==0).all())
        tiny=torch.zeros_like(fg);tiny[0,0,:3]=True;small=appearance_ratio(image,tiny,bg,components);assert small[1]==0.
        for axis in range(3):
            mirrored=appearance_ratio(image.flip(axis),fg.flip(axis),bg.flip(axis),components)
            assert mirrored[1]==trust and torch.allclose(mirrored[0].flip(axis),ratio,atol=1e-6)
        order=torch.arange(image.numel()-1,-1,-1);permuted=appearance_ratio(image.flatten()[order].reshape(shape),fg.flatten()[order].reshape(shape),bg.flatten()[order].reshape(shape),components)
        assert permuted[1]==trust and torch.allclose(permuted[0].flatten()[order].reshape(shape),ratio,atol=1e-6)
    l=torch.where(fg,torch.tensor(5.),torch.tensor(-6.));left=[l.clone().requires_grad_(True) for _ in range(8)];right=[l.clone() for _ in range(8)]
    cache=prepare(left,right,image,base_recipe)
    for recipe in definitions(base_recipe):
        score,shift,trust,info=fuse(cache,recipe);zero=dict(recipe,strength=0.)
        assert torch.equal(fuse(cache,zero)[0],cache['base_score']) and not score.requires_grad
        output=deploy(left,right,image,recipe);assert torch.equal(output['score'],score)
        assert output['automatic_point']==list(np.unravel_index(int(score.argmax()),shape))
    return dict(status='passed',identical_class_distributions_zero_trust=True,constant_and_small_seed_fallback=True,
        finite_density_and_positive_variance_mixture=True,foreground_background_exchange_sign=True,
        all_axis_mirror_and_voxel_order_invariant=True,lambda_zero_identity=True,autograd_disabled=True,
        final_score_point=True,patient_arrays_loaded=False,model_forwards=0)
