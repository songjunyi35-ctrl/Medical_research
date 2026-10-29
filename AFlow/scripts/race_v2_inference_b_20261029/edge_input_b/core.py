"""Proposed self-guided image filtering. No model, GT, IDs, or search input."""
import torch
from scripts.race_v2_inference_b_20261029.image_conflict_b.core import box

def transform(image,beta):
    assert image.ndim==3 and bool(image.isfinite().all()) and beta in (0.,.25,.5)
    support=image!=0;weight=support.to(image.dtype);mass=box(weight,5).clamp_min(1e-8)
    mu=box(image*weight,5)/mass
    variance=(box(image.square()*weight,5)/mass-mu.square()).clamp_min(0)
    a=variance/(variance+.04);b=(1-a)*mu
    q=(box(a*weight,5)/mass)*image+box(b*weight,5)/mass
    result=((1-beta)*image+beta*q)*weight
    assert bool(result.isfinite().all())
    return result

def cpu_gate():
    image=torch.ones((17,19,21));image[:3]=0;image[-3:]=0
    for beta in (.25,.5):
        assert torch.allclose(transform(image,beta),image,atol=1e-6)
        assert torch.equal(transform(image,0.),image)
    image[8:]=torch.where(image[8:]!=0,torch.tensor(3.),torch.tensor(0.))
    guided=transform(image,.5);weight=(image!=0).float();uniform=box(image,5)/box(weight,5).clamp_min(1e-8)
    edge=(image!=0)&(torch.arange(17)[:,None,None]>=6)&(torch.arange(17)[:,None,None]<=9)
    assert float((guided-image)[edge].abs().mean())<float((uniform-image)[edge].abs().mean())
    for beta in (.25,.5):
        value=transform(image,beta);assert bool((value[image==0]==0).all())
        assert float(value[image!=0].min())>=.999 and float(value.max())<=3.001
        for axis in range(3):assert torch.allclose(transform(image.flip(axis),beta).flip(axis),value,atol=2e-6)
    return dict(status='passed',constant_supported_image_fixed_point=True,zero_beta_identity=True,
        zero_support_preserved=True,step_edge_less_smoothed_than_uniform_box=True,all_axis_mirror_commutation=True,
        output_finite_and_synthetic_range_preserved=True,patient_arrays_loaded=False,GPU_operations=0,model_forwards=0)
