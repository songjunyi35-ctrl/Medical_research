"""RACE same-model8-view local dependency fusion; GT-free inference only.

The error proxy is disagreement with the other seven views' majority. It is not
an observed error probability. The inverse-moment rule is a JLF-inspired
adaptation, with nonnegative renormalization and conservative bounded updates.
"""
import itertools
import torch
import torch.nn.functional as F

def definitions():
    recipes=[]
    for mode,kernel,ridge,blend in itertools.product(['joint','diagonal'],[9,17],[.05,.2],[.5,1.]):
        recipes.append(dict(id=f'race8_local_{mode}_k{kernel}_ridge{ridge}_blend{blend}',
            kind='race_same_model8_local_dependency',dependency=mode,box_mm=kernel,lattice_stride=4,
            ridge=ridge,blend=blend,residual_clip=2.,uncertain_logit_band=2.,vote_logit_threshold=0.,
            output_logit_threshold=0.,point_rule='final_refined_logit_argmax',base_point_auxiliary=True,
            neural_case_forwards=8))
    assert len(recipes)==16
    return recipes

def equal8(logits):
    value=torch.zeros_like(logits[0])
    for z in logits:value.add_(z,alpha=.125)
    return value

def errors_from_other7(logits):
    votes=torch.stack([z>0 for z in logits]);count=votes.sum(0)
    others=(count[None]-votes.to(torch.int64))>=4
    return (votes!=others).float(),count

def local_moments(errors,kernel,stride=4):
    assert len(errors)==8 and errors.ndim==4 and kernel%2==1
    value=(errors[:,None]*errors[None,:]).reshape(1,64,*errors.shape[1:])
    # Separable box averages, with exactly the product of valid-axis counts.
    for axis in range(3):
        k=[1,1,1];p=[0,0,0];k[axis]=kernel;p[axis]=kernel//2
        value=F.avg_pool3d(value,kernel_size=k,stride=1,padding=p,count_include_pad=False)
    return value[0,:,::stride,::stride,::stride].reshape(8,8,*value[0,:,::stride,::stride,::stride].shape[-3:]).contiguous()

def coarse_weights(moments,dependency,ridge):
    shape=moments.shape[-3:];matrix=moments.permute(2,3,4,0,1).reshape(-1,8,8)
    matrix=(matrix+matrix.transpose(-1,-2))*.5
    if dependency=='diagonal':matrix=torch.diag_embed(matrix.diagonal(dim1=-2,dim2=-1))
    elif dependency!='joint':raise ValueError(dependency)
    matrix=matrix+ridge*torch.eye(8,device=matrix.device,dtype=matrix.dtype)
    raw=torch.linalg.solve(matrix,torch.ones((*matrix.shape[:-1],1),device=matrix.device,dtype=matrix.dtype)).squeeze(-1)
    value=raw.clamp_min(0);mass=value.sum(-1,keepdim=True)
    value=torch.where(mass>1e-10,value/mass.clamp_min(1e-10),torch.full_like(value,.125))
    assert bool(value.isfinite().all()) and bool((value>=0).all())
    return value.reshape(*shape,8).permute(3,0,1,2).contiguous()

def expand_weights(weights,shape,stride=4):
    axes=[]
    for n,coarse in zip(shape,weights.shape[-3:]):
        axes.append(torch.arange(n,device=weights.device,dtype=weights.dtype)/stride/max(coarse-1,1)*2-1 if coarse>1
                    else torch.zeros(n,device=weights.device,dtype=weights.dtype))
    d,h,w=torch.meshgrid(*axes,indexing='ij');grid=torch.stack((w,h,d),-1)[None]
    result=F.grid_sample(weights[None],grid,mode='bilinear',padding_mode='border',align_corners=True)[0]
    return result/result.sum(0,keepdim=True).clamp_min(1e-10)

def prepare(logits):
    assert len(logits)==8 and all(z.shape==logits[0].shape and z.dtype==torch.float32 and bool(z.isfinite().all()) for z in logits)
    base=equal8(logits);errors,count=errors_from_other7(logits)
    return dict(logits=logits,base=base,errors=errors,count=count,moments={},weighted={},statistics={})

def refine(cache,recipe):
    kernel=recipe['box_mm'];stride=recipe['lattice_stride'];key=(recipe['dependency'],kernel,recipe['ridge'])
    if kernel not in cache['moments']:cache['moments'][kernel]=local_moments(cache['errors'],kernel,stride)
    if key not in cache['weighted']:
        coarse=coarse_weights(cache['moments'][kernel],recipe['dependency'],recipe['ridge'])
        weights=expand_weights(coarse,cache['base'].shape,stride);weighted=torch.zeros_like(cache['base'])
        for i,z in enumerate(cache['logits']):weighted.add_(weights[i]*z)
        cache['weighted'][key]=weighted
        cache['statistics'][key]=dict(weight_min=float(weights.min()),weight_max=float(weights.max()),
            maximum_weight_sum_error=float((weights.sum(0)-1).abs().max()),coarse_shape=list(coarse.shape[-3:]))
    base=cache['base'];eligible=(cache['count']>0)&(cache['count']<8)&(base.abs()<=recipe['uncertain_logit_band'])
    correction=(cache['weighted'][key]-base).clamp(-recipe['residual_clip'],recipe['residual_clip'])
    result=torch.where(eligible,base+recipe['blend']*correction,base)
    assert bool(result.isfinite().all())
    return result,dict(cache['statistics'][key],eligible_voxels=int(eligible.sum()),
        changed_mask_voxels=int(((result>0)!=(base>0)).sum()),maximum_logit_correction=float((result-base).abs().max()))

@torch.inference_mode()
def deploy(logits,image,recipe):
    """Accept only aligned cropped logits and input image; no GT or case ID."""
    prototype=logits[0]
    tensors=[torch.as_tensor(z,dtype=torch.float32,device=prototype.device if torch.is_tensor(prototype) else 'cpu') for z in logits]
    assert tuple(image.shape)==tuple(tensors[0].shape)
    cache=prepare(tensors);result,diagnostic=refine(cache,recipe)
    point=list(map(int,torch.unravel_index(result.argmax(),result.shape)))
    base_point=list(map(int,torch.unravel_index(cache['base'].argmax(),cache['base'].shape)))
    return dict(mask=result>0,logits=result,automatic_point=point,base_automatic_point=base_point,diagnostic=diagnostic)

def numerical_gate(device='cpu'):
    generator=torch.Generator(device=device).manual_seed(31)
    logits=[torch.randn((19,21,23),generator=generator,device=device) for _ in range(8)]
    cache=prepare(logits);errors=cache['errors'];k=9
    direct=F.avg_pool3d((errors[:,None]*errors[None,:]).reshape(1,64,*errors.shape[1:]),k,stride=4,padding=k//2,count_include_pad=False)[0].reshape(8,8,5,6,6)
    separated=local_moments(errors,k);assert torch.allclose(direct,separated,atol=8e-7,rtol=0)
    recipe=definitions()[0];result,stats=refine(cache,recipe)
    assert stats['maximum_weight_sum_error']<1e-5
    zero=dict(recipe,blend=0.);unchanged,_=refine(cache,zero);assert torch.equal(unchanged,cache['base'])
    unanimous=prepare([logits[0]]*8);same,_=refine(unanimous,recipe);assert torch.equal(same,unanimous['base'])
    perm=[7,1,5,3,2,4,6,0];permuted=prepare([logits[i] for i in perm]);other,_=refine(permuted,recipe)
    assert torch.allclose(result,other,atol=3e-5,rtol=0)
    assert torch.equal(result[(cache['count']==0)|(cache['count']==8)],cache['base'][(cache['count']==0)|(cache['count']==8)])
    return dict(status='passed',device=device,separable_vs_full_box_max_error=float((direct-separated).abs().max()),
        alpha0_exact=True,unanimous_exact=True,permutation_max_error=float((result-other).abs().max()),
        weights_nonnegative_sum1=True,point_rule='refined argmax; base point separately reported',GT_input=False)
