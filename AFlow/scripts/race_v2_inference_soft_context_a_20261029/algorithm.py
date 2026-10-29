"""Full-native-grid input gain; only the scalar field is reflected/interpolated."""
import itertools,math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt,affine_transform
from sammed3d.race_seg import voxel_grid

def definitions():
    return [dict(id=f'race8_softcontext_r{r}_floor{f}',radius_mm=r,floor=f,
        seed_probability_threshold=.1,reference_symmetric=True,seed_weight=.125,
        views=list(range(1,8)),output_logit_threshold=0.,point_rule='actual_final_score_argmax',
        empty_seed='identity_reuse',reference_disabled='unreflected_field_only',
        image_resampling=False,input_renormalization=False,original_support_retained=True,
        neural_case_forwards=8) for r,f in itertools.product([10,20],[.75,.9])]

def distance(seed_logits,original_support):
    assert seed_logits.shape==original_support.shape
    suspected=(seed_logits>=math.log(.1/.9))&original_support
    if not suspected.any():return np.zeros(seed_logits.shape,np.float32),True
    return distance_transform_edt(~suspected).astype(np.float32),False

def prepare_grid(reflection,canvas):
    points=voxel_grid(canvas,reflection.device)
    mapped=torch.einsum('bij,dhwj->bdhwi',reflection[:,:3,:3].float(),points)
    mapped=mapped+reflection[:,None,None,None,:3,3].float()
    limits=torch.tensor(canvas,device=reflection.device,dtype=torch.float32)-1
    return (2*mapped/limits-1).flip(-1)

def context_field(distances,reflection,reference_enabled,radius_mm,grid=None):
    """Returns unit-amplitude max(exp-distance, reflected exp-distance)."""
    assert distances.ndim==5 and distances.shape[1]==1
    if grid is None:grid=prepare_grid(reflection,distances.shape[-3:])
    phi=torch.exp(-.5*(distances/radius_mm).square())
    reflected=F.grid_sample(phi,grid,align_corners=True,padding_mode='zeros',mode='bilinear')
    symmetric=torch.maximum(phi,reflected)
    return torch.where(reference_enabled[:,None,None,None,None],symmetric,phi)

def transform(image,field,floor):
    assert image.shape==field.shape and 0<floor<=1
    return image*(floor+(1-floor)*field)

def decode(logits,entry):
    if entry.get('mode')=='equal':
        out=torch.zeros_like(logits[0])
        for z in logits:out.add_(z,alpha=.125)
    else:
        recipe=entry.get('config',entry);assert recipe['seed_weight']==.125
        out=logits[0]*.125
        for z in logits[1:]:out=out+z*.125
    point=list(map(int,np.unravel_index(int(out.argmax()),tuple(out.shape))))
    return out,point

def numerical_gate(device='cpu'):
    shape=(19,21,23);rng=np.random.default_rng(1029);image=rng.normal(size=shape).astype(np.float32);image[:2]=0
    seed=np.full(shape,-10.,np.float32);seed[5:8,9:12,10:13]=4.
    d,empty=distance(seed,image!=0);assert not empty
    reflection=np.eye(4,dtype=np.float32);reflection[0,0]=-1;reflection[0,3]=shape[0]-1
    matrix=torch.from_numpy(reflection)[None].to(device);dt=torch.from_numpy(d)[None,None].to(device)
    checks=[];maximum=0.
    for r in (10,20):
        phi=np.exp(-.5*(d.astype(np.float64)/r)**2).astype(np.float32)
        expected=np.maximum(phi,affine_transform(phi,reflection[:3,:3],reflection[:3,3],order=1,mode='constant',cval=0,prefilter=False))
        actual=context_field(dt,matrix,torch.tensor([True],device=device),r)[0,0].cpu().numpy()
        error=float(np.max(np.abs(expected-actual)));maximum=max(maximum,error);assert error<3e-5
        assert np.all(actual[seed>=math.log(.1/.9)]==1)
        for floor in (.75,.9):
            value=transform(torch.from_numpy(image)[None,None].to(device),torch.from_numpy(actual)[None,None].to(device),floor).cpu().numpy()[0,0]
            assert np.array_equal(value!=0,image!=0) and np.isfinite(value).all()
            checks.append(dict(radius_mm=r,floor=floor,original_support_exact=True,field_reference_error=error))
    zero,empty=distance(np.full(shape,-99,np.float32),image!=0);assert empty and not zero.any()
    identity=context_field(torch.zeros_like(dt),matrix,torch.tensor([True],device=device),10)
    assert bool((identity==1).all())
    return dict(status='passed',checks=checks,max_field_CPU_reference_error=maximum,
        empty_seed_identity=True,no_patient_arrays=True,no_neural_forwards=True,
        interpolated_field_only=True,image_grid_and_reference_matrix_unchanged=True)
