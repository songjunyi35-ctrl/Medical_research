"""Full3D Gaussian/bilateral DenseCRF, isolated optional C++ implementation."""
import sys,time
from pathlib import Path
import numpy as np


def library(vendor):
    sys.path.insert(0,str(vendor))
    import pydensecrf.densecrf as dcrf
    return dcrf


def infer(probability,image,spatial_sigma,color_sigma,weight,iterations=(5,10),vendor=None,gaussian_weight=1.):
    assert probability.ndim==image.ndim==3 and probability.shape==image.shape
    dcrf=library(vendor);shape=probability.shape;n=probability.size
    p=np.clip(probability.astype(np.float32),1e-6,1-1e-6)
    unary=-np.log(np.stack([1-p,p]).reshape(2,n)).astype(np.float32)
    model=dcrf.DenseCRF(n,2);model.setUnaryEnergy(np.ascontiguousarray(unary))
    coords=np.indices(shape,dtype=np.float32).reshape(3,n)
    if gaussian_weight:
        model.addPairwiseEnergy(np.ascontiguousarray(coords/1.5),compat=gaussian_weight,kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)
    if weight:
        features=np.concatenate([coords/spatial_sigma,image.reshape(1,n).astype(np.float32)/color_sigma],axis=0)
        model.addPairwiseEnergy(np.ascontiguousarray(features),compat=weight,kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)
    q,t1,t2=model.startInference();outputs={}
    for iteration in range(1,max(iterations)+1):
        model.stepInference(q,t1,t2)
        if iteration in iterations:outputs[iteration]=np.array(q,dtype=np.float32,order='C',copy=True)[1].reshape(shape)
    return outputs


def gate(vendor):
    rng=np.random.default_rng(20261029);p=rng.uniform(.05,.95,(9,11,13)).astype(np.float32);image=rng.normal(size=p.shape).astype(np.float32)
    identity=infer(p,image,5.,.5,0.,iterations=(1,),vendor=vendor,gaussian_weight=0.)[1]
    assert np.allclose(identity,p,atol=2e-7,rtol=0)
    a=infer(p,image,5.,.5,1.,iterations=(3,),vendor=vendor)[3];b=infer(p,image,5.,.5,1.,iterations=(3,),vendor=vendor)[3]
    assert np.array_equal(a,b) and np.isfinite(a).all() and ((a>=0)&(a<=1)).all()
    return dict(status='passed',full3D_shape=p.shape,no_pairwise_identity_max_abs=float(np.abs(identity-p).max()),repeat_exact=True,GT_input=False)
