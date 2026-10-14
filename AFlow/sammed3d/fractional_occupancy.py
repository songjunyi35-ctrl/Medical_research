"""Exact separable cell-overlap averaging of a piecewise-constant binary mask."""
import numpy as np


def cell_average_axis(array,output_size,axis):
    if not isinstance(output_size,(int,np.integer)) or output_size<1:
        raise ValueError('positive integer output size required')
    a=np.moveaxis(np.asarray(array,dtype=np.float64),axis,0)
    n=a.shape[0]
    if n<1:raise ValueError('empty input axis')
    edges=np.arange(output_size+1,dtype=np.float64)*(n/output_size)
    edges[-1]=n
    slots=np.floor(edges).astype(np.int64)
    fractions=edges-slots
    prefix=np.concatenate((np.zeros_like(a[:1]),np.cumsum(a,axis=0,dtype=np.float64)),axis=0)
    integral=prefix[slots]+a[np.minimum(slots,n-1)]*fractions.reshape((-1,)+(1,)*(a.ndim-1))
    averaged=np.diff(integral,axis=0)/(n/output_size)
    return np.moveaxis(averaged,0,axis)


def fractional_occupancy(mask,output_shape=(128,128,128)):
    mask=np.asarray(mask)
    if mask.ndim!=3 or min(mask.shape)<1 or len(output_shape)!=3:
        raise ValueError('nonempty 3D mask and three output sizes required')
    if not np.isfinite(mask).all() or not np.all((mask==0)|(mask==1)):
        raise ValueError('binary finite source mask required')
    result=mask.astype(np.float64)
    for axis,size in enumerate(output_shape):result=cell_average_axis(result,size,axis)
    if not np.isfinite(result).all() or result.min() < -1e-10 or result.max() > 1+1e-10:
        raise ArithmeticError('occupancy outside numerical tolerance')
    result=np.clip(result,0,1)
    expected=float(mask.sum(dtype=np.float64))*np.prod(output_shape)/np.prod(mask.shape)
    error=abs(float(result.sum(dtype=np.float64))-expected)/max(expected,1.)
    if error>1e-10:raise ArithmeticError('float64 volume conservation failed')
    return result
