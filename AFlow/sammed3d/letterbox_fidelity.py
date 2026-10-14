"""Label-only aspect-preserving grid arithmetic, independently testable."""
import numpy as np
from sammed3d.spatial_roundtrip_audit import resize


def layout(extent,size=128):
    if len(extent)!=3 or min(extent)<1 or size<1:raise ValueError('positive three axes required')
    shape=tuple(max(1,int(np.floor(size*e/max(extent)+.5))) for e in extent)
    before=tuple((size-e)//2 for e in shape)
    return shape,tuple(slice(o,o+e) for o,e in zip(before,shape))


def encode(array,size=128):
    shape,slices=layout(array.shape,size)
    small=resize(array,shape,'nearest-exact')
    box=np.zeros((size,size,size),dtype=small.dtype);box[slices]=small
    return box


def decode(box,extent):
    if box.ndim!=3 or len(set(box.shape))!=1:raise ValueError('cube required')
    shape,slices=layout(extent,box.shape[0])
    assert box[slices].shape==shape
    return resize(box[slices],extent,'nearest-exact')
