"""Project-original labels, not nnU-Net's differently recoded example labels."""
import numpy as np
from sammed3d.spatial_roundtrip_audit import resize


def regions(classes):
    a=np.asarray(classes)
    if not np.isin(a,[0,1,2,3,4]).all():raise ValueError('project original labels 0..4 required')
    return {'whole_tumor':np.isin(a,[1,2,3]),'tumor_core':np.isin(a,[1,3]),'enhancing_tissue':a==3,'resection_cavity':a==4}


def torch_sample(raw,origin,extent,size=128):
    box=tuple(slice(o,o+e) for o,e in zip(origin,extent))
    return resize(raw[box],(size,size,size),'nearest-exact').astype(np.uint8)


def numpy_sample(raw,origin,extent,size=128):
    # Independent direct indexing; no Torch interpolation or shared index helper.
    axes=[o+np.floor((np.arange(size,dtype=np.float64)+.5)*e/size).astype(np.int64) for o,e in zip(origin,extent)]
    return np.asarray(raw[np.ix_(*axes)],dtype=np.uint8)
