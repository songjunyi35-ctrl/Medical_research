"""Prediction-only fixed component cleanup; highest-logit location is invariant."""
import numpy as np
from scipy import ndimage


def cleanup_logits(logits):
    logits=np.asarray(logits)
    if logits.ndim!=3 or not np.issubdtype(logits.dtype,np.floating):
        raise ValueError('one floating-point D,H,W logit array required')
    if not np.isfinite(logits).all():raise ValueError('nonfinite logits')
    point=int(logits.argmax());mask=logits>0
    labels,n=ndimage.label(mask,structure=np.ones((3,3,3),dtype=bool))
    result=logits.copy()
    if n==0:
        return result,{'components_before':0,'components_after':0,'largest_component':0,'cutoff_voxels':0,'protected_component':0,'removed_voxels':0,'point_flat_index':point}
    counts=np.bincount(labels.ravel(),minlength=n+1)
    largest=int(counts[1:].max())
    cutoff=min(largest,max(8,(largest+99)//100))
    protected=int(labels.ravel()[point]);assert protected>0
    keep=counts>=cutoff;keep[0]=False;keep[protected]=True
    removed=mask & ~keep[labels]
    result[removed]=-1.
    assert int(result.argmax())==point and np.all((result>0)<=mask)
    assert (result>0).any()
    return result,{'components_before':int(n),'components_after':int(keep[1:].sum()),'largest_component':largest,'cutoff_voxels':cutoff,'protected_component':protected,'removed_voxels':int(removed.sum()),'point_flat_index':point}
