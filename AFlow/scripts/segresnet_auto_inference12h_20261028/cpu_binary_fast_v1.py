"""Exact separable box morphology; retain the same global179-configuration space."""
from scipy import ndimage as ndi
from scripts.segresnet_auto_inference12h_20261028 import cpu_binary_v1 as original

_REFERENCE_APPLY=original.apply


def fast_apply(selected,baseline,recipe,cache):
    op=recipe['operation']
    if op not in ['opening','closing','erosion','dilation','opening_closing','closing_opening']:
        return _REFERENCE_APPLY(selected,baseline,recipe,cache)
    source=recipe['source']
    if source not in cache:
        cache[source]=selected|baseline if source=='union' else selected&baseline
    value=cache[source];radius=recipe['radius'];connectivity=recipe['connectivity']
    def primitive(mask,kind):
        if connectivity==3:
            function=ndi.maximum_filter if kind=='dilation' else ndi.minimum_filter
            return function(mask,size=2*radius+1,mode='constant',cval=False)
        return getattr(ndi,'binary_'+kind)(mask,structure=ndi.generate_binary_structure(3,1),iterations=radius,border_value=0)
    for step in op.split('_'):
        if step=='opening':value=primitive(primitive(value,'erosion'),'dilation')
        elif step=='closing':value=primitive(primitive(value,'dilation'),'erosion')
        else:value=primitive(value,step)
    return value


# Spawned CPU children import this main module before resolving original.case_task.
original.apply=fast_apply


if __name__=='__main__':original.main()
