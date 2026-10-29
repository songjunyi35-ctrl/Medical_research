"""GT-free score entry for a ready seed+normalized-seven cache."""
import time
import numpy as np
import torch
from scripts.race_v2_inference_normalization_a_20261029.algorithm import decode
from scripts.race_v2_inference_innovation_a_20261029.deployment import deploy as legacy_deploy

def deploy(logits,image,entry):
    if entry.get('mode')!='race8_seed_health_normalization':return legacy_deploy(logits,image,entry)
    assert len(logits)==8 and all(tuple(z.shape)==tuple(image.shape) for z in logits)
    value,point=decode(logits[0],logits[1:],entry['config'])
    seed_point=list(map(int,np.unravel_index(int(logits[0].argmax()),tuple(logits[0].shape))))
    return dict(logits=value,mask=value>0,automatic_point=point,base_automatic_point=seed_point,
        diagnostic=dict(input_kind='one_original_seed_plus_seven_health_normalized',GT_input=False,
            auxiliary_point_note='original identity seed argmax; ordinary equal8 unavailable unless separately cached'))

def gate_network(network,canvas,tail):
    from scripts.race_v2_inference_normalization_a_20261029.algorithm import transform,forward_view
    from scripts.race_v2_inference_a_20261029.predictor import forward_view as original_forward
    torch.manual_seed(197);torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    image=torch.randn(8,1,*canvas,device='cuda');support=image!=0
    g=dict(canvas_offset=[0,0,0],extent=list(canvas),reference_enabled=True,
        reflection=[[-1,0,0,canvas[0]-1],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    expected=original_forward(network,image,[g]*8,7,canvas)
    identity=forward_view(network,image,support,[g]*8,7,canvas)
    assert all(np.array_equal(a,b) for a,b in zip(expected,identity))
    changed=transform(image,support,dict(gain=1.075,offset=.075))
    full=forward_view(network,changed,support,[g]*8,7,canvas)
    partial=forward_view(network,changed[:tail],support[:tail],[g]*tail,1,canvas)
    torch.cuda.synchronize();peak=torch.cuda.max_memory_reserved()/2**30;assert peak<=38
    return dict(status='passed',canvas=list(canvas),batch=8,tail=tail,
        identity_original_forward_bit_exact=True,normalized_full_and_tail_finite=True,
        peak_reserved_gib=peak,seconds=time.monotonic()-started,patient_forwards=0)
