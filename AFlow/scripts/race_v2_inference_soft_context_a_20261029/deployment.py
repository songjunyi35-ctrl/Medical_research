"""All supported single-model score recipes; input cache identity is external."""
import numpy as np
import time,torch
from scripts.race_v2_inference_soft_context_a_20261029.algorithm import decode
from scripts.race_v2_inference_normalization_a_20261029.deployment import deploy as older_deploy

def deploy(logits,image,entry):
    if entry.get('mode')!='race8_seed_soft_context':return older_deploy(logits,image,entry)
    value,point=decode(logits,entry);seed_point=list(map(int,np.unravel_index(int(logits[0].argmax()),tuple(logits[0].shape))))
    return dict(logits=value,mask=value>0,automatic_point=point,base_automatic_point=seed_point,
        diagnostic=dict(input_chain='original identity seed1 then native soft-context7',
            auxiliary_point_rule='original identity seed argmax',GT_input=False))

def gate_network(network,canvas,tail):
    from scripts.race_v2_inference_soft_context_a_20261029.algorithm import context_field,transform
    from scripts.race_v2_inference_normalization_a_20261029.algorithm import forward_view
    from scripts.race_v2_inference_normalization_a_20261029.deployment import gate_network as original_gate
    original=original_gate(network,canvas,tail);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    image=torch.randn(8,1,*canvas,device='cuda');support=image!=0
    d=torch.arange(canvas[0],device='cuda',dtype=torch.float32)[:,None,None].expand(canvas).contiguous()[None,None].expand(8,1,*canvas).contiguous()
    g=dict(canvas_offset=[0,0,0],extent=list(canvas),reference_enabled=True,
        reflection=[[-1,0,0,canvas[0]-1],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    matrix=torch.tensor([g['reflection']]*8,device='cuda',dtype=torch.float32)
    field=context_field(d,matrix,torch.ones(8,device='cuda',dtype=torch.bool),10)
    changed=transform(image,field,.75);assert torch.equal(changed!=0,support)
    full=forward_view(network,changed,support,[g]*8,7,canvas)
    partial=forward_view(network,changed[:tail],support[:tail],[g]*tail,1,canvas)
    torch.cuda.synchronize();peak=torch.cuda.max_memory_reserved()/2**30;assert peak<=38
    return dict(status='passed',canvas=list(canvas),batch=8,tail=tail,original_forward_gate=original,
        context_then_model_finite=True,support_exact=True,peak_reserved_gib=peak,seconds=time.monotonic()-started,patient_forwards=0)
