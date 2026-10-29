"""Detached local reference gain targets with balanced beneficial/harmful cells.

Benefit compares an ungated proposal against the base prediction. Closing the
gate therefore cannot erase a positive target. Neutral proposals are ignored;
there is no gate floor, target utilization, or assumption normal means useful.
"""
import torch
from torch.nn import functional as F
from sammed3d.race_seg import typed_contrast

def local_average(value,size,weight):
    numerator=F.adaptive_avg_pool3d(value*weight,size)
    denominator=F.adaptive_avg_pool3d(weight,size)
    return numerator/denominator.clamp_min(1e-8),denominator

def gain_calibration(gate_logits,base_logits,proposal_logits,target,valid,available,margin=.002):
    # The complete target construction is detached, including the base and
    # candidate errors. It supervises the gate, not a differentiable oracle.
    with torch.no_grad():
        base=F.binary_cross_entropy_with_logits(base_logits.detach().float(),target.float(),reduction='none')
        proposal=F.binary_cross_entropy_with_logits(proposal_logits.detach().float(),target.float(),reduction='none')
        # Case-local foreground/background balance prevents large easy brain
        # regions overwhelming lesion and boundary cells; labels are train-only.
        fg=(target>.5).float()*valid
        bg=(target<=.5).float()*valid
        axes=(2,3,4)
        fg_count=fg.sum(axes,keepdim=True).clamp_min(1)
        bg_count=bg.sum(axes,keepdim=True).clamp_min(1)
        weight=(fg/fg_count+bg/bg_count)*valid.sum(axes,keepdim=True).clamp_min(1)/2
        gain,support=local_average(base-proposal,gate_logits.shape[-3:],weight)
        usable=(support>0)&available.bool()
        good=usable&(gain>margin); bad=usable&(gain < -margin)
        target_gate=(gain>0).float()
    lossmap=F.binary_cross_entropy_with_logits(gate_logits.float(),target_gate,reduction='none')
    per_case=[]
    for i in range(len(gate_logits)):
        terms=[lossmap[i][mask[i]].mean() for mask in (good,bad) if bool(mask[i].any())]
        per_case.append(torch.stack(terms).mean() if terms else gate_logits[i].sum()*0)
    loss=torch.stack(per_case).mean()
    return loss,dict(gain_positive=int(good.sum()),gain_negative=int(bad.sum()),
                     gain_neutral=int((usable&~good&~bad).sum()),
                     target_gain_mean=float(gain[usable].mean()) if bool(usable.any()) else 0.)

def objective(model,out,target,classes,valid,segmentation_loss,coefficient=.05,margin=.002):
    supervised=.5*segmentation_loss(out.segmentation_logits,target)+.5*segmentation_loss(out.base_logits,target)
    contrast=typed_contrast(out.embedding,classes,model.prototypes,valid)
    calibration,counts=gain_calibration(out.gate_logits,out.base_logits,out.proposal_logits,target,valid,out.has_reference,margin)
    if out.bad_gate_logits is not None:
        bad,bad_counts=gain_calibration(out.bad_gate_logits,out.base_logits,out.bad_proposal_logits,target,valid,out.has_reference,margin)
        calibration=(calibration+bad)/2
        counts.update({'mismatch_'+k:v for k,v in bad_counts.items()})
    loss=supervised+.05*contrast+coefficient*calibration
    return loss,dict(supervised=float(supervised.detach()),contrast=float(contrast.detach()),
                     calibration=float(calibration.detach()),gate_mean=float(out.gate.detach().mean()),**counts)
