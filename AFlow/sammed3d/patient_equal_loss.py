"""Patient-equal coefficients, full scan coverage; ordinary unweighted validation."""
from collections import Counter
import torch
from torch import nn
from torch.nn import functional as F
from monai.losses import DiceLoss


def patient_weights(rows):
    if len({r['case_id'] for r in rows})!=len(rows):raise ValueError('duplicate scan')
    counts=Counter(r['subject_id'] for r in rows)
    if not counts:raise ValueError('empty cohort')
    return {r['case_id']:len(rows)/(len(counts)*counts[r['subject_id']]) for r in rows}


class PatientEqualDiceCE(nn.Module):
    def __init__(self):
        super().__init__();self.dice=DiceLoss(sigmoid=True,squared_pred=False,batch=False,reduction='none')
    def forward(self,logits,target,weights):
        if weights.shape!=(logits.shape[0],) or not bool(torch.isfinite(weights).all() and (weights>0).all()):
            raise ValueError('one positive finite coefficient per scan required')
        d=self.dice(logits,target).flatten(1).mean(1)
        ce=F.binary_cross_entropy_with_logits(logits,target,reduction='none').flatten(1).mean(1)
        return ((d+.1*ce)*weights.to(d.dtype)).mean()
