"""Parameter-identical RACE with observable gates and ungated proposals.

The deployed segmentation formula is unchanged. Proposal outputs exist only for
training/diagnostic benefit targets and never consume labels in forward().
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from sammed3d.race_seg import RaceSeg

@dataclass
class BenefitOutput:
    segmentation_logits: torch.Tensor
    base_logits: torch.Tensor
    embedding: torch.Tensor
    gate: torch.Tensor
    has_reference: torch.Tensor
    gate_logits: torch.Tensor
    proposal_logits: torch.Tensor
    correction_low: torch.Tensor
    bad_logits: torch.Tensor | None = None
    bad_gate: torch.Tensor | None = None
    bad_gate_logits: torch.Tensor | None = None
    bad_proposal_logits: torch.Tensor | None = None

class RaceBenefitSeg(RaceSeg):
    def reference_details(self,features,matrix,valid,base,corrupt=False):
        query,reference,entropy,available=self.reference(features,matrix,valid,corrupt=corrupt)
        delta=query-reference
        evidence=self.evidence(torch.cat([query,reference,delta,delta.abs()],1))
        native_gate_logits=self.gate_head(torch.cat([evidence,entropy],1))
        gate_logits=native_gate_logits.float()
        gate=torch.sigmoid(native_gate_logits)*available
        raw_native=torch.tanh(self.correction(evidence))
        raw=raw_native.float()*available
        scale=2*torch.tanh(self.strength)
        # Keep the historical bf16 multiplication before fp32 interpolation.
        correction=F.interpolate((gate*raw_native).float(),size=base.shape[-3:],mode='trilinear',align_corners=True)
        proposal=F.interpolate(raw,size=base.shape[-3:],mode='trilinear',align_corners=True)
        return (base.float()+scale*correction,F.normalize(self.project(evidence).float(),dim=1),
                gate,available,gate_logits,base.float()+scale*proposal,raw)
    def forward(self,image,reflection=None,valid=None,*,corruption=False):
        if reflection is None or valid is None: raise ValueError('RACE requires geometry/support')
        network=self.backbone.network
        features,skip=network.encode(image)
        base=network.decode(features,list(reversed(skip)))
        good=self.reference_details(features,reflection,valid,base)
        bad=(self.reference_details(features,reflection,valid,base,True) if corruption else None)
        return BenefitOutput(good[0],base,*good[1:],
                             *((bad[i] for i in (0,2,4,5)) if bad else (None,None,None,None)))
