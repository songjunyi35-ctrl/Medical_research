"""Numerical tests for gain targets and exact deployment compatibility."""
import torch
from torch.nn import functional as F
from scripts.race_v2_20261029.common import *
from scripts.race_v2_20261029.model import RaceBenefitSeg
from scripts.race_v2_20261029.benefit import gain_calibration
from sammed3d.race_seg import RaceSeg

def main():
    torch.set_num_threads(2); torch.manual_seed(42)
    tests=[]
    base=RaceSeg(width=8,dropout=0,dim=4).eval()
    model=RaceBenefitSeg(width=8,dropout=0,dim=4).eval()
    model.load_state_dict(base.state_dict(),strict=True)
    with torch.no_grad(): base.strength.fill_(.35); model.strength.fill_(.35)
    image=torch.randn(1,1,32,32,32); valid=torch.ones_like(image)
    matrix=torch.eye(4)[None]; matrix[:,0,0]=-1; matrix[:,0,3]=24
    old=base(image,matrix,valid,corruption=True); new=model(image,matrix,valid,corruption=True)
    assert list(base.state_dict())==list(model.state_dict())
    for name in ('segmentation_logits','base_logits','gate','bad_logits','bad_gate'):
        assert torch.equal(getattr(old,name),getattr(new,name)),name
    tests.append('parameter/state_dict and normal/mismatch deployment outputs exact')
    new_invalid=model(image,matrix,valid*0,corruption=True)
    assert torch.equal(new_invalid.segmentation_logits,new_invalid.base_logits)
    assert not new_invalid.has_reference.any()
    tests.append('all-invalid reference finite exact base fallback')
    # Uniform positive labels; half proposal helps and half harms the base.
    target=torch.ones(1,1,8,8,8); z=torch.zeros_like(target,requires_grad=True)
    proposal=torch.cat([torch.ones(1,1,4,8,8),-torch.ones(1,1,4,8,8)],2).requires_grad_()
    gate=torch.full((1,1,2,2,2),-20.,requires_grad=True)
    available=torch.ones_like(gate,dtype=torch.bool)
    loss,counts=gain_calibration(gate,z,proposal,target,torch.ones_like(target),available)
    loss.backward()
    assert counts['gain_positive']==4 and counts['gain_negative']==4
    assert gate.grad[:,:,:1].mean()<0 and gate.grad[:,:,1:].mean()>0
    assert z.grad is None and proposal.grad is None
    tests.append('detached gain target; collapsed positive gates retain nonzero waking gradient')
    neutral_gate=torch.ones_like(gate,requires_grad=True)
    neutral,count=gain_calibration(neutral_gate,z,z,target,torch.ones_like(target),available)
    neutral.backward(); assert neutral==0 and torch.count_nonzero(neutral_gate.grad)==0
    tests.append('neutral proposals impose no gate-opening pressure')
    empty_gate=torch.ones_like(gate,requires_grad=True)
    empty,count=gain_calibration(empty_gate,z,proposal,target,torch.zeros_like(target),available*False)
    empty.backward(); assert empty==0 and bool(empty_gate.grad.isfinite().all())
    tests.append('empty support finite zero objective')
    write(R/'CPU_CHECKS.json',dict(at=now(),status='passed',tests=tests,patient_forwards=0,
          candidate_parameters_unchanged=True,new_source_sha256={str(Path(p).relative_to(ROOT)):sha(p) for p in (Path(__file__),Path(__file__).with_name('model.py'),Path(__file__).with_name('benefit.py'))}))
    print(json.dumps(dict(status='passed',tests=tests),ensure_ascii=False))

if __name__=='__main__': main()
