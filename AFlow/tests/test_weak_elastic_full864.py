import copy
import torch
from monai.losses import DiceCELoss
from sammed3d.structured_background_aux import StructuredBackgroundLocator,conditional_rc_loss,export_plain_state
from scripts import weak_elastic_full864_protocol as p
from scripts.train_weak_elastic_full864 import full_weights


def test_full_manifest_is_exact_authorized_union():
    full,dev=p.check_roles()
    assert len(full)==864 and dev==[]
    assert len({r['subject_id'] for r in full})==391


def test_zero_coefficient_auxiliary_target_does_not_change_any_gradient():
    torch.manual_seed(13)
    a=StructuredBackgroundLocator();b=copy.deepcopy(a)
    x=torch.randn(2,1,16,16,16)
    wt=torch.zeros_like(x);wt[:,:,3:8,3:8,3:8]=1
    rc=torch.zeros_like(x);rc[:,:,10:13,10:13,10:13]=1
    loss_fn=DiceCELoss(sigmoid=True,squared_pred=False,lambda_dice=1,lambda_ce=.1)
    for model,target in ((a,rc),(b,torch.zeros_like(rc))):
        torch.manual_seed(14)
        pred,aux=model(x)
        loss=loss_fn(pred,wt)+0*conditional_rc_loss(aux,target,wt)
        loss.backward()
    for (ka,pa),(kb,pb) in zip(a.named_parameters(),b.named_parameters(),strict=True):
        assert ka==kb and torch.equal(pa.grad,pb.grad)
    assert all(torch.equal(export_plain_state(a)[k],v) for k,v in export_plain_state(b).items())


def test_full_subject_weights_cover_all391():
    full,_=p.check_roles();ids=[r['case_id'] for r in full]
    # Synthetic native volumes: no new cache or patient pixels used.
    w=full_weights(ids,{case:100+i for i,case in enumerate(ids)},'subject_elastic')
    for subject in {r['subject_id'] for r in full}:
        slots=[i for i,r in enumerate(full) if r['subject_id']==subject]
        assert abs(float(w[slots].sum())-1)<1e-12
