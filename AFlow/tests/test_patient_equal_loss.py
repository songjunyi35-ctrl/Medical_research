import pytest
import torch
from monai.losses import DiceCELoss
from sammed3d.patient_equal_loss import PatientEqualDiceCE,patient_weights


def test_equal_patient_mass_and_global_mean():
    rows=[{'case_id':f'{s}-{i}','subject_id':s} for s,n in [('a',1),('b',2),('c',10)] for i in range(n)]
    w=patient_weights(rows)
    assert sum(w.values())==pytest.approx(len(rows))
    for s in ['a','b','c']:assert sum(w[r['case_id']] for r in rows if r['subject_id']==s)==pytest.approx(len(rows)/3)
    with pytest.raises(ValueError):patient_weights(rows+rows[:1])


def test_all_one_weights_preserve_existing_loss_and_gradients():
    torch.manual_seed(1);a=torch.randn(4,1,3,4,5,requires_grad=True);b=a.detach().clone().requires_grad_();y=(torch.rand_like(a)>.8).float()
    old=DiceCELoss(sigmoid=True,squared_pred=False,lambda_dice=1.,lambda_ce=.1)(a,y)
    new=PatientEqualDiceCE()(b,y,torch.ones(4));old.backward();new.backward()
    assert torch.allclose(old,new,atol=1e-6) and torch.allclose(a.grad,b.grad,atol=1e-6)


def test_weights_are_not_renormalized_inside_batch():
    x=torch.randn(2,1,3,3,3,requires_grad=True);y=torch.ones_like(x);criterion=PatientEqualDiceCE()
    a=criterion(x,y,torch.ones(2));b=criterion(x,y,torch.ones(2)*2)
    assert torch.allclose(b,2*a,atol=1e-6)
    with pytest.raises(ValueError):criterion(x,y,torch.tensor([1.,0.]))


def test_full972_single_batch8_schedule():
    from scripts.patient_equal_common import batches
    for epoch in (1,100,200):
        values=batches(epoch)
        assert len(values)==122 and len(values[-1])==4
        assert sorted(i for row in values for i in row)==list(range(972))
