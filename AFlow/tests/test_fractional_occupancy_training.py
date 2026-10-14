import torch
from monai.losses import DiceCELoss
from sammed3d.structured_background_aux import conditional_rc_loss


def test_existing_loss_accepts_soft_targets_and_produces_finite_changed_gradients():
    hard=torch.zeros(2,1,16,16,16);hard[:,:,4:12,4:12,4:12]=1
    soft=hard.clone();soft[:,:,4,4:12,4:12]=.3;soft[:,:,12,4:12,4:12]=.2
    loss_fn=DiceCELoss(sigmoid=True,squared_pred=False,lambda_dice=1,lambda_ce=.1)
    gradients=[]
    for target in (hard,soft):
        logits=torch.zeros_like(target,requires_grad=True)
        value=loss_fn(logits,target)+0*conditional_rc_loss(logits,torch.zeros_like(target),target)
        value.backward();assert torch.isfinite(value) and torch.isfinite(logits.grad).all()
        gradients.append(logits.grad)
    assert not torch.equal(gradients[0],gradients[1])


def test_binary_special_case_matches_exact_original_loss():
    y=torch.zeros(2,1,16,16,16);y[:,:,3:8,3:8,3:8]=1
    logits=torch.randn(y.shape,generator=torch.Generator().manual_seed(1))
    criterion=DiceCELoss(sigmoid=True,squared_pred=False,lambda_dice=1,lambda_ce=.1)
    assert torch.equal(criterion(logits,y),criterion(logits,y.half().float()))
