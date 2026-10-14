import pytest
import torch
from sammed3d.segresnet_semantic_wt import SemanticSegResNet, SemanticLoss, wt_logit, map_classes
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from scripts.segresnet_two_ideas_common import batches_for_epoch, steps_per_epoch, lr_for


def test_union_matches_softmax_and_excludes_cavity():
    torch.manual_seed(3)
    z=torch.randn(2,4,3,4,5,requires_grad=True)
    assert torch.allclose(wt_logit(z).sigmoid(),z.softmax(1)[:,1:].sum(1,keepdim=True),atol=1e-6)
    c=map_classes(torch.tensor([0.,1.,2.,3.,4.]))
    assert c.tolist()==[0,1,2,3,0]
    with pytest.raises(ValueError): map_classes(torch.tensor([1.5]))


def test_initialization_preserves_binary_function():
    torch.set_num_threads(2); torch.manual_seed(19)
    binary=AutomaticPromptLocator(center_aware=False).eval()
    semantic=SemanticSegResNet().eval(); semantic.initialize_from_binary(binary.state_dict())
    x=torch.randn(1,1,16,16,16)
    with torch.no_grad():
        a=binary(x).segmentation_logits; b=semantic(x).segmentation_logits
    assert torch.allclose(a,b,atol=2e-6)


def test_loss_finite_with_absent_classes_and_backward():
    z=torch.randn(2,4,4,4,4,requires_grad=True)
    c=torch.zeros(2,1,4,4,4); c[0,:,1:3,1:3,1:3]=2; c[1,:,1:3,1:3,1:3]=3
    loss=SemanticLoss()(z,c); loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
    with pytest.raises(ValueError): SemanticLoss()(z,torch.zeros_like(c))


@pytest.mark.parametrize('n',[972,334])
def test_full_coverage_and_tail(n):
    ranks=[list(batches_for_epoch(n,20260819,7,r)) for r in (0,1)]
    assert len(ranks[0])==len(ranks[1])==steps_per_epoch(n)
    ids=[v for row in ranks for b in row for v in b]
    assert sorted(ids)==list(range(n))
    assert all(len(a)==len(b) for a,b in zip(*ranks))
    assert len(ranks[0][-1])==(n%8)//2


def test_schedules():
    assert lr_for('pretrain',75)==pytest.approx(2e-4)
    assert lr_for('pretrain',150)==pytest.approx(1e-5)
    assert lr_for('finetune',100)==pytest.approx(2e-4)
    assert lr_for('finetune',200)==pytest.approx(1e-5)
