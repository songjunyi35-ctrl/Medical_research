import torch
import pytest
from scripts.sammed3d_mumford_shah_loss import region_tv,mumford_shah,FirstPromptMumfordShah,REFERENCE_WEIGHT
from scripts.train_sammed3d_mumford_shah_972 import windows


def test_2d_author_sum_and_gradient_equivalence():
    torch.manual_seed(74)
    p=torch.softmax(torch.randn(2,2,7,5),dim=1).detach().requires_grad_(True)
    x=torch.rand(2,1,7,5)
    region,tv=region_tv(p,x)
    oracle=p.new_zeros(())
    for b in range(2):
        for k in range(2):
            values=x[b,0];weights=p[b,k]
            mean=(values*weights).sum()/weights.sum()
            oracle=oracle+(weights*(values-mean)**2).sum()
    tv_reference=(p[:,:,1:]-p[:,:,:-1]).abs().sum()+(p[:,:,:,1:]-p[:,:,:,:-1]).abs().sum()
    target=oracle+.001*tv_reference
    got=(region+.001*tv)*2*7*5
    assert torch.allclose(got,target,rtol=1e-6,atol=1e-6)
    a=torch.autograd.grad(got,p,retain_graph=True)[0];b=torch.autograd.grad(target,p)[0]
    assert torch.allclose(a,b,rtol=1e-5,atol=1e-6)
    assert REFERENCE_WEIGHT==pytest.approx(1e-6*4*256**2)


def test_3d_background_and_axis_energy_and_batch_invariance():
    torch.manual_seed(81)
    z=torch.randn(1,1,3,4,5,requires_grad=True);x=torch.rand_like(z)
    loss,region,tv=mumford_shah(z,x)
    assert region>0 and tv>0
    swapped=mumford_shah(-z,x)[0]
    assert torch.allclose(loss,swapped,atol=1e-7)
    assert torch.allclose(loss,mumford_shah(z.repeat(3,1,1,1,1),x.repeat(3,1,1,1,1))[0],atol=1e-7)
    assert torch.allclose(loss,mumford_shah(z,x*17+12)[0],atol=1e-6)
    loss.backward();assert torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
    flat=torch.zeros(1,1,3,4,5)
    assert mumford_shah(flat,torch.ones_like(flat))[0]==0


def test_first_prompt_and_empty_target():
    x=torch.arange(64.).reshape(1,1,4,4,4);g=(x>32).float();z=torch.zeros_like(x,requires_grad=True)
    f=FirstPromptMumfordShah(x)
    assert torch.allclose(f(z,g),f.seg(z,g)+mumford_shah(z,x)[0])
    for _ in range(10):assert torch.allclose(f(z,g),f.seg(z,g))
    assert f.calls==11
    with pytest.raises(ValueError):FirstPromptMumfordShah(x)(z,torch.zeros_like(g))


def test_two_update_windows_cover_every_case_once_and_tail_exact():
    a=windows(1,0);b=windows(1,1);full=[]
    assert list(map(len,a))==[63,59]
    for aw,bw,total in zip(a,b,[504,468]):
        ids=[]
        for x,y in zip(aw,bw):
            assert len(x)==len(y);ids+=x+y
        assert len(ids)==total;full+=ids
    assert sorted(full)==list(range(972)) and len(set(full))==972
    assert len(a[-1][-1])==2 and len(b[-1][-1])==2
    assert windows(1,0)!=windows(2,0)
