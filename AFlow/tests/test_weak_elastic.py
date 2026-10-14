import pytest
import torch
import torch.nn.functional as F
from sammed3d.weak_elastic import WeakElastic,dense_field,field_grid,displacement_bound


def arrays(shape=(24,28,32)):
    lab=torch.zeros(2,2,*shape)
    lab[:,0,6:13,6:13,6:13]=1
    lab[:,1,15:19,15:19,15:19]=1
    return lab[:,:1].clone(),lab


def test_independent_rng_replay_and_binary_alignment():
    x,y=arrays();a=WeakElastic(11,1);b=WeakElastic(11,1)
    state=torch.get_rng_state().clone()
    ax,ay=a(x,y);bx,by=b(x,y)
    assert torch.equal(state,torch.get_rng_state())
    assert torch.equal(ax,bx) and torch.equal(ay,by)
    assert torch.equal(y,arrays()[1])
    assert set(ay.unique().tolist()) == {0.,1.}
    assert ((ax>.5)==(ay[:,:1]>.5)).float().mean()>.995
    assert a.receipt()==b.receipt()
    assert a.selected==2 and a.max_bound<=.250001


def test_zero_probability_exact_identity():
    x,y=arrays();a=WeakElastic(12,0)
    ax,ay=a(x,y)
    assert ax is x and ay is y and a.selected==0


def test_boundary_gradient_amplitude_and_grid_order():
    field=dense_field(torch.rand(2,3,7,7,7,generator=torch.Generator().manual_seed(7))*2-1,(20,24,28),'cpu')
    assert field.abs().max()<=2.000001
    assert displacement_bound(field).max()<=.250001
    for axis in (2,3,4):
        assert field.select(axis,0).count_nonzero()==0
        assert field.select(axis,-1).count_nonzero()==0
    grid=field_grid(torch.zeros_like(field))
    ramp=torch.arange(20*24*28,dtype=torch.float32).reshape(1,1,20,24,28).expand(2,-1,-1,-1,-1)
    actual=F.grid_sample(ramp,grid,mode='nearest',align_corners=False)
    assert torch.equal(actual,ramp)


def test_overlap_rejected():
    x,y=arrays();y[:,1]=y[:,0]
    with pytest.raises(RuntimeError,match='overlapping'):WeakElastic(2,1)(x,y)


def test_input_validation_and_nonbinary_rejected():
    x,y=arrays()
    with pytest.raises(ValueError):WeakElastic(2)(x[0],y)
    y[:,0]=.3
    with pytest.raises(RuntimeError,match='nonbinary'):WeakElastic(2,1)(x,y)
