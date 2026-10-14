import pytest
import torch
from sammed3d.small_rotation import SmallRotation,rotate_pair,rotation_matrix


def pair():
    y=torch.zeros(3,2,16,16,16);y[:,0,5:8,6:9,7:10]=1;y[:,1,9:11,8:10,6:8]=1
    return y[:,:1].clone(),y


def test_identity():
    x,y=pair();a,b=rotate_pair(x,y,torch.zeros(3,3))
    torch.testing.assert_close(x,a,atol=1e-6,rtol=0);assert torch.equal(y,b)


def test_quarter_turn_coordinate_mapping():
    x=torch.zeros(1,1,9,9,9);x[0,0,4,4,6]=1
    a,b=rotate_pair(x,x,torch.tensor([[0.,0.,torch.pi/2]]))
    assert b[0,0,4,2,4]==1 and b.sum()==1
    torch.testing.assert_close(a,b,atol=1e-6,rtol=0)


def test_rigid_matrix():
    a=torch.tensor([[.1,.2,-.15],[-.2,-.2,.2]])
    r=rotation_matrix(a)
    torch.testing.assert_close(r@r.transpose(-1,-2),torch.eye(3).expand(2,3,3))
    torch.testing.assert_close(torch.linalg.det(r),torch.ones(2))


def test_randomness_and_pairing():
    x,y=pair();s=torch.random.get_rng_state().clone()
    a=SmallRotation(11,probability=1);b=SmallRotation(11,probability=1)
    for _ in range(4):
        xa,ya=a(x,y);xb,yb=b(x,y)
        assert torch.equal(xa,xb) and torch.equal(ya,yb)
        assert set(ya.unique().tolist())<={0,1} and not (ya[:,0]*ya[:,1]).any()
        assert (ya[:,0].flatten(1).sum(1)>0).all()
    assert torch.equal(s,torch.random.get_rng_state()) and a.receipt()==b.receipt()
    assert a.receipt()['max_observed_degrees']<=15


def test_disabled_byte_identity():
    x,y=pair();a,b=SmallRotation(1,probability=0)(x,y)
    assert a is x and b is y


def test_non_cube_rejected():
    with pytest.raises(ValueError):rotate_pair(torch.zeros(1,1,4,5,6),torch.zeros(1,1,4,5,6),torch.zeros(1,3))
