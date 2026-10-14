import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
import torch
from sammed3d.image_blur import ImageBlur,gaussian_blur


def test_scipy_reference_axes_boundaries():
    torch.manual_seed(2);x=torch.rand(2,1,9,11,13)
    s=[.5,.91];actual=gaussian_blur(x,s)
    for i,sigma in enumerate(s):
        expected=gaussian_filter(x[i,0].numpy(),sigma=sigma,radius=3,mode='mirror')
        np.testing.assert_allclose(actual[i,0].numpy(),expected,rtol=2e-6,atol=2e-7)


def test_constant_no_case_mixing():
    x=torch.ones(2,1,7,8,9);x[1]*=4
    torch.testing.assert_close(gaussian_blur(x,[.5,1.]),x)


def test_high_frequency_reduction():
    grid=torch.arange(9).reshape(1,1,1,1,9).expand(1,1,9,9,9)%2
    blurred=gaussian_blur(grid.float(),[1.])
    assert blurred.diff(dim=-1).abs().mean()<grid.float().diff(dim=-1).abs().mean()/4


def test_labels_identity_rng_replay():
    x=torch.ones(8,1,8,8,8);labels=torch.rand(8,2,8,8,8)>.5
    before=torch.random.get_rng_state().clone();saved=labels.clone()
    a,b=ImageBlur(11),ImageBlur(11)
    for _ in range(3):
        xa,ya=a(x,labels);xb,yb=b(x,labels)
        assert torch.equal(xa,xb) and ya is labels and yb is labels and torch.equal(labels,saved)
    assert a.receipt()==b.receipt() and torch.equal(before,torch.random.get_rng_state())


def test_probability_zero_byte_identity():
    x=torch.rand(2,1,8,8,8);y=torch.ones_like(x)
    a,b=ImageBlur(1,probability=0)(x,y);assert a is x and b is y


def test_bad_sigma():
    with pytest.raises(ValueError):gaussian_blur(torch.ones(1,1,8,8,8),[0.])
