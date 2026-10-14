import numpy as np
import pytest
from sammed3d.fractional_occupancy import fractional_occupancy


def overlap_matrix(n,m):
    return np.array([[max(0,min(i+1,(o+1)*n/m)-max(i,o*n/m))/(n/m) for i in range(n)] for o in range(m)])


@pytest.mark.parametrize('shape,out', [((3,5,7),(2,4,3)),((2,3,4),(5,7,9)),((7,4,3),(4,8,2))])
def test_matches_independent_cell_overlap_tensor_product(shape,out):
    mask=(np.random.default_rng(11).random(shape)>.6).astype(np.uint8)
    matrices=[overlap_matrix(n,m) for n,m in zip(shape,out)]
    expected=np.einsum('ai,bj,ck,ijk->abc',*matrices,mask,optimize=True)
    actual=fractional_occupancy(mask,out)
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
    assert abs(actual.sum()*np.prod(shape)/np.prod(out)-mask.sum())<1e-10


def test_half_cell_noninteger_and_identity():
    assert fractional_occupancy(np.array([1,0]).reshape(2,1,1),(1,1,1)).item()==.5
    a=fractional_occupancy(np.array([1,0,0]).reshape(3,1,1),(2,1,1))
    np.testing.assert_allclose(a.ravel(),[2/3,0],rtol=0,atol=1e-15)
    x=(np.random.default_rng(12).random((9,7,5))>.7).astype(np.uint8)
    np.testing.assert_array_equal(fractional_occupancy(x,x.shape),x)


@pytest.mark.parametrize('value',[0,1])
def test_constants_and_float16_mass(value):
    mask=np.full((17,19,23),value,dtype=np.uint8)
    q=fractional_occupancy(mask,(16,16,16))
    np.testing.assert_allclose(q,value,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(q.astype(np.float16),np.full((16,16,16),value))


def test_realistic_shape_quantization_range_and_mass():
    mask=np.zeros((136,174,142),dtype=np.uint8);mask[42:75,61:80,50:68]=1;mask[12,12,12]=1
    q=fractional_occupancy(mask);stored=q.astype(np.float16)
    expected=mask.sum()*128**3/np.prod(mask.shape)
    assert abs(stored.sum(dtype=np.float64)-expected)/expected<.001
    assert stored.max()==1 and stored.min()==0 and ((stored>0)&(stored<1)).any()


@pytest.mark.parametrize('mask,out',[(np.full((3,3,3),.5),(2,2,2)),(np.ones((3,3)),(2,2,2)),(np.ones((3,3,3)),(0,2,2)),(np.ones((3,3,3)),(2.5,2,2))])
def test_invalid_inputs(mask,out):
    with pytest.raises(ValueError):fractional_occupancy(mask,out)
