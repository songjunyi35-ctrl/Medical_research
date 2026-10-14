import numpy as np
import pytest
from sammed3d.letterbox_fidelity import layout,encode,decode


def test_known_layout():
    shape,slices=layout((136,174,142))
    assert shape==(100,128,104)
    assert [(s.start,s.stop) for s in slices]==[(14,114),(0,128),(12,116)]


def test_identity_cube():
    x=np.arange(512,dtype=np.float32).reshape(8,8,8)
    assert np.array_equal(encode(x,8),x)
    assert np.array_equal(decode(x,x.shape),x)


def test_integer_upscale_roundtrip_and_padding():
    x=np.arange(24,dtype=np.float32).reshape(2,3,4)
    box=encode(x,8);shape,slices=layout(x.shape,8)
    assert shape==(4,6,8)
    assert np.array_equal(decode(box,x.shape),x)
    copy=box.copy();copy[slices]=0;assert not copy.any()


def test_binary_and_components_ids_share_mapping():
    x=np.zeros((7,11,9),dtype=np.float32);x[1:3,2:4,4:7]=1;x[4:6,7:10,1:3]=2
    assert np.array_equal(encode(x,8)>0,encode(x>0,8)>0)


def test_invalid_shape():
    with pytest.raises(ValueError):layout((0,2,3))
