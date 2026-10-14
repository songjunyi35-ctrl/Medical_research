import numpy as np
import pytest
from sammed3d.peak_protected_cleanup import cleanup_logits


def test_removes_tiny_nonpeak_preserves_peak_and_large_components():
    x=np.full((24,24,24),-2.,dtype=np.float32)
    x[2:12,2:12,2:12]=2  # 1000: cutoff10
    x[18:21,18:21,18:21]=1
    x[15,2,2]=5  # small peak island must stay
    x[15,15,2]=3  # other one voxel is removed
    before=x.copy();y,r=cleanup_logits(x)
    assert np.array_equal(x,before)
    assert np.argmax(y)==np.argmax(x) and y[15,2,2]==5 and y[15,15,2]<0
    assert r['cutoff_voxels']==10 and r['removed_voxels']==1 and r['components_after']==3
    assert np.array_equal(cleanup_logits(y)[0],y)


def test_26_connected_diagonal_and_zero_threshold():
    x=np.full((12,12,12),-1.,dtype=np.float32)
    for i in range(10):x[i,i,i]=1
    x[0,11,0]=2;x[0,11,11]=0
    y,r=cleanup_logits(x)
    assert r['components_before']==2 and r['removed_voxels']==0
    assert not (y>0)[0,11,11]


def test_empty_and_sub8_largest_ties():
    x=np.full((12,12,12),-2.,dtype=np.float32)
    y,r=cleanup_logits(x);assert np.array_equal(x,y) and r['components_before']==0
    x[1:3,1:3,1]=1;x[7:9,7:9,7]=1;x[1,10,10]=4;x[10,1,10]=2
    y,r=cleanup_logits(x)
    assert r['cutoff_voxels']==4 and r['removed_voxels']==1
    assert (y[1:3,1:3,1]>0).all() and (y[7:9,7:9,7]>0).all()
    assert int(y.argmax())==int(x.argmax())


def test_tied_peak_first_preserved_and_deterministic():
    x=np.full((20,20,20),-1.,dtype=np.float32)
    x[4:14,4:14,4:14]=1;x[0,0,0]=3;x[19,19,19]=3
    y,r=cleanup_logits(x);z,s=cleanup_logits(x)
    assert y.argmax()==x.argmax()==0 and r['removed_voxels']==1
    assert np.array_equal(y,z) and r==s


@pytest.mark.parametrize('x',[np.ones((3,3)),np.ones((3,3,3),dtype=int),np.full((3,3,3),np.nan)])
def test_invalid_inputs(x):
    with pytest.raises(ValueError):cleanup_logits(x)
