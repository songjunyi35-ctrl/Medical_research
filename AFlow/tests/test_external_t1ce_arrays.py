import numpy as np
import pytest
import torch
from sammed3d.external_t1ce_arrays import prepare_external_t1ce_arrays
from sammed3d.t1ce_binary import prepare_t1ce_supervised_arrays,T1ceBinaryDataError


@pytest.mark.parametrize('shape,target', [((13,17,11),(8,10,12)),((8,9,10),(16,16,16))])
def test_exact_numerical_parity_with_existing_braTS_preprocessing(shape,target):
    rng=np.random.default_rng(17)
    image=rng.uniform(.1,9,shape).astype(np.float32);image[:2]=0
    labels=rng.integers(0,5,shape,dtype=np.uint8)
    old=prepare_t1ce_supervised_arrays('BraTS-GLI-99999-000',image,labels,(1.,1.2,2.),output_shape=target)
    new=prepare_external_t1ce_arrays('sub-BO118-0',image,labels,(1.,1.2,2.),output_shape=target)
    assert new.case_id=='sub-BO118-0'
    for field in ['image','target','source_target','brain_mask']:
        assert torch.equal(getattr(old,field),getattr(new,field)),field
    assert old.normalization==new.normalization and old.geometry==new.geometry


def test_input_checks_remain():
    image=np.arange(27,dtype=np.float32).reshape(3,3,3);labels=np.zeros((3,3,3),np.uint8)
    with pytest.raises(T1ceBinaryDataError):prepare_external_t1ce_arrays('sub-BA2-0',image,labels[:2],(1,1,1))
    labels[0,0,0]=7
    with pytest.raises(T1ceBinaryDataError):prepare_external_t1ce_arrays('sub-BA2-0',image,labels,(1,1,1))
    with pytest.raises(T1ceBinaryDataError):prepare_external_t1ce_arrays('pretend-BraTS',image,labels,(1,1,1))
