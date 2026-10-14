import numpy as np
import pytest
from sammed3d.subregion_labels import regions,torch_sample,numpy_sample


def test_original_brats_mapping_not_recoded_example():
    a=np.arange(5,dtype=np.uint8);r=regions(a)
    assert r['whole_tumor'].tolist()==[False,True,True,True,False]
    assert r['tumor_core'].tolist()==[False,True,False,True,False]
    assert r['enhancing_tissue'].tolist()==[False,False,False,True,False]
    assert r['resection_cavity'].tolist()==[False,False,False,False,True]
    assert np.all(r['enhancing_tissue']<=r['tumor_core']) and np.all(r['tumor_core']<=r['whole_tumor'])
    assert not (r['whole_tumor']&r['resection_cavity']).any()


@pytest.mark.parametrize('size',[4,8,16])
def test_independent_sampling_including_noninteger_scales(size):
    a=(np.arange(19*23*17).reshape(19,23,17)%5).astype(np.uint8)
    assert np.array_equal(torch_sample(a,[2,3,1],[13,17,15],size),numpy_sample(a,[2,3,1],[13,17,15],size))


def test_invalid_label_rejected():
    with pytest.raises(ValueError):regions(np.array([0,5]))
