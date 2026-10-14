import numpy as np
import pytest
from sammed3d.error_anatomy import count_errors


def test_exact_decomposition():
    labels=np.array([0,1,2,3,4,2,0,4],dtype=np.uint8).reshape(2,2,2)
    pred=np.array([1,1,0,1,1,0,0,0],dtype=bool).reshape(2,2,2)
    r=count_errors(pred,labels,4)
    assert (r['intersection'],r['target_voxels'],r['predicted_voxels'])==(2,4,4)
    assert r['false_negatives_by_class']=={'1':0,'2':2,'3':0}
    assert r['false_positives_by_class']=={'0':1,'4':1}
    assert not r['point_hit'] and r['point_class']==4


def test_perfect_wt_and_invalid_labels():
    labels=np.arange(8,dtype=np.uint8).reshape(2,2,2)%5
    r=count_errors(np.isin(labels,[1,2,3]),labels,1)
    assert r['false_positive_voxels']==r['false_negative_voxels']==0 and r['point_hit']
    labels.flat[0]=5
    with pytest.raises(ValueError):count_errors(np.zeros_like(labels),labels,1)
