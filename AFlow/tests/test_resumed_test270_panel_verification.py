import copy
import numpy as np
import pytest
from scripts.verify_resumed_test270_panel import verify_case


def fixture():
    target=np.array([1,0,1,0,0,0,0,0],dtype=bool).reshape(2,2,2)
    pred=np.array([1,1,0,0,0,0,0,0],dtype=bool).reshape(2,2,2)
    row={'intersection':1,'predicted_voxels':2,'target_voxels':2,'exact_count_dice':.5,'dice':.5,'point_flat_index':0,'point_hit':True}
    return row,pred,target


def test_saved_masks_not_only_self_consistent_numbers():
    row,pred,target=fixture();verify_case(row,pred,target)
    changed=pred.copy();changed.flat[0]=0;changed.flat[2]=1
    # Equal aggregate counts cannot prove shape identity, but wrong actual intersection is rejected.
    changed.flat[2]=0;changed.flat[3]=1
    with pytest.raises(AssertionError):verify_case(row,changed,target)


def test_point_hit_and_float32_tolerance_are_checked():
    row,pred,target=fixture();bad=copy.deepcopy(row);bad['point_flat_index']=1
    with pytest.raises(AssertionError):verify_case(bad,pred,target)
    row['dice']+=2e-8;verify_case(row,pred,target)
    row['dice']+=1e-5
    with pytest.raises(AssertionError):verify_case(row,pred,target)
