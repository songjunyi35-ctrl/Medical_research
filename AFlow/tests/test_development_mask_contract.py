import copy
from pathlib import Path
import numpy as np
import pytest
from sammed3d.development_mask_contract import validate_index,validate_case


def test_index_shape_case_order_and_paths():
    ids=[f'case{i}' for i in range(180)];directory=Path('/tmp/synthetic_mask_contract')
    index={'schema':'aflow.packed_prediction_masks','shape':[128]*3,'cases':180,'case_ids':ids,'bitorder':'little','flatten_order':'C','bits_per_case':128**3,'bytes_per_case':128**3//8,'padding_bits_per_case':0,'files':{'a':{'path':str(directory/'a.bits'),'cases':180,'bytes':180*128**3//8}}}
    validate_index(index,ids,['a'],directory)
    for field,value in [('shape',[1,1,1]),('cases',179),('bytes_per_case',1)]:
        bad=copy.deepcopy(index);bad[field]=value
        with pytest.raises(AssertionError):validate_index(bad,ids,['a'],directory)
    bad=copy.deepcopy(index);bad['files']['a']['path']='/tmp/another.bits'
    with pytest.raises(AssertionError):validate_index(bad,ids,['a'],directory)


def test_actual_counts_and_point_not_only_arithmetic():
    truth=np.zeros((2,2,2),dtype=bool);truth.flat[0]=True;pred=truth.copy()
    row={'intersection':1,'predicted_voxels':1,'target_voxels':1,'dice':1.,'point_flat_index':0,'point_hit':True}
    validate_case(row,pred,truth,expected_shape=(2,2,2))
    changed=pred.copy();changed.flat[0]=False;changed.flat[1]=True
    with pytest.raises(AssertionError):validate_case(row,changed,truth,expected_shape=(2,2,2))
    with pytest.raises(ValueError):validate_case(row,np.ones((1,1,1),dtype=bool),truth,expected_shape=(2,2,2))
