"""Pure development mask contracts: reject accidental shape broadcasting."""
from pathlib import Path
import numpy as np


def validate_index(index,case_ids,keys,directory):
    assert index['schema']=='aflow.packed_prediction_masks'
    assert len(case_ids)==len(set(case_ids))==180 and index['case_ids']==list(case_ids)
    assert index['shape']==[128,128,128] and index['cases']==180
    assert index['bitorder']=='little' and index['flatten_order']=='C'
    assert index['bits_per_case']==128**3 and index['bytes_per_case']==128**3//8 and index['padding_bits_per_case']==0
    assert set(index['files'])==set(keys)
    for key,info in index['files'].items():
        assert Path(info['path'])==Path(directory)/f'{key}.bits'
        assert info['cases']==180 and info['bytes']==180*128**3//8


def validate_case(row,prediction,truth,expected_shape=(128,128,128)):
    if prediction.shape!=truth.shape or prediction.shape!=tuple(expected_shape):raise ValueError('exact grid required, broadcasting forbidden')
    if prediction.dtype!=np.bool_ or truth.dtype!=np.bool_:raise ValueError('boolean masks required')
    nt=int(truth.sum());npred=int(prediction.sum());inter=int((prediction&truth).sum())
    assert nt>0
    assert (inter,npred,nt)==(row['intersection'],row['predicted_voxels'],row['target_voxels'])
    assert row['dice']==2*inter/(nt+npred)
    point=row['point_flat_index'];assert isinstance(point,int) and not isinstance(point,bool) and 0<=point<truth.size
    assert bool(truth.reshape(-1)[point])==row['point_hit']
