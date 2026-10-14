import numpy as np
from scripts.evaluate_peak_cleanup import score,gates_for


def test_score_uses_same_zero_threshold_and_argmax():
    x=np.array([2.,0.,-1.,1.],dtype=np.float32).reshape(1,2,2)
    y=np.array([1,1,0,0],dtype=bool).reshape(1,2,2)
    r=score(x,y,'case','subject')
    assert r['dice']==.5 and r['intersection']==1 and r['predicted_voxels']==2
    assert r['point_flat_index']==0 and r['point_hit']


def test_point_exact_is_stricter_than_unchanged_net_count():
    c={'subject_delta':.005,'scan_delta':.004,'point_delta':0,'subject_95ci':[-.002,.01],'scan_cluster_95ci':[-.002,.01]}
    assert all(gates_for(c,True).values())
    assert not all(gates_for(c,False).values())
    c['scan_delta']=0
    assert not all(gates_for(c,True).values())
