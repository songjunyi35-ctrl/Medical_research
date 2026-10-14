import torch
import numpy as np
import pytest
from sammed3d.t1ce_wt_capg3d import LocatorOutput
from scripts.evaluate_plain972_e193_flip8_validation import AXES,predictions,rows_for,paired

class Equivariant(torch.nn.Module):
    def __init__(self):super().__init__();self.calls=0
    def forward(self,x):self.calls+=1;return LocatorOutput(2*x+1,None)

def test_all_eight_inverse_transforms_and_batch_preserved():
    x=torch.arange(120.).reshape(2,1,3,4,5);model=Equivariant();identity,average=predictions(model,x)
    assert model.calls==8 and len(set(AXES))==8
    assert torch.equal(identity,2*x+1) and torch.equal(average,identity)
    y=torch.ones_like(x);batch={'case_id':['a','b'],'subject_id':['s1','s2']}
    rows=rows_for(identity,y,batch)
    assert [r['case_id'] for r in rows]==['a','b'] and all(r['locator_dsc']==1 for r in rows)
    assert all(r['automatic_point']==[2,3,4] for r in rows)

def test_patient_paired_delta_and_alignment():
    a=[{'case_id':str(i),'subject_id':s,'locator_dsc':v,'automatic_point_hits_gt':True} for i,(s,v) in enumerate([('a',.2),('a',.4),('b',.8)])]
    b=[{**r,'locator_dsc':r['locator_dsc']+.01} for r in a]
    d=paired(a,b);assert d['scan_delta']==pytest.approx(.01) and d['subject_delta']==pytest.approx(.01)
    assert np.allclose(d['scan_delta_ci95'],[.01,.01]) and d['point_hits_delta']==0
    with pytest.raises(AssertionError):paired(a,b[::-1])
