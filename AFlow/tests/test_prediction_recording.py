from types import SimpleNamespace
import numpy as np
import pytest
import torch
from sammed3d.prediction_recording import recording_locator


class Base(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(2.))
    def forward(self,x):
        self.last=SimpleNamespace(segmentation_logits=x*self.weight,center_logits=None)
        return self.last


class Writer:
    def __init__(self):self.rows=[]
    def append(self,key,masks,ids):self.rows.append((key,masks.copy(),list(ids)))


def test_exact_output_identity_state_and_rng():
    writer=Writer();cls,c=recording_locator(Base,writer,['a','b'],['one'])
    m=cls();before=torch.random.get_rng_state().clone();x=torch.tensor([-1.,2.]).reshape(2,1,1,1,1)
    out=m(x)
    assert out is m.last and torch.equal(out.segmentation_logits,x*2)
    assert list(m.state_dict())==['weight'] and torch.equal(before,torch.random.get_rng_state())
    assert writer.rows[0][0]=='one' and writer.rows[0][2]==['a','b']
    assert np.array_equal(writer.rows[0][1].flatten(),[False,True])
    assert c['forwards']==2 and c['rows_by_key']=={'one':2}


def test_independent_offsets_and_no_extra_models():
    w=Writer();cls,c=recording_locator(Base,w,['a','b'],['first','second'])
    a,b=cls(),cls();x=torch.ones(1,1,1,1,1)
    a(x);a(x);b(torch.ones(2,1,1,1,1))
    assert [r[2] for r in w.rows]==[['a'],['b'],['a','b']]
    assert c['rows_by_key']=={'first':2,'second':2}
    with pytest.raises(RuntimeError):a(x)
    with pytest.raises(RuntimeError):cls()
