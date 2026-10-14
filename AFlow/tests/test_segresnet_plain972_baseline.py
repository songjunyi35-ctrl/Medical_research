import json
import torch
import pytest
import scripts.segresnet_plain972_common as c
from scripts.train_t1ce_wt_capg3d import CACHE_SCHEMA


def test_full972_no_replacement_or_tail_drop():
    left=list(c.batches_for_epoch(1,0));right=list(c.batches_for_epoch(1,1));assert len(left)==len(right)==122
    ids=[]
    for a,b in zip(left,right):assert len(a)==len(b);ids.extend(a+b)
    assert sorted(ids)==list(range(972)) and len(set(ids))==972
    assert len(left[-1])==len(right[-1])==2
    assert list(c.batches_for_epoch(2,0))!=left


def test_augmentation_never_drops_the_case_when_gt_would_disappear(monkeypatch):
    image=torch.arange(16.).reshape(2,1,2,2,2);label=torch.zeros_like(image);label[...,0,0,0]=1
    expected_image=image.clone();expected_label=label.clone()
    def destroy(x,y):return x+99,torch.zeros_like(y)
    monkeypatch.setattr(c,'augment_batch',lambda x,y,max_shift:destroy(x,y))
    x,y,count=c.safe_augment(image,label)
    assert count==2 and torch.equal(x,expected_image) and torch.equal(y,expected_label)
    with pytest.raises(RuntimeError):c.safe_augment(image,torch.zeros_like(label))


def test_test_cache_rejected_before_payload_open(tmp_path):
    (tmp_path/'index.json').write_text(json.dumps({'schema':CACHE_SCHEMA,'test_access':True,'split':'test'}))
    with pytest.raises(ValueError,match='test-accessible'):c.BrainCropDataset(tmp_path,expected_split='train')


def test_historical_schedule_endpoints():
    f=lambda e:c.learning_rate_for_epoch(e,base_lr=2e-4,min_lr=1e-5,constant_epochs=100,total_epochs=200)
    assert f(1)==f(100)==f(101)==2e-4
    assert f(200)==pytest.approx(1e-5)
    assert all(f(i)>=f(i+1) for i in range(101,200))
