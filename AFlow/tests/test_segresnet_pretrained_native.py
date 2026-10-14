import json
import numpy as np
import torch
import pytest
import scripts.segresnet_pretrained_native_common as c


def test_canvas_covers_every_canonical_crop_and_is_divisible_by_8():
    assert all(v%8==0 for v in c.CANVAS)
    for split,path in c.CANONICAL.items():
        geometry=json.loads((path/'index.json').read_text())['crop_geometry']
        ext=np.array([g['extent'] for g in geometry.values()])
        assert ext.shape[0]==(972 if split=='train' else 108)
        assert (ext.max(0)<=np.array(c.CANVAS)).all()


def test_pad_unpad_round_trip_and_oversize_rejection():
    vol=np.random.default_rng(0).normal(size=(5,7,6)).astype(np.float32)
    padded,off=c.pad_to_canvas(vol,0.)
    assert padded.shape==c.CANVAS and padded.dtype==np.float32
    geo={'canvas_offset':off,'extent':list(vol.shape)}
    assert np.array_equal(padded[c.canvas_slices(geo)],vol)
    assert float(np.abs(padded).sum())==pytest.approx(float(np.abs(vol).sum()))
    with pytest.raises(ValueError):c.canvas_offset((c.CANVAS[0]+1,8,8))
    with pytest.raises(ValueError):c.canvas_offset((0,8,8))


@pytest.mark.parametrize('per_gpu',c.BATCH_CANDIDATES)
def test_full972_coverage_equal_ranks_even_tail(per_gpu):
    left=list(c.batches_for_epoch(1,0,per_gpu));right=list(c.batches_for_epoch(1,1,per_gpu))
    assert len(left)==len(right)==c.steps_per_epoch(per_gpu)
    ids=[]
    for a,b in zip(left,right):assert len(a)==len(b)>0;ids.extend(a+b)
    assert sorted(ids)==list(range(972))
    assert list(c.batches_for_epoch(2,0,per_gpu))!=left
    assert 2*c.steps_per_epoch(per_gpu)*per_gpu>=972


def test_case_metrics_exclude_canvas_padding():
    geo={'canvas_offset':[10,20,30],'extent':[8,9,10]}
    label=torch.zeros(1,*c.CANVAS);logits=torch.full((1,*c.CANVAS),-5.)
    s=(slice(None),)+c.canvas_slices(geo)
    label[s][:, :4]=1;logits[s][:, :4]=3.          # perfect inside the crop
    logits[0,0,0,0]=50.;logits[0,150,200,150]=50. # junk in the padding must be ignored
    m=c.case_metrics(logits,label,geo)
    assert m['locator_dsc']==pytest.approx(1.0)
    assert m['automatic_point_hits_gt'] and 0<=m['automatic_point'][0]<4
    assert m['crop_extent']==[8,9,10]
    empty=torch.zeros_like(label);assert c.case_metrics(logits,empty,geo)['locator_dsc']==0.0


def test_native_dataset_rejects_test_or_wrong_canvas_cache(tmp_path):
    base={'schema':c.SCHEMA,'split':'train','canvas':list(c.CANVAS),'count':1,'case_ids':['BraTS-GLI-00001-100'],'preprocessing':{'uses_gt_for_localization':False},'geometry':{}}
    (tmp_path/'index.json').write_text(json.dumps({**base,'test_access':True}))
    with pytest.raises(ValueError,match='test-accessible'):c.NativeCanvasDataset(tmp_path,expected_split='train')
    (tmp_path/'index.json').write_text(json.dumps({**base,'test_access':False,'canvas':[128,128,128]}))
    with pytest.raises(ValueError,match='canvas'):c.NativeCanvasDataset(tmp_path,expected_split='train')
    (tmp_path/'index.json').write_text(json.dumps({**base,'test_access':False}))
    with pytest.raises(ValueError,match='split'):c.NativeCanvasDataset(tmp_path,expected_split='validation')


def test_augmentation_never_drops_the_case_when_gt_would_disappear(monkeypatch):
    image=torch.arange(16.).reshape(2,1,2,2,2);label=torch.zeros_like(image);label[...,0,0,0]=1
    monkeypatch.setattr(c,'augment_batch',lambda x,y,max_shift:(x+99,torch.zeros_like(y)))
    x,y,count=c.safe_augment(image.clone(),label.clone())
    assert count==2 and torch.equal(x,image) and torch.equal(y,label)
    with pytest.raises(RuntimeError):c.safe_augment(image,torch.zeros_like(label))


def test_schedule_endpoints():
    f=lambda e:c.learning_rate_for_epoch(e,base_lr=2e-4,min_lr=1e-5,constant_epochs=100,total_epochs=200)
    assert f(1)==f(100)==f(101)==2e-4 and f(200)==pytest.approx(1e-5)
    assert all(f(i)>=f(i+1) for i in range(101,200))


def test_pretrained_init_shapes_and_frozen_batch():
    assert c.BATCH_CANDIDATES==(4,) and c.steps_per_epoch(4)==122
    assert c.sha(c.PRETRAIN_CKPT)==c.PRETRAIN_SHA
    model=c.model_for(torch.device('cpu'))
    state=torch.load(c.PRETRAIN_CKPT,map_location='cpu',weights_only=False)
    assert state['epoch']==150
    model.load_state_dict(state['model_state_dict'],strict=True)
    assert all(torch.equal(model.state_dict()[k],v) for k,v in state['model_state_dict'].items())

def test_validation_numeric_mode_restores_training_flags():
    before=(torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark,torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    @c.deterministic_validation
    def probe():
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
        assert not torch.backends.cuda.matmul.allow_tf32
    probe()
    assert before==(torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark,torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
