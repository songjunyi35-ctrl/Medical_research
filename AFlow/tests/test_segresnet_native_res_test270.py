import json
import numpy as np
import torch
import pytest
import scripts.segresnet_native_res_test_common as c


def test_test_canvas_covers_every_canonical_test_crop_and_is_divisible_by_8():
    assert all(v%8==0 for v in c.TEST_CANVAS)
    geometry=json.loads((c.CANONICAL_TEST/'index.json').read_text())['crop_geometry']
    ext=np.array([g['extent'] for g in geometry.values()]);assert ext.shape[0]==270
    assert (ext.max(0)<=np.array(c.TEST_CANVAS)).all()
    assert c.TEST_CANVAS[:2]==(160,208) and c.TEST_CANVAS[2]==168   # training canvas extended only on the last axis


def test_pad_unpad_round_trip_and_oversize_rejection():
    vol=np.random.default_rng(0).normal(size=(5,7,6)).astype(np.float32)
    padded,off=c.pad_to_canvas(vol,0.);assert padded.shape==c.TEST_CANVAS
    geo={'canvas_offset':off,'extent':list(vol.shape)};assert np.array_equal(padded[c.canvas_slices(geo)],vol)
    with pytest.raises(ValueError):c.canvas_offset((161,8,8))
    with pytest.raises(ValueError):c.canvas_offset((8,8,169))


def test_case_metrics_exclude_canvas_padding():
    geo={'canvas_offset':[10,20,30],'extent':[8,9,10]}
    label=torch.zeros(1,*c.TEST_CANVAS);logits=torch.full((1,*c.TEST_CANVAS),-5.)
    s=(slice(None),)+c.canvas_slices(geo);label[s][:, :4]=1;logits[s][:, :4]=3.
    logits[0,0,0,0]=50.;logits[0,150,200,160]=50.
    m=c.case_metrics(logits,label,geo);assert m['locator_dsc']==pytest.approx(1.0) and m['automatic_point_hits_gt']


def test_native_test_dataset_refuses_wrong_identity(tmp_path):
    base={'schema':c.SCHEMA+'.test','split':'test','test_access':True,'canvas':list(c.TEST_CANVAS),'count':270,'case_ids':['x']*270,'preprocessing':{'uses_gt_for_localization':False},'geometry':{}}
    (tmp_path/'index.json').write_text(json.dumps({**base,'test_access':False}))
    with pytest.raises(ValueError,match='declared test cache'):c.NativeTestDataset(tmp_path,['x']*270)
    (tmp_path/'index.json').write_text(json.dumps({**base,'canvas':[160,208,160]}))
    with pytest.raises(ValueError,match='canvas'):c.NativeTestDataset(tmp_path,['x']*270)
    (tmp_path/'index.json').write_text(json.dumps(base))
    with pytest.raises(ValueError,match='order/count'):c.NativeTestDataset(tmp_path,['y']*270)
