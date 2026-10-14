import json
import numpy as np
import pytest
import torch
from scripts import followup_test270_common as c


def test_frozen_canvas_covers_metadata_without_pixel_access():
    index=json.loads((c.CANONICAL_TEST/'index.json').read_text())
    assert c.TEST_CANVAS==(160,208,168)
    assert len(index['case_ids'])==270 and len(set(index['case_ids']))==270
    assert all(max(g['extent'][i] for g in index['crop_geometry'].values())<=c.TEST_CANVAS[i] for i in range(3))


def test_padding_roundtrip_and_metrics_ignore_padding():
    volume=np.ones((4,5,6),np.uint8);padded,offset=c.pad_to_canvas(volume)
    geo={'extent':[4,5,6],'canvas_offset':offset};assert np.array_equal(padded[c.canvas_slices(geo)],volume)
    label=torch.from_numpy(padded.copy())[None].float();logits=label*2-1;logits[0,0,0,0]=100
    assert c.case_metrics(logits,label,geo)['locator_dsc']==1
    assert c.case_metrics(logits,label,geo)['automatic_point_hits_gt']
    with pytest.raises(ValueError):c.canvas_offset((160,208,169))


def test_test_dataset_identity_and_role_guards(tmp_path):
    ids=[f'BraTS-GLI-{i:05d}-100' for i in range(270)]
    meta={'schema':c.SCHEMA+'.test','split':'test','test_access':True,'canvas':list(c.TEST_CANVAS),
          'count':269,'case_ids':ids,'preprocessing':{'uses_gt_for_localization':False},'geometry':{}}
    (tmp_path/'index.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='identity/count'):c.NativeTestDataset(tmp_path,ids)
    meta.update(count=270,test_access=False);(tmp_path/'index.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='declared test cache'):c.NativeTestDataset(tmp_path,ids)


def test_once_only_receipt(tmp_path):
    p=tmp_path/'MODEL_PREACCESS.json';c.write_new(p,{'first':True})
    with pytest.raises(FileExistsError):c.write_new(p,{'first':False})
    assert json.loads(p.read_text())=={'first':True}
