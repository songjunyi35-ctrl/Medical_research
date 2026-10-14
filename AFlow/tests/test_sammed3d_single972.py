import copy
import pytest
import torch
from scripts.sammed3d_graymean_single972 import gray_loss, FirstPromptLoss
from scripts.sammed3d_full972_protocol import check_roles,validate_rows,TEST
import json

def test_original_split_counts_and_subject_separation():
 fit,val=check_roles();assert len(fit)==972 and len(val)==108
 assert len({r['subject_id'] for r in fit})==444
 test=[json.loads(x) for x in TEST.read_text().splitlines()]
 broken=copy.deepcopy(val);broken[0]['subject_id']=fit[0]['subject_id']
 with pytest.raises(ValueError):validate_rows(fit,broken,test)

def test_gray_gradient_and_empty_target_guard():
 x=torch.tensor([0.,0.,2.,2.]).reshape(1,1,1,1,4);g=(x>0).float();z=torch.zeros_like(x,requires_grad=True)
 loss=gray_loss(z,x,g);assert float(loss)==pytest.approx(1.)
 loss.backward();assert (z.grad[...,:2]>0).all() and (z.grad[...,2:]<0).all()
 with pytest.raises(ValueError):gray_loss(z,x,torch.zeros_like(g))

def test_gray_applies_only_to_initial_prompt():
 x=torch.arange(8.).reshape(1,1,2,2,2);g=(x>3).float();z=torch.zeros_like(x,requires_grad=True)
 loss=FirstPromptLoss(x,1.)
 assert torch.allclose(loss(z,g),loss.seg(z,g)+gray_loss(z,x,g))
 for _ in range(10):assert torch.allclose(loss(z,g),loss.seg(z,g))
 assert loss.calls==11

def test_constant_intensity_does_not_guarantee_correct_geometry():
 x=torch.ones(1,1,2,2,2)*3;g=torch.zeros_like(x);g[...,0]=1
 assert float(gray_loss(torch.randn_like(x),x,g))<1e-10
