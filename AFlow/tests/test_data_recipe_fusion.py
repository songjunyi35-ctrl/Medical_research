import pytest
import torch
from sammed3d.data_recipe_fusion import equal_recipe_logits


def test_exact_fixed_rule_no_input_mutation():
    a=torch.tensor([1.,-2.,4.]);b=torch.tensor([-3.,6.,2.]);ac=a.clone();bc=b.clone()
    assert torch.equal(equal_recipe_logits(a,b),torch.tensor([-1.,2.,3.]))
    assert torch.equal(a,ac) and torch.equal(b,bc)
    assert torch.equal(equal_recipe_logits(a,b),equal_recipe_logits(b,a))


def test_same_member_noop_and_invalid_inputs():
    a=torch.tensor([1.,2.],dtype=torch.bfloat16)
    assert torch.equal(equal_recipe_logits(a,a),a.float())
    with pytest.raises(ValueError):equal_recipe_logits(a,a[:1])
    with pytest.raises(ValueError):equal_recipe_logits(torch.tensor([float('nan')]),torch.ones(1))
