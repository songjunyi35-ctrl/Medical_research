"""Geometry and failure-mode tests; all inputs are synthetic."""
import torch
from sammed3d.race_seg import RaceSeg, LocalReference, feature_reference_grid, spatial_augment, typed_contrast


def reflection(center):
    m = torch.eye(4)
    m[0,0] = -1
    m[0,3] = 2*center
    return m


def test_reference_feature_centers_and_translation():
    matrix = reflection(12)[None]
    grid, valid = feature_reference_grid(matrix, (32,32,32), (4,4,4))
    # feature index zero reflects to input24 -> feature3, not 31 -> 3.875.
    assert grid[0,0,0,0,2] == 1
    assert grid[0,3,0,0,2] == -1
    assert valid.all()
    x = torch.zeros(1,32,32,32)
    y = x.clone(); y[0,4,5,6] = 1
    _, yy, _, _, m = spatial_augment(x,y,y.clone(),torch.ones_like(x),matrix[0],flips=(True,False,False),shifts=(2,-1,0))
    assert yy[0,29,4,6] == 1
    old = torch.tensor([4.,5.,6.,1.])
    new = torch.tensor([29.,4.,6.,1.])
    assert torch.allclose(m@new, torch.tensor([13.,4.,6.,1.]))
    assert torch.allclose(matrix[0]@old,torch.tensor([20.,5.,6.,1.]))


def test_all_invalid_reference_is_finite_zero():
    module=LocalReference(8,4)
    features=torch.randn(1,8,4,4,4,requires_grad=True)
    query, ref, entropy, available=module(features,reflection(12)[None],torch.zeros(1,1,32,32,32))
    assert torch.equal(query,module.value(features).float())
    assert not available.any() and torch.count_nonzero(ref)==0
    assert torch.isfinite(entropy).all()
    (query.sum()+ref.sum()).backward()
    assert torch.isfinite(features.grad).all()


def test_initialization_matches_backbone_and_residual_can_wake():
    torch.manual_seed(4)
    model=RaceSeg(width=8,dropout=0,dim=4).eval()
    x=torch.randn(1,1,32,32,32)
    result=model(x,reflection(12)[None],torch.ones_like(x),corruption=True)
    with torch.no_grad():
        expected=model.backbone(x).segmentation_logits
    assert torch.equal(result.segmentation_logits,expected)
    assert torch.equal(result.bad_logits,expected)
    result.segmentation_logits.mean().backward()
    assert model.strength.grad is not None and model.strength.grad.abs()>0


def test_no_reference_matches_base_after_nonzero_strength():
    model=RaceSeg(width=8,dropout=0,dim=4).eval()
    with torch.no_grad(): model.strength.fill_(.8)
    x=torch.randn(1,1,32,32,32)
    output=model(x,reflection(12)[None],torch.zeros_like(x))
    assert torch.equal(output.segmentation_logits,output.base_logits)


def test_missing_classes_and_prototype_gradients():
    emb=torch.randn(2,4,4,4,4,requires_grad=True)
    labels=torch.ones(2,1,32,32,32)
    proto=torch.randn(3,2,4,requires_grad=True)
    loss=typed_contrast(emb,labels,proto,torch.ones_like(labels),8)
    loss.backward()
    assert loss.isfinite() and proto.grad.isfinite().all() and proto.grad.abs().sum()>0
    empty=typed_contrast(emb,labels,proto,torch.zeros_like(labels),8)
    assert empty==0 and empty.isfinite()
    proto.grad=None
    empty.backward()
    assert proto.grad is not None and torch.count_nonzero(proto.grad)==0


def test_auxiliary_sampling_does_not_change_backbone_rng():
    torch.manual_seed(8)
    embedding=torch.randn(1,4,4,4,4)
    proto=torch.randn(3,2,4)
    labels=torch.zeros(1,1,32,32,32)
    before=torch.random.get_rng_state().clone()
    typed_contrast(embedding,labels,proto,torch.ones_like(labels),max_per_class=4)
    assert torch.equal(before,torch.random.get_rng_state())


def test_matched_backbone_initialization_and_auxiliary_gradients():
    torch.manual_seed(19)
    control=RaceSeg(width=8,dropout=0,dim=4,enabled=False)
    torch.manual_seed(19)
    candidate=RaceSeg(width=8,dropout=0,dim=4,enabled=True)
    assert all(torch.equal(v,candidate.backbone.state_dict()[k]) for k,v in control.backbone.state_dict().items())
    x=torch.randn(1,1,32,32,32)
    output=candidate(x,reflection(12)[None],torch.ones_like(x),corruption=True)
    labels=torch.zeros_like(x,dtype=torch.uint8);labels[:,:,8:24,8:24,8:24]=1
    contrast=typed_contrast(output.embedding,labels,candidate.prototypes,torch.ones_like(x),8)
    loss=output.segmentation_logits.square().mean()+.05*contrast+.02*output.bad_gate.mean()
    loss.backward()
    for module in (candidate.reference.query,candidate.project,candidate.gate_head):
        assert module.weight.grad is not None and torch.isfinite(module.weight.grad).all()
        assert module.weight.grad.abs().sum()>0


def test_typed_contrast_weights_cases_equally_with_different_present_classes():
    torch.manual_seed(22)
    embedding=torch.randn(2,4,4,4,4)
    prototypes=torch.randn(3,2,4)
    labels=torch.zeros(2,1,32,32,32,dtype=torch.uint8)
    labels[1,:,8:16]=1
    labels[1,:,16:24]=2
    valid=torch.ones_like(labels,dtype=torch.float32)
    joint=typed_contrast(embedding,labels,prototypes,valid,128)
    separate=torch.stack([typed_contrast(embedding[i:i+1],labels[i:i+1],prototypes,valid[i:i+1],128) for i in range(2)]).mean()
    assert torch.allclose(joint,separate,atol=1e-6,rtol=0)
