from types import SimpleNamespace
import pytest
import torch
from sammed3d.automatic_fixed_workflow import AXES, WORKFLOWS, online_logits, restored_atom, fuse_atoms


class CountingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        # Deliberately orientation-sensitive, not an equivariant identity toy.
        ramp = torch.arange(x.shape[-1], dtype=x.dtype, device=x.device)
        return SimpleNamespace(segmentation_logits=x * .37 + ramp)


@pytest.mark.parametrize('views', list(WORKFLOWS.values()))
def test_exact_online_subset_and_execution_count(views):
    model = CountingModel()
    x = torch.randn(2, 1, 5, 6, 7, generator=torch.Generator().manual_seed(19))
    before = x.clone()
    atoms = [restored_atom(model, x, i) for i in range(8)]
    model.calls = 0
    actual = online_logits(model, x, views)
    assert model.calls == len(views)
    assert torch.equal(actual, fuse_atoms(atoms, views))
    assert torch.equal(x, before)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize('views', [(), (1,), (0, 0), (1, 0), (0, 8), (0, -1), (0, True), (0, 1.0)])
def test_invalid_views_fail_before_model(views):
    model = CountingModel()
    with pytest.raises(ValueError):
        online_logits(model, torch.zeros(1, 1, 4, 4, 4), views)
    assert model.calls == 0


def test_axes_are_distinct_and_inverse_is_self():
    x = torch.arange(3*4*5).reshape(1, 1, 3, 4, 5)
    assert len(set(AXES)) == 8
    for axes in AXES:
        if axes:
            assert torch.equal(x, torch.flip(torch.flip(x, axes), axes))
