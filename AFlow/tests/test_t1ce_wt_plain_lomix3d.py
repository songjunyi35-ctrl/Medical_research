from __future__ import annotations

from collections import OrderedDict

from monai.losses import DiceCELoss
import pytest
import torch

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from sammed3d.t1ce_wt_lomix3d import (
    MAP_NAMES,
    MUTANT_MAP_NAMES,
    ORIGINAL_MAP_NAMES,
    SUBSETS,
    LoMix3DTrainingModel,
    SubsetAttentionWeightedFusion,
)


def _plain(*, center_aware: bool = False) -> AutomaticPromptLocator:
    torch.manual_seed(11)
    return AutomaticPromptLocator(
        center_aware=center_aware, init_filters=8, dropout=0.1
    )


def _small_model() -> LoMix3DTrainingModel:
    return LoMix3DTrainingModel.from_plain_model(_plain())


def test_real_decoder_stages_are_32_64_128_and_all_19_maps_are_raw_shape() -> None:
    model = _small_model().eval()
    image = torch.randn((1, 1, 128, 128, 128))
    with torch.inference_mode():
        output = model(image)
    assert [tuple(value.shape) for value in output.native_original_logits] == [
        (1, 1, 32, 32, 32),
        (1, 1, 64, 64, 64),
        (1, 1, 128, 128, 128),
    ]
    assert output.final_logits is output.original_logits[-1]
    assert tuple(output.mutant_logits) == MUTANT_MAP_NAMES
    assert tuple(output.all_logits()) == MAP_NAMES
    assert len(output.all_logits()) == 19
    assert all(value.shape == output.final_logits.shape for value in output.all_logits().values())


def test_every_subset_has_independent_registered_concat_and_awf_modules() -> None:
    model = _small_model()
    subset_names = tuple(
        "_".join(("s32", "s64", "s128")[index] for index in subset)
        for subset in SUBSETS
    )
    assert tuple(model.concat_fusions) == subset_names
    assert tuple(model.attention_fusions) == subset_names
    assert len({id(module) for module in model.concat_fusions.values()}) == 4
    assert len({id(module) for module in model.attention_fusions.values()}) == 4
    assert tuple(model.loss_alphas) == MAP_NAMES
    assert MAP_NAMES[:3] == ("original_32", "original_64", "original_128")
    assert MAP_NAMES[3:7] == (
        "add_s32_s64",
        "multiply_s32_s64",
        "concat_s32_s64",
        "awf_s32_s64",
    )
    for alpha in model.loss_alphas.values():
        assert alpha.item() == 0.0
        assert torch.nn.functional.softplus(alpha).item() == pytest.approx(
            0.6931471805599453
        )


def test_all_training_only_parameters_receive_finite_nonzero_gradient() -> None:
    torch.manual_seed(19)
    model = _small_model().train()
    image = torch.randn((1, 1, 16, 16, 16))
    label = (torch.rand((1, 1, 16, 16, 16)) > 0.75).float()
    output = model(image)
    loss_function = DiceCELoss(
        sigmoid=True,
        squared_pred=False,
        lambda_dice=1.0,
        lambda_ce=0.1,
    )
    loss = model.lomix_loss(output, label, loss_function)
    assert tuple(loss.per_map) == MAP_NAMES
    assert tuple(loss.effective_weights) == MAP_NAMES
    assert torch.isfinite(loss.total)
    assert sum(float(value.detach()) for value in loss.effective_weights.values()) == pytest.approx(
        19.0 * 0.6931471805599453
    )
    loss.total.backward()

    named = list(model.training_only_named_parameters())
    assert named
    assert [parameter for _, parameter in named] == list(
        model.training_only_parameters()
    )
    for name, parameter in named:
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum().item() > 0.0, name
    assert all(not name.startswith("network.") for name, _ in named)


def test_wrapper_final_is_exactly_plain_and_deploy_export_is_strict() -> None:
    plain = _plain().eval()
    model = LoMix3DTrainingModel.from_plain_model(plain).eval()
    image = torch.randn((1, 1, 16, 16, 16))
    with torch.inference_mode():
        plain_logits = plain(image).segmentation_logits
        wrapper_logits = model(image).final_logits
    assert torch.equal(wrapper_logits, plain_logits)

    deploy_state = model.deploy_state_dict(cpu=True)
    assert isinstance(deploy_state, OrderedDict)
    assert tuple(deploy_state) == tuple(plain.state_dict())
    assert not any(
        token in name for name in deploy_state
        for token in ("auxiliary", "concat", "attention", "alpha")
    )
    deploy = model.to_deploy_model().eval()
    with torch.inference_mode():
        deploy_logits = deploy(image).segmentation_logits
    assert torch.equal(wrapper_logits, deploy_logits)
    rng_before = torch.get_rng_state().clone()
    assert model.deploy_max_abs_difference(image) == 0.0
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert model.training is False


def test_training_and_deploy_state_roundtrips_are_exact() -> None:
    torch.manual_seed(23)
    model = _small_model().eval()
    image = torch.randn((1, 1, 16, 16, 16))
    clone = LoMix3DTrainingModel(init_filters=8, dropout=0.1).eval()
    clone.load_state_dict(model.state_dict(), strict=True)
    with torch.inference_mode():
        left = model(image)
        right = clone(image)
    assert torch.equal(left.final_logits, right.final_logits)
    assert all(
        torch.equal(left.all_logits()[name], right.all_logits()[name])
        for name in MAP_NAMES
    )

    deploy = _plain().eval()
    deploy.load_state_dict(model.deploy_state_dict(), strict=True)
    with torch.inference_mode():
        assert torch.equal(
            model(image).final_logits, deploy(image).segmentation_logits
        )


def test_plain_checkpoint_mapping_and_path_load_strictly(tmp_path) -> None:
    plain = _plain().eval()
    checkpoint = {
        "architecture": {
            "class": "AutomaticPromptLocator",
            "center_aware": False,
            "init_filters": 8,
            "dropout": 0.1,
        },
        "model_state_dict": plain.state_dict(),
    }
    mapped = LoMix3DTrainingModel.from_plain_checkpoint(checkpoint).eval()
    path = tmp_path / "historical_plain.pth"
    torch.save(checkpoint, path)
    loaded = LoMix3DTrainingModel.from_plain_checkpoint(path).eval()
    image = torch.randn((1, 1, 16, 16, 16))
    with torch.inference_mode():
        expected = plain(image).segmentation_logits
        assert torch.equal(mapped(image).final_logits, expected)
        assert torch.equal(loaded(image).final_logits, expected)

    broken = dict(checkpoint)
    broken_state = OrderedDict(checkpoint["model_state_dict"])
    broken_state.pop(next(iter(broken_state)))
    broken["model_state_dict"] = broken_state
    with pytest.raises(RuntimeError):
        LoMix3DTrainingModel.from_plain_checkpoint(broken)


@pytest.mark.parametrize(
    "image,exception",
    [
        (torch.randn((1, 2, 16, 16, 16)), ValueError),
        (torch.randn((1, 1, 15, 16, 16)), ValueError),
        (torch.ones((1, 1, 16, 16, 16), dtype=torch.int64), TypeError),
    ],
)
def test_invalid_images_are_rejected(image: torch.Tensor, exception: type[Exception]) -> None:
    with pytest.raises(exception):
        _small_model()(image)


def test_invalid_plain_and_loss_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="center_aware"):
        LoMix3DTrainingModel.from_plain_model(_plain(center_aware=True))
    with pytest.raises(ValueError, match="architecture"):
        LoMix3DTrainingModel.from_plain_checkpoint({"model_state_dict": {}})

    model = _small_model()
    output = model(torch.randn((1, 1, 16, 16, 16)))
    loss_function = DiceCELoss(sigmoid=True, lambda_dice=1.0, lambda_ce=0.1)
    with pytest.raises(ValueError, match="label shape"):
        model.lomix_loss(
            output, torch.zeros((1, 1, 8, 8, 8)), loss_function
        )
    with pytest.raises(TypeError, match="floating"):
        model.lomix_loss(
            output, torch.zeros_like(output.final_logits, dtype=torch.int64), loss_function
        )


def test_attention_fusion_rejects_wrong_count_and_shape() -> None:
    fusion = SubsetAttentionWeightedFusion(2)
    value = torch.randn((1, 1, 8, 8, 8))
    with pytest.raises(ValueError, match="count"):
        fusion((value,))
    with pytest.raises(ValueError, match="identical"):
        fusion((value, torch.randn((1, 1, 4, 8, 8))))


def test_frozen_add_multiply_and_attention_formulas() -> None:
    first = torch.linspace(-2.0, 2.0, 64).reshape(1, 1, 4, 4, 4)
    second = torch.flip(first, dims=(-1,)) + 0.25
    fusion = SubsetAttentionWeightedFusion(2)
    with torch.no_grad():
        fusion.attention_logits.weight.zero_()
        fusion.attention_logits.bias.zero_()
    assert torch.equal(
        torch.stack((first, second), dim=0).sum(dim=0), first + second
    )
    assert torch.equal(
        torch.stack((first, second), dim=0).prod(dim=0), first * second
    )
    assert torch.allclose(fusion((first, second)), (first + second) / 2.0)
