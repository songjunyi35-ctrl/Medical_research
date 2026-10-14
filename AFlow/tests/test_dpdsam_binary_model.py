from pathlib import Path

import pytest
import torch
from torch import nn

from sammed3d import dpdsam_binary_model as binary_model


class TinyImageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv3d(1, 3, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.patch_embed.proj(images)


class TinyPromptEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_dim = 3
        self.no_mask_embed = nn.Embedding(1, 3)
        self.register_buffer("pe", torch.zeros(1, 3, 2, 2, 2))

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe


class TinyMaskDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.25))
        self.iou_prediction_head = nn.Linear(3, 1)
        self.last_sparse: torch.Tensor | None = None
        self.last_dense: torch.Tensor | None = None

    def forward(self, **kwargs):
        assert kwargs["multimask_output"] is False
        self.last_sparse = kwargs["sparse_prompt_embeddings"]
        self.last_dense = kwargs["dense_prompt_embeddings"]
        batch = kwargs["image_embeddings"].shape[0]
        logits = self.bias.expand(batch, 1, 1, 1, 1)
        return logits, logits.new_zeros((batch, 1))


class TinySAM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = TinyImageEncoder()
        self.prompt_encoder = TinyPromptEncoder()
        self.mask_decoder = TinyMaskDecoder()


def factory(model_type: str) -> TinySAM:
    assert model_type == "tiny"
    return TinySAM()


def test_b0_forward_is_single_channel_label_free_and_batch_safe(tmp_path: Path) -> None:
    checkpoint = tmp_path / "base.pt"
    torch.save({"model_state_dict": TinySAM().state_dict()}, checkpoint)
    model, metadata = binary_model.load_untrained_b0(
        checkpoint, factory, device="cpu", model_type="tiny", embedding_dim=3
    )
    output = model(torch.randn(2, 1, 2, 2, 2))
    assert output.logits.shape == (2, 1, 2, 2, 2)
    assert output.probabilities.shape == output.logits.shape
    assert output.iou_predictions.shape == (2, 1)
    assert model.sam_model.mask_decoder.last_sparse.shape == (2, 1, 3)
    assert metadata["automatic_inference_reads_gt"] is False
    assert metadata["target"]["foreground_raw_labels"] == [1, 2, 3]
    with pytest.raises(ValueError, match=r"\[B,1"):
        model(torch.randn(2, 4, 2, 2, 2))


def test_b0_trainable_contract_freezes_image_and_prompt_encoders(tmp_path: Path) -> None:
    checkpoint = tmp_path / "base.pt"
    torch.save({"model_state_dict": TinySAM().state_dict()}, checkpoint)
    model, _ = binary_model.load_untrained_b0(
        checkpoint, factory, device="cpu", model_type="tiny", embedding_dim=3
    )
    trainable = binary_model.configure_b0_trainable(model)
    assert trainable
    assert not any(p.requires_grad for p in model.sam_model.image_encoder.parameters())
    assert not any(p.requires_grad for p in model.sam_model.prompt_encoder.parameters())
    assert model.sam_model.mask_decoder.bias.requires_grad
    assert not any(
        p.requires_grad for p in model.sam_model.mask_decoder.iou_prediction_head.parameters()
    )
    assert all(p.requires_grad for p in model.sam_model.target_embed.parameters())


def test_b0_target_vocabulary_is_frozen() -> None:
    config = binary_model.make_b0_adapter_config(model_type="tiny", embedding_dim=3)
    assert config["target_token"]["vocab"] == ["residual_tumor_123"]
    assert config["encoder_qv_lora"]["enabled"] is False


def test_b1_lora_parameters_are_trainable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "base.pt"
    model = TinySAM()
    model.image_encoder.blocks = nn.ModuleList([nn.Module() for _ in range(12)])
    for block in model.image_encoder.blocks:
        block.attn = nn.Module()
        block.attn.qkv = nn.Linear(3, 9)
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)

    def lora_factory(model_type: str) -> TinySAM:
        value = TinySAM()
        value.image_encoder.blocks = nn.ModuleList([nn.Module() for _ in range(12)])
        for block in value.image_encoder.blocks:
            block.attn = nn.Module()
            block.attn.qkv = nn.Linear(3, 9)
        return value

    adapted, metadata = binary_model.load_untrained_b0(
        checkpoint,
        lora_factory,
        device="cpu",
        model_type="tiny",
        embedding_dim=3,
        lora_rank=1,
        lora_alpha=2,
    )
    binary_model.configure_b0_trainable(adapted)

    lora = list(binary_model.peft.iter_qv_lora_modules(adapted.sam_model))
    assert len(lora) == 12
    assert all(parameter.requires_grad for parameter in lora[0].lora_parameters())
    assert metadata["arm"] == "B1_static_prompt_lora_r1"


def _load_tiny_d1(
    tmp_path: Path, *, centroids: torch.Tensor | None = None
) -> tuple[binary_model.DynamicDomainPromptBinarySAM, dict]:
    checkpoint = tmp_path / "d1_base.pt"
    torch.save({"model_state_dict": TinySAM().state_dict()}, checkpoint)
    return binary_model.load_untrained_d1(
        checkpoint,
        factory,
        device="cpu",
        model_type="tiny",
        centroids=centroids,
    )


def _load_tiny_a2(
    tmp_path: Path,
) -> tuple[binary_model.SpatialAutomaticPromptBinarySAM, dict]:
    checkpoint = tmp_path / "a2_base.pt"
    torch.save({"model_state_dict": TinySAM().state_dict()}, checkpoint)
    return binary_model.load_untrained_a2(
        checkpoint,
        factory,
        device="cpu",
        model_type="tiny",
        embedding_dim=3,
    )


def test_a2_zero_initialization_is_exact_no_mask_dense_prompt(tmp_path: Path) -> None:
    model, metadata = _load_tiny_a2(tmp_path)
    images = torch.randn(2, 1, 2, 2, 2)
    output = model(images)

    assert torch.count_nonzero(output.dense_prompt_residual) == 0
    expected = model.sam_model.prompt_encoder.no_mask_embed.weight.reshape(
        1, 3, 1, 1, 1
    ).expand(2, 3, 2, 2, 2)
    assert torch.equal(output.dense_prompt_embeddings, expected)
    assert torch.equal(model.sam_model.mask_decoder.last_dense, expected)
    assert metadata["dense_prompt"]["output_initialization"].startswith(
        "pointwise_zero"
    )
    assert metadata["scientific_status"].endswith("not_dpdsam_paper")


def test_a2_dense_prompt_is_spatial_and_image_conditioned(tmp_path: Path) -> None:
    model, _ = _load_tiny_a2(tmp_path)
    with torch.no_grad():
        projection = model.sam_model.image_encoder.patch_embed.proj
        projection.weight.fill_(1.0)
        projection.bias.zero_()
        generator = model.spatial_prompt_generator
        generator.depthwise.weight.zero_()
        generator.depthwise.bias.zero_()
        generator.depthwise.weight[:, 0, 1, 1, 1] = 1.0
        generator.pointwise.weight.zero_()
        generator.pointwise.bias.zero_()
        for channel in range(3):
            generator.pointwise.weight[channel, channel, 0, 0, 0] = 1.0

    images = torch.stack(
        (torch.zeros(1, 2, 2, 2), torch.ones(1, 2, 2, 2)), dim=0
    )
    output = model(images)

    assert output.dense_prompt_residual.shape == (2, 3, 2, 2, 2)
    assert not torch.equal(
        output.dense_prompt_residual[0], output.dense_prompt_residual[1]
    )
    no_mask = model.sam_model.prompt_encoder.no_mask_embed.weight.reshape(
        1, 3, 1, 1, 1
    )
    assert torch.allclose(
        output.dense_prompt_embeddings,
        no_mask + output.dense_prompt_residual,
    )
    assert torch.equal(
        model.sam_model.mask_decoder.last_dense, output.dense_prompt_embeddings
    )
    assert model.sam_model.mask_decoder.last_sparse.shape == (2, 1, 3)
    expected_sparse = model.sam_model.target_embed.weight.reshape(1, 1, 3).expand(
        2, 1, 3
    )
    assert torch.equal(model.sam_model.mask_decoder.last_sparse, expected_sparse)


def test_a2_trainable_contract_freezes_sam_bases(tmp_path: Path) -> None:
    model, _ = _load_tiny_a2(tmp_path)
    trainable = binary_model.configure_a2_trainable(model)
    model.train()

    assert trainable
    assert not model.sam_model.image_encoder.training
    assert not model.sam_model.prompt_encoder.training
    assert not any(p.requires_grad for p in model.sam_model.image_encoder.parameters())
    assert not any(p.requires_grad for p in model.sam_model.prompt_encoder.parameters())
    assert all(p.requires_grad for p in model.spatial_prompt_generator.parameters())
    assert all(p.requires_grad for p in model.sam_model.target_embed.parameters())
    assert model.sam_model.mask_decoder.bias.requires_grad
    assert not any(
        p.requires_grad for p in model.sam_model.mask_decoder.iou_prediction_head.parameters()
    )


def test_a2_optional_lora_only_unfreezes_encoder_adapters(tmp_path: Path) -> None:
    def lora_tiny() -> TinySAM:
        value = TinySAM()
        value.image_encoder.blocks = nn.ModuleList([nn.Module() for _ in range(12)])
        for block in value.image_encoder.blocks:
            block.attn = nn.Module()
            block.attn.qkv = nn.Linear(3, 9)
        return value

    checkpoint = tmp_path / "a2_lora_base.pt"
    torch.save({"model_state_dict": lora_tiny().state_dict()}, checkpoint)
    model, metadata = binary_model.load_untrained_a2(
        checkpoint,
        lambda _: lora_tiny(),
        device="cpu",
        model_type="tiny",
        embedding_dim=3,
        lora_rank=1,
        lora_alpha=2,
    )
    binary_model.configure_a2_trainable(model)

    adapters = list(binary_model.peft.iter_qv_lora_modules(model.sam_model))
    lora_parameter_ids = {
        id(parameter)
        for adapter in adapters
        for parameter in adapter.lora_parameters()
    }
    assert len(adapters) == 12
    assert metadata["image_encoder"]["optional_qv_lora"] is True
    assert all(
        parameter.requires_grad
        for adapter in adapters
        for parameter in adapter.lora_parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.sam_model.image_encoder.parameters()
        if id(parameter) not in lora_parameter_ids
    )


def test_a2_checkpoint_round_trip_restores_spatial_prompt(tmp_path: Path) -> None:
    model, metadata = _load_tiny_a2(tmp_path)
    with torch.no_grad():
        model.spatial_prompt_generator.pointwise.weight.normal_(std=0.1)
        model.spatial_prompt_generator.pointwise.bias.fill_(0.2)
    model.eval()
    images = torch.randn(2, 1, 2, 2, 2)
    expected = model(images)

    checkpoint = tmp_path / "a2_checkpoint.pt"
    torch.save(
        binary_model.checkpoint_payload(
            model, adapter_config=metadata["adapter_config"], extra={"epoch": 3}
        ),
        checkpoint,
    )
    restored, restored_metadata = binary_model.load_a2_checkpoint(
        checkpoint, factory, device="cpu"
    )
    actual = restored(images)

    assert restored_metadata["schema"] == binary_model.A2_SCHEMA
    assert restored_metadata["adapter_config"] == metadata["adapter_config"]
    assert torch.allclose(
        actual.dense_prompt_residual, expected.dense_prompt_residual
    )
    assert torch.allclose(
        actual.dense_prompt_embeddings, expected.dense_prompt_embeddings
    )
    assert torch.allclose(actual.logits, expected.logits)
    assert torch.equal(
        restored.sam_model.target_embed.weight,
        model.sam_model.target_embed.weight,
    )


def test_rank_one_domain_prompt_formula_is_exact() -> None:
    generator = binary_model.RankOneDomainPromptGenerator(2, 2, 3)
    with torch.no_grad():
        generator.shared_prompt.copy_(
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        )
        generator.domain_u.copy_(torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]))
        generator.domain_v.copy_(
            torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
        )

    bases = generator.domain_bases()
    expected_zero = generator.shared_prompt * (
        generator.domain_u[0] @ generator.domain_v[0].transpose(0, 1)
    )
    expected_one = generator.shared_prompt * (
        generator.domain_u[1] @ generator.domain_v[1].transpose(0, 1)
    )
    assert torch.equal(bases[0], expected_zero)
    assert torch.equal(bases[1], expected_one)
    mixed = generator(torch.tensor([[0.25, 0.75]]))
    assert torch.allclose(mixed[0], 0.25 * expected_zero + 0.75 * expected_one)


def test_domain_registry_requires_one_finite_external_fit() -> None:
    registry = binary_model.LatentDomainRegistry(4, 3)
    with pytest.raises(RuntimeError, match="not been fitted"):
        registry.pseudo_domain_ids(torch.zeros(1, 3))
    with pytest.raises(ValueError, match="shape"):
        registry.set_centroids(torch.zeros(3, 3))

    centroids = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
    )
    registry.set_centroids(centroids)
    ids = registry.pseudo_domain_ids(
        torch.tensor([[1.9, 0.0, 0.0], [0.0, 0.1, 1.8]])
    )
    assert ids.tolist() == [1, 3]
    with pytest.raises(RuntimeError, match="already fitted"):
        registry.set_centroids(centroids)


def test_d1_forward_uses_gap_conditioned_dynamic_prefix(tmp_path: Path) -> None:
    centroids = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    )
    model, metadata = _load_tiny_d1(tmp_path, centroids=centroids)
    with torch.no_grad():
        model.sam_model.image_encoder.patch_embed.proj.weight.fill_(1.0)
        model.sam_model.image_encoder.patch_embed.proj.bias.zero_()
        model.domain_adapter.fc1.weight.copy_(torch.eye(3))
        model.domain_adapter.fc1.bias.zero_()
        model.domain_adapter.fc2.weight.copy_(
            torch.tensor(
                [[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            )
        )
        model.domain_adapter.fc2.bias.zero_()
        model.prompt_generator.shared_prompt.fill_(1.0)
        model.prompt_generator.domain_u.copy_(
            torch.tensor([[[1.0]], [[2.0]], [[3.0]], [[4.0]]])
        )
        model.prompt_generator.domain_v.fill_(1.0)

    images = torch.stack(
        (torch.zeros(1, 2, 2, 2), torch.ones(1, 2, 2, 2)), dim=0
    )
    output = model(images)

    assert output.global_descriptor.shape == (2, 3)
    assert output.domain_affinities.shape == (2, 4)
    assert torch.allclose(output.domain_affinities.sum(dim=1), torch.ones(2))
    assert output.prompt_prefix.shape == (2, 1, 3)
    assert not torch.allclose(output.domain_affinities[0], output.domain_affinities[1])
    assert not torch.allclose(output.prompt_prefix[0], output.prompt_prefix[1])
    assert model.sam_model.mask_decoder.last_sparse.shape == (2, 1, 3)
    assert torch.allclose(model.sam_model.mask_decoder.last_sparse, output.prompt_prefix)
    assert metadata["cls_surrogate"]["type"] == "global_average_pooling"
    assert metadata["latent_domains"]["count"] == 4
    assert metadata["prompt"]["rank"] == 1
    assert metadata["prompt"]["tokens"] == 1
    assert metadata["reproduction_status"].endswith("not_bit_exact")


def test_d1_trainable_contract_keeps_sam_encoders_frozen(tmp_path: Path) -> None:
    model, _ = _load_tiny_d1(tmp_path, centroids=torch.zeros(4, 3))
    trainable = binary_model.configure_d1_trainable(model)
    model.train()

    assert trainable
    assert not model.sam_model.image_encoder.training
    assert not model.sam_model.prompt_encoder.training
    assert not any(p.requires_grad for p in model.sam_model.image_encoder.parameters())
    assert not any(p.requires_grad for p in model.sam_model.prompt_encoder.parameters())
    assert all(p.requires_grad for p in model.domain_adapter.parameters())
    assert all(p.requires_grad for p in model.prompt_generator.parameters())
    assert model.sam_model.mask_decoder.bias.requires_grad
    assert not any(
        p.requires_grad for p in model.sam_model.mask_decoder.iou_prediction_head.parameters()
    )


def test_d1_checkpoint_round_trip_restores_dynamic_state(tmp_path: Path) -> None:
    centroids = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    )
    model, metadata = _load_tiny_d1(tmp_path, centroids=centroids)
    model.eval()
    images = torch.randn(2, 1, 2, 2, 2)
    expected = model(images)

    checkpoint = tmp_path / "d1_checkpoint.pt"
    torch.save(
        binary_model.checkpoint_payload(
            model, adapter_config=metadata["adapter_config"], extra={"epoch": 7}
        ),
        checkpoint,
    )
    restored, restored_metadata = binary_model.load_d1_checkpoint(
        checkpoint, factory, device="cpu"
    )
    actual = restored(images)

    assert restored_metadata["schema"] == binary_model.D1_SCHEMA
    assert restored_metadata["adapter_config"] == metadata["adapter_config"]
    assert bool(restored.domain_registry.fitted)
    assert torch.equal(restored.domain_registry.centroids, centroids)
    assert torch.allclose(actual.domain_logits, expected.domain_logits)
    assert torch.allclose(actual.domain_affinities, expected.domain_affinities)
    assert torch.allclose(actual.prompt_prefix, expected.prompt_prefix)
    assert torch.allclose(actual.logits, expected.logits)


def test_d1_forward_fails_closed_without_centroids(tmp_path: Path) -> None:
    model, metadata = _load_tiny_d1(tmp_path)
    assert metadata["latent_domains"]["centroids_fitted_externally"] is False
    with pytest.raises(RuntimeError, match="not been fitted"):
        model(torch.zeros(1, 1, 2, 2, 2))
