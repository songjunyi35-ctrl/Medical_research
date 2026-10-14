import sys
from pathlib import Path

import torch
from torch import nn

from sammed3d import peft
from sammed3d.paths import SAMMED3D_ROOT
from scripts.train_sammed3d_brats import configure_trainable


class TinyAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)


class TinyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = TinyAttention(dim)


class TinyImageEncoder(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock(dim) for _ in range(depth)])


class TinyPromptEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.embed_dim = dim
        self.projection = nn.Linear(dim, dim)


class TinySam(nn.Module):
    def __init__(self, dim=8, depth=2, mask_decoder=None):
        super().__init__()
        self.image_encoder = TinyImageEncoder(dim, depth)
        self.prompt_encoder = TinyPromptEncoder(dim)
        self.mask_decoder = mask_decoder or nn.Linear(dim, dim)


def tiny_config(*, target_token=True, lora_rank=2, dim=8):
    return peft.make_adapter_config(
        target_token=target_token,
        lora_rank=lora_rank,
        lora_alpha=4,
        model_type="tiny",
        lora_blocks=(0, 1),
        target_vocab=("target_a", "target_b"),
        target_embedding_dim=dim,
    )


def test_default_adapter_config_is_vit_b_ori_rank4_on_all_12_blocks():
    config = peft.make_adapter_config()

    assert config["model_type"] == "vit_b_ori"
    assert config["encoder_qv_lora"]["rank"] == 4
    assert config["encoder_qv_lora"]["alpha"] == 8.0
    assert config["encoder_qv_lora"]["blocks"] == list(range(12))


def test_qv_lora_initial_output_and_base_state_keys_are_unchanged():
    torch.manual_seed(7)
    model = TinySam()
    x = torch.randn(2, 3, 8)
    original = model.image_encoder.blocks[0].attn.qkv
    expected = original(x)

    adapters = peft.inject_qv_lora(model, rank=2, alpha=4, block_indexes=(0,))
    actual = model.image_encoder.blocks[0].attn.qkv(x)

    assert torch.equal(actual, expected)
    assert adapters[0].weight is original.weight
    state_keys = set(model.state_dict())
    assert "image_encoder.blocks.0.attn.qkv.weight" in state_keys
    assert "image_encoder.blocks.0.attn.qkv.bias" in state_keys
    assert not any("qkv.base" in key for key in state_keys)


def test_qv_lora_only_trains_adapters_and_never_changes_k_branch():
    torch.manual_seed(11)
    model = TinySam()
    adapter = peft.inject_qv_lora(model, rank=2, alpha=4, block_indexes=(0,))[0]
    configure_trainable(model, "decoder_only")

    assert not adapter.weight.requires_grad
    assert adapter.bias is not None and not adapter.bias.requires_grad
    assert all(parameter.requires_grad for parameter in adapter.lora_parameters())
    assert any(parameter.requires_grad for parameter in model.image_encoder.parameters())

    with torch.no_grad():
        adapter.lora_B_q.normal_()
        adapter.lora_B_v.normal_()
    delta_q, delta_k, delta_v = adapter.delta_weight().chunk(3, dim=0)
    assert torch.count_nonzero(delta_q) > 0
    assert torch.count_nonzero(delta_k) == 0
    assert torch.count_nonzero(delta_v) > 0

    adapter(torch.randn(2, 8)).sum().backward()
    assert adapter.weight.grad is None
    assert adapter.bias.grad is None


def test_target_token_shape_is_accepted_by_real_mask_decoder():
    sys.path.insert(0, str(SAMMED3D_ROOT))
    from segment_anything.modeling.mask_decoder3D import MaskDecoder3D

    decoder = MaskDecoder3D(
        transformer_dim=32,
        num_multimask_outputs=3,
        iou_head_depth=2,
        iou_head_hidden_dim=16,
    )
    model = TinySam(dim=32, depth=1, mask_decoder=decoder)
    target_embed = peft.inject_target_embedding(
        model,
        target_vocab=("target_a", "target_b"),
        embedding_dim=32,
    )
    sparse = torch.randn(2, 2, 32)
    target_ids = target_embed.ids(("target_a", "target_b"))
    conditioned = peft.append_target_token(model, sparse, target_ids)

    assert conditioned.shape == (2, 3, 32)
    masks, iou = decoder(
        image_embeddings=torch.randn(2, 32, 2, 2, 2),
        image_pe=torch.randn(2, 32, 2, 2, 2),
        sparse_prompt_embeddings=conditioned,
        dense_prompt_embeddings=torch.randn(2, 32, 2, 2, 2),
        multimask_output=False,
    )
    assert masks.shape == (2, 1, 8, 8, 8)
    assert iou.shape == (2, 1)


def test_factorial_components_use_paired_initialization_and_preserve_rng():
    target_only = TinySam()
    lora_only = TinySam()
    combined = TinySam()
    direct_target = TinySam()

    torch.manual_seed(101)
    peft.apply_adapter_config(
        target_only,
        tiny_config(target_token=True, lora_rank=0),
    )
    torch.manual_seed(101)
    peft.apply_adapter_config(
        lora_only,
        tiny_config(target_token=False, lora_rank=2),
    )
    torch.manual_seed(101)
    peft.apply_adapter_config(combined, tiny_config(target_token=True, lora_rank=2))
    torch.manual_seed(101)
    peft.inject_target_embedding(
        direct_target,
        target_vocab=("target_a", "target_b"),
        embedding_dim=8,
    )

    assert torch.equal(target_only.target_embed.weight, combined.target_embed.weight)
    assert torch.equal(target_only.target_embed.weight, direct_target.target_embed.weight)
    for lora_module, combined_module in zip(
        peft.iter_qv_lora_modules(lora_only),
        peft.iter_qv_lora_modules(combined),
    ):
        for parameter, combined_parameter in zip(
            lora_module.lora_parameters(),
            combined_module.lora_parameters(),
        ):
            assert torch.equal(parameter, combined_parameter)

    rng_probe_model = TinySam()
    torch.manual_seed(303)
    expected_next = torch.rand(4)
    torch.manual_seed(303)
    peft.apply_adapter_config(rng_probe_model, tiny_config())
    actual_next = torch.rand(4)
    assert torch.equal(actual_next, expected_next)


def test_legacy_checkpoint_loads_strictly_before_adapter_injection(tmp_path):
    torch.manual_seed(17)
    original = TinySam()
    original_state = {key: value.clone() for key, value in original.state_dict().items()}
    checkpoint = tmp_path / "legacy_raw_state.pth"
    torch.save(original.state_dict(), checkpoint)

    loaded, effective = peft.load_model_with_adapters(
        checkpoint,
        lambda model_type: TinySam() if model_type == "tiny" else None,
        device="cpu",
        requested_config=tiny_config(),
        default_model_type="tiny",
    )

    assert effective == peft.normalize_adapter_config(tiny_config())
    assert isinstance(loaded.target_embed, peft.TargetEmbedding)
    assert len(list(peft.iter_qv_lora_modules(loaded))) == 2
    for key, expected in original_state.items():
        assert torch.equal(loaded.state_dict()[key], expected)


def test_adapter_disabled_checkpoint_can_start_lora_finetuning(tmp_path):
    torch.manual_seed(19)
    original = TinySam()
    disabled = tiny_config(target_token=False, lora_rank=0)
    checkpoint = tmp_path / "decoder_only_checkpoint.pth"
    torch.save(
        {
            "model_state_dict": original.state_dict(),
            "adapter_config": disabled,
        },
        checkpoint,
    )
    requested = tiny_config(target_token=False, lora_rank=2)

    loaded, effective = peft.load_model_with_adapters(
        checkpoint,
        lambda model_type: TinySam() if model_type == "tiny" else None,
        device="cpu",
        requested_config=requested,
        default_model_type="tiny",
    )

    assert effective == peft.normalize_adapter_config(requested)
    assert len(list(peft.iter_qv_lora_modules(loaded))) == 2
    for key, expected in original.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], expected)


def test_adapter_checkpoint_rebuilds_architecture_before_strict_load(tmp_path):
    torch.manual_seed(23)
    config = tiny_config()
    original = TinySam()
    peft.apply_adapter_config(original, config)
    with torch.no_grad():
        next(peft.iter_qv_lora_modules(original)).lora_B_q.normal_()
        original.target_embed.weight.normal_()
    checkpoint = tmp_path / "adapter_checkpoint.pth"
    torch.save(
        {
            "model_state_dict": original.state_dict(),
            "adapter_config": config,
        },
        checkpoint,
    )

    loaded, effective = peft.load_model_with_adapters(
        checkpoint,
        lambda model_type: TinySam() if model_type == "tiny" else None,
        device="cpu",
        requested_config=config,
        default_model_type="tiny",
    )

    assert effective == peft.normalize_adapter_config(config)
    assert loaded.state_dict().keys() == original.state_dict().keys()
    for key, expected in original.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], expected)
