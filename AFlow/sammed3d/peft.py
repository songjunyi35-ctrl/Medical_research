"""Parameter-efficient adapters for SAM-Med3D training.

This module deliberately does not change the upstream SAM-Med3D classes. It
loads legacy checkpoints into the original architecture first, then injects
optional adapters. Checkpoints that already contain adapters carry enough
metadata to rebuild the augmented architecture before strict state loading.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


ADAPTER_CONFIG_VERSION = 1
DEFAULT_MODEL_TYPE = "vit_b_ori"
DEFAULT_LORA_BLOCKS = tuple(range(12))
DEFAULT_LORA_RANK = 4
DEFAULT_LORA_ALPHA = 8.0
DEFAULT_TARGET_EMBED_DIM = 384
TARGET_VOCAB = (
    "tumor_WT",
    "tumor_TC",
    "tumor_ET",
    "label_SNFH",
    "label_RC",
    "label_NETC",
)


class QVLoRALinear(nn.Module):
    """Add independent low-rank Q/V deltas to a fused QKV projection.

    The base ``weight`` and ``bias`` parameters live directly on this module,
    preserving the original ``...attn.qkv.weight`` and ``.bias`` state keys.
    """

    def __init__(self, linear: nn.Linear, rank: int = DEFAULT_LORA_RANK, alpha: float = DEFAULT_LORA_ALPHA):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {alpha}")
        if linear.out_features != 3 * linear.in_features:
            raise ValueError(
                "fused QKV projection must have out_features == 3 * in_features, "
                f"got {linear.in_features} -> {linear.out_features}"
            )

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank

        self.weight = linear.weight
        self.bias = linear.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        parameter_kwargs = {
            "device": self.weight.device,
            "dtype": self.weight.dtype,
        }
        dim = self.in_features
        self.lora_A_q = nn.Parameter(torch.empty(self.rank, dim, **parameter_kwargs))
        self.lora_B_q = nn.Parameter(torch.zeros(dim, self.rank, **parameter_kwargs))
        self.lora_A_v = nn.Parameter(torch.empty(self.rank, dim, **parameter_kwargs))
        self.lora_B_v = nn.Parameter(torch.zeros(dim, self.rank, **parameter_kwargs))
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A_q, a=5**0.5)
        nn.init.kaiming_uniform_(self.lora_A_v, a=5**0.5)
        nn.init.zeros_(self.lora_B_q)
        nn.init.zeros_(self.lora_B_v)

    def lora_parameters(self) -> tuple[nn.Parameter, ...]:
        return (
            self.lora_A_q,
            self.lora_B_q,
            self.lora_A_v,
            self.lora_B_v,
        )

    def delta_weight(self) -> torch.Tensor:
        dim = self.in_features
        zeros = self.weight.new_zeros((dim, dim))
        delta_q = (self.lora_B_q @ self.lora_A_q) * self.scale
        delta_v = (self.lora_B_v @ self.lora_A_v) * self.scale
        return torch.cat((delta_q, zeros, delta_v), dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        delta_q = F.linear(F.linear(x, self.lora_A_q), self.lora_B_q) * self.scale
        delta_v = F.linear(F.linear(x, self.lora_A_v), self.lora_B_v) * self.scale
        return base + torch.cat((delta_q, torch.zeros_like(delta_q), delta_v), dim=-1)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}"
        )


class TargetEmbedding(nn.Embedding):
    """Embedding table with an explicit, checkpointed-via-config target order."""

    def __init__(self, target_vocab: Sequence[str], embedding_dim: int):
        vocab = tuple(target_vocab)
        if not vocab or len(vocab) != len(set(vocab)):
            raise ValueError("target vocab must be non-empty and contain unique names")
        if embedding_dim <= 0:
            raise ValueError("target embedding dimension must be positive")
        self.target_vocab = vocab
        self.target_to_id = {target: index for index, target in enumerate(vocab)}
        super().__init__(len(vocab), embedding_dim)
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def ids(self, targets: Sequence[str], device: torch.device | str | None = None) -> torch.Tensor:
        unknown = sorted(set(targets).difference(self.target_to_id))
        if unknown:
            raise ValueError(f"unknown target names: {', '.join(unknown)}")
        return torch.tensor(
            [self.target_to_id[target] for target in targets],
            dtype=torch.long,
            device=device,
        )


def iter_qv_lora_modules(model: nn.Module):
    for module in model.modules():
        if isinstance(module, QVLoRALinear):
            yield module


def inject_qv_lora(
    model: nn.Module,
    rank: int = DEFAULT_LORA_RANK,
    alpha: float = DEFAULT_LORA_ALPHA,
    block_indexes: Sequence[int] = DEFAULT_LORA_BLOCKS,
) -> list[QVLoRALinear]:
    """Replace selected image-encoder fused QKV projections with Q/V LoRA."""
    blocks = model.image_encoder.blocks
    indexes = tuple(int(index) for index in block_indexes)
    if len(indexes) != len(set(indexes)):
        raise ValueError("LoRA block indexes must be unique")
    invalid = [index for index in indexes if index < 0 or index >= len(blocks)]
    if invalid:
        raise ValueError(f"LoRA block indexes out of range for {len(blocks)} blocks: {invalid}")

    injected = []
    for index in indexes:
        qkv = blocks[index].attn.qkv
        if isinstance(qkv, QVLoRALinear):
            if qkv.rank != rank or qkv.alpha != float(alpha):
                raise ValueError(
                    f"block {index} already has rank={qkv.rank}, alpha={qkv.alpha} LoRA"
                )
            injected.append(qkv)
            continue
        if not isinstance(qkv, nn.Linear):
            raise TypeError(f"block {index} qkv must be nn.Linear, got {type(qkv).__name__}")
        adapter = QVLoRALinear(qkv, rank=rank, alpha=alpha)
        blocks[index].attn.qkv = adapter
        injected.append(adapter)
    return injected


def inject_target_embedding(
    model: nn.Module,
    target_vocab: Sequence[str] = TARGET_VOCAB,
    embedding_dim: int = DEFAULT_TARGET_EMBED_DIM,
) -> TargetEmbedding:
    """Attach one explicit target embedding table to a SAM model instance."""
    prompt_dim = int(model.prompt_encoder.embed_dim)
    if embedding_dim != prompt_dim:
        raise ValueError(
            f"target embedding dim {embedding_dim} does not match prompt dim {prompt_dim}"
        )
    existing = getattr(model, "target_embed", None)
    if existing is not None:
        if not isinstance(existing, TargetEmbedding):
            raise TypeError("model.target_embed exists but is not TargetEmbedding")
        if existing.target_vocab != tuple(target_vocab) or existing.embedding_dim != embedding_dim:
            raise ValueError("existing target embedding does not match requested config")
        return existing

    target_embed = TargetEmbedding(target_vocab, embedding_dim)
    reference = next(model.prompt_encoder.parameters(), None)
    if reference is None:
        reference = next(model.parameters(), None)
    if reference is not None:
        target_embed = target_embed.to(device=reference.device, dtype=reference.dtype)
    model.target_embed = target_embed
    return target_embed


def append_target_token(
    model: nn.Module,
    sparse_embeddings: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    target_embed = getattr(model, "target_embed", None)
    if not isinstance(target_embed, TargetEmbedding):
        raise RuntimeError("target embedding has not been injected")
    if target_ids.ndim != 1 or target_ids.shape[0] != sparse_embeddings.shape[0]:
        raise ValueError(
            "target_ids must have shape [batch] matching sparse embeddings, "
            f"got {tuple(target_ids.shape)} and batch {sparse_embeddings.shape[0]}"
        )
    token = target_embed(target_ids).unsqueeze(1).to(dtype=sparse_embeddings.dtype)
    return torch.cat((sparse_embeddings, token), dim=1)


def target_ids_for_model(
    model: nn.Module,
    targets: Sequence[str],
    device: torch.device | str,
) -> torch.Tensor:
    target_embed = getattr(model, "target_embed", None)
    if not isinstance(target_embed, TargetEmbedding):
        raise RuntimeError("target embedding has not been injected")
    return target_embed.ids(targets, device=device)


def make_adapter_config(
    *,
    target_token: bool = False,
    lora_rank: int = DEFAULT_LORA_RANK,
    lora_alpha: float = DEFAULT_LORA_ALPHA,
    model_type: str = DEFAULT_MODEL_TYPE,
    lora_blocks: Sequence[int] = DEFAULT_LORA_BLOCKS,
    target_vocab: Sequence[str] = TARGET_VOCAB,
    target_embedding_dim: int = DEFAULT_TARGET_EMBED_DIM,
) -> dict[str, Any]:
    if lora_rank < 0:
        raise ValueError(f"LoRA rank must be non-negative, got {lora_rank}")
    if lora_rank > 0 and lora_alpha <= 0:
        raise ValueError(f"LoRA alpha must be positive, got {lora_alpha}")
    if not model_type:
        raise ValueError("model_type must not be empty")
    return {
        "schema_version": ADAPTER_CONFIG_VERSION,
        "model_type": model_type,
        "target_token": {
            "enabled": bool(target_token),
            "vocab": list(target_vocab),
            "embedding_dim": int(target_embedding_dim),
        },
        "encoder_qv_lora": {
            "enabled": lora_rank > 0,
            "rank": int(lora_rank),
            "alpha": float(lora_alpha),
            "blocks": [int(index) for index in lora_blocks],
            "projections": ["q", "v"],
        },
    }


def normalize_adapter_config(config: Mapping[str, Any]) -> dict[str, Any]:
    version = int(config.get("schema_version", ADAPTER_CONFIG_VERSION))
    if version != ADAPTER_CONFIG_VERSION:
        raise ValueError(f"unsupported adapter config schema version: {version}")
    target = config.get("target_token", {})
    lora = config.get("encoder_qv_lora", {})
    if not isinstance(target, Mapping) or not isinstance(lora, Mapping):
        raise TypeError("target_token and encoder_qv_lora config entries must be mappings")
    rank = int(lora.get("rank", 0)) if bool(lora.get("enabled", False)) else 0
    return make_adapter_config(
        target_token=bool(target.get("enabled", False)),
        lora_rank=rank,
        lora_alpha=float(lora.get("alpha", DEFAULT_LORA_ALPHA)),
        model_type=str(config.get("model_type", DEFAULT_MODEL_TYPE)),
        lora_blocks=tuple(lora.get("blocks", DEFAULT_LORA_BLOCKS)),
        target_vocab=tuple(target.get("vocab", TARGET_VOCAB)),
        target_embedding_dim=int(target.get("embedding_dim", DEFAULT_TARGET_EMBED_DIM)),
    )


def adapters_enabled(config: Mapping[str, Any]) -> bool:
    normalized = normalize_adapter_config(config)
    return bool(
        normalized["target_token"]["enabled"]
        or normalized["encoder_qv_lora"]["enabled"]
    )


def _capture_torch_rng_state() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return cpu_state, cuda_states


def _restore_torch_rng_state(
    state: tuple[torch.Tensor, list[torch.Tensor] | None],
) -> None:
    cpu_state, cuda_states = state
    torch.random.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


def apply_adapter_config(model: nn.Module, config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = normalize_adapter_config(config)
    target = normalized["target_token"]
    lora = normalized["encoder_qv_lora"]
    # Every optional component starts from the same RNG snapshot. This keeps
    # target-only vs target+LoRA and LoRA-only vs target+LoRA factorial arms
    # paired even when all parameters are initialized on the same device.
    rng_state = _capture_torch_rng_state()
    try:
        if target["enabled"]:
            _restore_torch_rng_state(rng_state)
            inject_target_embedding(
                model,
                target_vocab=target["vocab"],
                embedding_dim=target["embedding_dim"],
            )
        if lora["enabled"]:
            _restore_torch_rng_state(rng_state)
            inject_qv_lora(
                model,
                rank=lora["rank"],
                alpha=lora["alpha"],
                block_indexes=lora["blocks"],
            )
    finally:
        _restore_torch_rng_state(rng_state)
    return normalized


def extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, Mapping) and "model_state_dict" in payload:
        state = payload["model_state_dict"]
    elif isinstance(payload, Mapping) and "state_dict" in payload:
        state = payload["state_dict"]
    else:
        state = payload
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint does not contain a state dict")
    state = dict(state)
    if any(key.startswith("module.") for key in state):
        state = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state.items()
        }
    return state


def load_model_with_adapters(
    checkpoint: str | Path,
    model_factory: Callable[[str], nn.Module],
    *,
    device: torch.device | str,
    requested_config: Mapping[str, Any] | None = None,
    default_model_type: str = DEFAULT_MODEL_TYPE,
) -> tuple[nn.Module, dict[str, Any]]:
    """Strictly load legacy or adapter-aware checkpoints.

    Legacy checkpoints are loaded into the unmodified model before requested
    adapters are injected. Adapter checkpoints are rebuilt from their saved
    config before state loading, so no adapter key can be silently discarded.
    """
    payload = torch.load(checkpoint, map_location=device)
    state = extract_state_dict(payload)
    saved_raw = None
    if isinstance(payload, Mapping) and (
        "model_state_dict" in payload or "state_dict" in payload
    ):
        saved_raw = payload.get("adapter_config")
    saved = normalize_adapter_config(saved_raw) if saved_raw is not None else None
    requested = normalize_adapter_config(requested_config) if requested_config is not None else None

    if saved is not None and requested is not None:
        if saved["model_type"] != requested["model_type"]:
            raise ValueError(
                f"checkpoint model_type={saved['model_type']} does not match "
                f"requested model_type={requested['model_type']}"
            )
        if adapters_enabled(saved) and saved != requested:
            raise ValueError("requested adapter config does not match checkpoint adapter architecture")

    architecture = saved or requested or make_adapter_config(
        model_type=default_model_type,
        lora_rank=0,
    )
    model = model_factory(architecture["model_type"]).to(device)
    if saved is not None:
        apply_adapter_config(model, saved)
    model.load_state_dict(state, strict=True)

    effective = saved or make_adapter_config(
        model_type=architecture["model_type"],
        lora_rank=0,
    )
    if requested is not None and (saved is None or not adapters_enabled(saved)):
        apply_adapter_config(model, requested)
        effective = requested
    model.eval()
    return model, effective
