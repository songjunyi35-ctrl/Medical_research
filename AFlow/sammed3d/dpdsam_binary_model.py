"""Label-free T1ce binary SAM-Med3D baseline primitives.

This is the B0 reconstruction arm: one learned, case-independent residual-tumor
prompt token and the official single-mask decoder path. Ground truth is not an
input to ``forward`` and is used only by the training loss/evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from sammed3d import peft


TARGET_NAME = "residual_tumor_123"
SCHEMA = "aflow.sammed3d.dpdsam_t1ce_binary_b0"
SCHEMA_VERSION = 1
D1_SCHEMA = "aflow.sammed3d.dpdsam_t1ce_binary_d1"
D1_SCHEMA_VERSION = 1
A2_SCHEMA = "aflow.sammed3d.t1ce_binary_a2_spatial_prompt"
A2_SCHEMA_VERSION = 1
DEFAULT_DOMAIN_COUNT = 4
DEFAULT_PROMPT_TOKENS = 1


@dataclass(frozen=True)
class BinarySAMOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    iou_predictions: torch.Tensor


@dataclass(frozen=True)
class DynamicBinarySAMOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    iou_predictions: torch.Tensor
    domain_logits: torch.Tensor
    domain_affinities: torch.Tensor
    pseudo_domain_ids: torch.Tensor
    global_descriptor: torch.Tensor
    prompt_prefix: torch.Tensor


@dataclass(frozen=True)
class SpatialPromptBinarySAMOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    iou_predictions: torch.Tensor
    dense_prompt_residual: torch.Tensor
    dense_prompt_embeddings: torch.Tensor


class SpatialDensePromptGenerator(nn.Module):
    """Lightweight depthwise-separable 3D residual over frozen image embeddings."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = int(embedding_dim)
        self.depthwise = nn.Conv3d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=3,
            padding=1,
            groups=self.embedding_dim,
            bias=True,
        )
        self.activation = nn.GELU()
        self.pointwise = nn.Conv3d(
            self.embedding_dim, self.embedding_dim, kernel_size=1, bias=True
        )
        nn.init.kaiming_normal_(self.depthwise.weight, nonlinearity="linear")
        nn.init.zeros_(self.depthwise.bias)
        nn.init.zeros_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        if image_embeddings.ndim != 5 or image_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                "image embeddings must be [B,C,D,H,W] with the configured channels"
            )
        return self.pointwise(self.activation(self.depthwise(image_embeddings)))


class LatentDomainRegistry(nn.Module):
    """Immutable-at-runtime centroids fitted externally on training descriptors."""

    def __init__(
        self,
        domain_count: int,
        descriptor_dim: int,
        centroids: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if domain_count <= 1:
            raise ValueError("domain_count must be greater than one")
        if descriptor_dim <= 0:
            raise ValueError("descriptor_dim must be positive")
        self.domain_count = int(domain_count)
        self.descriptor_dim = int(descriptor_dim)
        self.register_buffer(
            "centroids",
            torch.zeros(self.domain_count, self.descriptor_dim, dtype=torch.float32),
        )
        self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool))
        if centroids is not None:
            self.set_centroids(centroids)

    def set_centroids(self, centroids: torch.Tensor) -> None:
        if bool(self.fitted):
            raise RuntimeError("latent-domain centroids are already fitted")
        values = torch.as_tensor(centroids, dtype=torch.float32, device=self.centroids.device)
        expected = (self.domain_count, self.descriptor_dim)
        if tuple(values.shape) != expected:
            raise ValueError(f"centroids must have shape {expected}, got {tuple(values.shape)}")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("centroids must contain only finite values")
        with torch.no_grad():
            self.centroids.copy_(values)
            self.fitted.fill_(True)

    def pseudo_domain_ids(self, descriptors: torch.Tensor) -> torch.Tensor:
        if not bool(self.fitted):
            raise RuntimeError("latent-domain centroids have not been fitted")
        if descriptors.ndim != 2 or descriptors.shape[1] != self.descriptor_dim:
            raise ValueError(
                "descriptors must be [B,descriptor_dim], got "
                f"{tuple(descriptors.shape)}"
            )
        centroids = self.centroids.to(
            device=descriptors.device, dtype=descriptors.dtype
        )
        squared_distances = torch.sum(
            (descriptors[:, None, :] - centroids[None, :, :]).square(), dim=-1
        )
        return torch.argmin(squared_distances, dim=1)


class DomainAffinityAdapter(nn.Module):
    """Paper-declared two-layer MLP followed by domain Softmax."""

    def __init__(self, descriptor_dim: int, hidden_dim: int, domain_count: int) -> None:
        super().__init__()
        if min(descriptor_dim, hidden_dim, domain_count) <= 0:
            raise ValueError("adapter dimensions must be positive")
        self.descriptor_dim = int(descriptor_dim)
        self.hidden_dim = int(hidden_dim)
        self.domain_count = int(domain_count)
        self.fc1 = nn.Linear(self.descriptor_dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_dim, self.domain_count)

    def forward(self, descriptors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if descriptors.ndim != 2 or descriptors.shape[1] != self.descriptor_dim:
            raise ValueError("adapter descriptors have an invalid shape")
        logits = self.fc2(self.activation(self.fc1(descriptors)))
        return logits, torch.softmax(logits, dim=-1)


class RankOneDomainPromptGenerator(nn.Module):
    """Reconstruct P_k = P_shared * (U_k V_k^T), then mix by affinity."""

    def __init__(
        self,
        domain_count: int,
        prompt_tokens: int,
        prompt_dim: int,
    ) -> None:
        super().__init__()
        if min(domain_count, prompt_tokens, prompt_dim) <= 0:
            raise ValueError("prompt-generator dimensions must be positive")
        self.domain_count = int(domain_count)
        self.prompt_tokens = int(prompt_tokens)
        self.prompt_dim = int(prompt_dim)
        self.shared_prompt = nn.Parameter(torch.empty(self.prompt_tokens, self.prompt_dim))
        self.domain_u = nn.Parameter(
            torch.empty(self.domain_count, self.prompt_tokens, 1)
        )
        self.domain_v = nn.Parameter(torch.empty(self.domain_count, self.prompt_dim, 1))
        nn.init.normal_(self.shared_prompt, mean=0.0, std=0.02)
        nn.init.normal_(self.domain_u, mean=1.0, std=0.02)
        nn.init.normal_(self.domain_v, mean=1.0, std=0.02)

    def domain_bases(self) -> torch.Tensor:
        modulation = self.domain_u @ self.domain_v.transpose(-1, -2)
        return self.shared_prompt.unsqueeze(0) * modulation

    def forward(self, affinities: torch.Tensor) -> torch.Tensor:
        if affinities.ndim != 2 or affinities.shape[1] != self.domain_count:
            raise ValueError(
                f"affinities must have shape [B,{self.domain_count}], got "
                f"{tuple(affinities.shape)}"
            )
        return torch.einsum("bk,knd->bnd", affinities, self.domain_bases())


def _validate_single_channel_core(model: nn.Module) -> None:
    try:
        projection = model.image_encoder.patch_embed.proj
    except AttributeError as exc:
        raise TypeError("model lacks image_encoder.patch_embed.proj") from exc
    if not isinstance(projection, nn.Conv3d) or projection.in_channels != 1:
        raise ValueError("B0 requires an unchanged single-channel SAM-Med3D encoder")
    prompt_encoder = getattr(model, "prompt_encoder", None)
    if (
        prompt_encoder is None
        or not hasattr(prompt_encoder, "embed_dim")
        or not hasattr(prompt_encoder, "no_mask_embed")
        or not hasattr(prompt_encoder, "get_dense_pe")
    ):
        raise TypeError("model does not expose the required prompt-encoder contract")
    mask_decoder = getattr(model, "mask_decoder", None)
    if mask_decoder is None or not callable(mask_decoder):
        raise TypeError("model does not expose a callable mask decoder")


def decode_static_binary_prompt(
    model: nn.Module,
    image_embeddings: torch.Tensor,
    *,
    output_size: tuple[int, int, int],
) -> BinarySAMOutput:
    """Decode one learned label-free prompt through official token zero."""
    if image_embeddings.ndim != 5:
        raise ValueError("image_embeddings must be [B,C,D,H,W]")
    if len(output_size) != 3 or any(int(value) <= 0 for value in output_size):
        raise ValueError("output_size must contain three positive dimensions")
    target_embed = getattr(model, "target_embed", None)
    if not isinstance(target_embed, peft.TargetEmbedding):
        raise RuntimeError("B0 residual-tumor prompt token has not been injected")
    if target_embed.target_vocab != (TARGET_NAME,):
        raise ValueError("B0 target embedding vocabulary is not frozen")

    batch, channels, depth, height, width = image_embeddings.shape
    if target_embed.embedding_dim != channels:
        raise ValueError("target token and image embedding dimensions differ")
    sparse = target_embed.weight.to(
        device=image_embeddings.device, dtype=image_embeddings.dtype
    ).reshape(1, 1, channels).expand(batch, 1, channels)
    no_mask = model.prompt_encoder.no_mask_embed.weight.to(
        device=image_embeddings.device, dtype=image_embeddings.dtype
    )
    if tuple(no_mask.shape) != (1, channels):
        raise ValueError("no_mask_embed shape does not match image embeddings")
    dense = no_mask.reshape(1, channels, 1, 1, 1).expand(
        batch, channels, depth, height, width
    )
    image_pe = model.prompt_encoder.get_dense_pe().to(
        device=image_embeddings.device, dtype=image_embeddings.dtype
    )
    logits, iou = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    if logits.ndim != 5 or logits.shape[:2] != (batch, 1):
        raise RuntimeError(f"mask decoder returned invalid logits {tuple(logits.shape)}")
    if logits.shape[-3:] != output_size:
        logits = F.interpolate(
            logits, size=output_size, mode="trilinear", align_corners=False
        )
    if iou.ndim != 2 or iou.shape != (batch, 1):
        raise RuntimeError(f"mask decoder returned invalid IoU shape {tuple(iou.shape)}")
    return BinarySAMOutput(
        logits=logits,
        probabilities=torch.sigmoid(logits),
        iou_predictions=iou,
    )


def decode_dynamic_binary_prompt(
    model: nn.Module,
    image_embeddings: torch.Tensor,
    prompt_prefix: torch.Tensor,
    *,
    output_size: tuple[int, int, int],
) -> BinarySAMOutput:
    """Decode an image-conditioned continuous prefix through official token zero."""
    if image_embeddings.ndim != 5:
        raise ValueError("image_embeddings must be [B,C,D,H,W]")
    if prompt_prefix.ndim != 3:
        raise ValueError("prompt_prefix must be [B,N,C]")
    batch, channels, depth, height, width = image_embeddings.shape
    if prompt_prefix.shape[0] != batch or prompt_prefix.shape[2] != channels:
        raise ValueError("prompt prefix does not match image-embedding batch/channels")
    if prompt_prefix.shape[1] <= 0:
        raise ValueError("prompt prefix must contain at least one token")
    if len(output_size) != 3 or any(int(value) <= 0 for value in output_size):
        raise ValueError("output_size must contain three positive dimensions")
    if not bool(torch.isfinite(prompt_prefix).all()):
        raise ValueError("prompt prefix contains non-finite values")

    no_mask = model.prompt_encoder.no_mask_embed.weight.to(
        device=image_embeddings.device, dtype=image_embeddings.dtype
    )
    if tuple(no_mask.shape) != (1, channels):
        raise ValueError("no_mask_embed shape does not match image embeddings")
    dense = no_mask.reshape(1, channels, 1, 1, 1).expand(
        batch, channels, depth, height, width
    )
    image_pe = model.prompt_encoder.get_dense_pe().to(
        device=image_embeddings.device, dtype=image_embeddings.dtype
    )
    logits, iou = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=image_pe,
        sparse_prompt_embeddings=prompt_prefix.to(dtype=image_embeddings.dtype),
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    if logits.ndim != 5 or logits.shape[:2] != (batch, 1):
        raise RuntimeError(f"mask decoder returned invalid logits {tuple(logits.shape)}")
    if logits.shape[-3:] != output_size:
        logits = F.interpolate(
            logits, size=output_size, mode="trilinear", align_corners=False
        )
    if iou.ndim != 2 or iou.shape != (batch, 1):
        raise RuntimeError(f"mask decoder returned invalid IoU shape {tuple(iou.shape)}")
    return BinarySAMOutput(
        logits=logits,
        probabilities=torch.sigmoid(logits),
        iou_predictions=iou,
    )


def decode_spatial_binary_prompt(
    model: nn.Module,
    image_embeddings: torch.Tensor,
    dense_prompt_residual: torch.Tensor,
    *,
    output_size: tuple[int, int, int],
) -> SpatialPromptBinarySAMOutput:
    """Add a spatial residual to no-mask dense prompt and decode token zero."""
    if image_embeddings.ndim != 5:
        raise ValueError("image_embeddings must be [B,C,D,H,W]")
    if dense_prompt_residual.shape != image_embeddings.shape:
        raise ValueError("dense prompt residual must match image embeddings exactly")
    if len(output_size) != 3 or any(int(value) <= 0 for value in output_size):
        raise ValueError("output_size must contain three positive dimensions")
    if not bool(torch.isfinite(dense_prompt_residual).all()):
        raise ValueError("dense prompt residual contains non-finite values")

    target_embed = getattr(model, "target_embed", None)
    if not isinstance(target_embed, peft.TargetEmbedding):
        raise RuntimeError("A2 residual-tumor sparse target token has not been injected")
    if target_embed.target_vocab != (TARGET_NAME,):
        raise ValueError("A2 target embedding vocabulary is not frozen")

    batch, channels, depth, height, width = image_embeddings.shape
    if target_embed.embedding_dim != channels:
        raise ValueError("target token and image embedding dimensions differ")
    sparse = target_embed.weight.to(
        device=image_embeddings.device, dtype=image_embeddings.dtype
    ).reshape(1, 1, channels).expand(batch, 1, channels)
    no_mask = model.prompt_encoder.no_mask_embed.weight.to(
        device=image_embeddings.device, dtype=image_embeddings.dtype
    )
    if tuple(no_mask.shape) != (1, channels):
        raise ValueError("no_mask_embed shape does not match image embeddings")
    dense_base = no_mask.reshape(1, channels, 1, 1, 1).expand(
        batch, channels, depth, height, width
    )
    dense = dense_base + dense_prompt_residual.to(dtype=image_embeddings.dtype)
    image_pe = model.prompt_encoder.get_dense_pe().to(
        device=image_embeddings.device, dtype=image_embeddings.dtype
    )
    logits, iou = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    if logits.ndim != 5 or logits.shape[:2] != (batch, 1):
        raise RuntimeError(f"mask decoder returned invalid logits {tuple(logits.shape)}")
    if logits.shape[-3:] != output_size:
        logits = F.interpolate(
            logits, size=output_size, mode="trilinear", align_corners=False
        )
    if iou.ndim != 2 or iou.shape != (batch, 1):
        raise RuntimeError(f"mask decoder returned invalid IoU shape {tuple(iou.shape)}")
    return SpatialPromptBinarySAMOutput(
        logits=logits,
        probabilities=torch.sigmoid(logits),
        iou_predictions=iou,
        dense_prompt_residual=dense_prompt_residual,
        dense_prompt_embeddings=dense,
    )


class StaticPromptBinarySAM(nn.Module):
    """T1ce-only automatic binary SAM with no label-bearing forward argument."""

    def __init__(self, sam_model: nn.Module) -> None:
        super().__init__()
        _validate_single_channel_core(sam_model)
        target_embed = getattr(sam_model, "target_embed", None)
        if not isinstance(target_embed, peft.TargetEmbedding):
            raise RuntimeError("B0 residual-tumor prompt token has not been injected")
        if target_embed.target_vocab != (TARGET_NAME,):
            raise ValueError("unexpected B0 target vocabulary")
        self.sam_model = sam_model

    @property
    def image_encoder(self) -> nn.Module:
        return self.sam_model.image_encoder

    def forward(self, images: torch.Tensor) -> BinarySAMOutput:
        if images.ndim != 5 or images.shape[1] != 1:
            raise ValueError(f"B0 images must be [B,1,D,H,W], got {tuple(images.shape)}")
        embeddings = self.sam_model.image_encoder(images)
        return decode_static_binary_prompt(
            self.sam_model, embeddings, output_size=tuple(images.shape[-3:])
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "arm": "B0_static_learned_prompt",
            "input": "T1ce_only_single_channel_whole_volume_128cubed",
            "target": {"name": TARGET_NAME, "foreground_raw_labels": [1, 2, 3]},
            "automatic_inference_reads_gt": False,
            "prompt": "one_learned_case_independent_continuous_token",
            "decoder": "official_single_mask_token_zero",
            "threshold": 0.5,
            "reproduction_status": "declared_setting_reconstruction_not_bit_exact",
        }


class DynamicDomainPromptBinarySAM(nn.Module):
    """Minimal D1 reconstruction with GAP descriptor and rank-one domain prompts."""

    def __init__(
        self,
        sam_model: nn.Module,
        *,
        centroids: torch.Tensor | None = None,
        domain_count: int = DEFAULT_DOMAIN_COUNT,
        prompt_tokens: int = DEFAULT_PROMPT_TOKENS,
        adapter_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        _validate_single_channel_core(sam_model)
        if getattr(sam_model, "target_embed", None) is not None:
            raise ValueError("D1 owns its prompt prefix and cannot reuse a B0 target token")
        if list(peft.iter_qv_lora_modules(sam_model)):
            raise ValueError("D1 reconstruction requires an encoder without Q/V LoRA")
        prompt_dim = int(sam_model.prompt_encoder.embed_dim)
        hidden_dim = prompt_dim if adapter_hidden_dim is None else int(adapter_hidden_dim)
        self.sam_model = sam_model
        self.domain_registry = LatentDomainRegistry(
            domain_count, prompt_dim, centroids=centroids
        )
        self.domain_adapter = DomainAffinityAdapter(
            prompt_dim, hidden_dim, domain_count
        )
        self.prompt_generator = RankOneDomainPromptGenerator(
            domain_count, prompt_tokens, prompt_dim
        )

    @property
    def image_encoder(self) -> nn.Module:
        return self.sam_model.image_encoder

    def set_domain_centroids(self, centroids: torch.Tensor) -> None:
        self.domain_registry.set_centroids(centroids)

    def train(self, mode: bool = True) -> "DynamicDomainPromptBinarySAM":
        super().train(mode)
        self.sam_model.image_encoder.eval()
        self.sam_model.prompt_encoder.eval()
        return self

    def forward(self, images: torch.Tensor) -> DynamicBinarySAMOutput:
        if images.ndim != 5 or images.shape[1] != 1:
            raise ValueError(f"D1 images must be [B,1,D,H,W], got {tuple(images.shape)}")
        with torch.no_grad():
            image_embeddings = self.sam_model.image_encoder(images)
        if image_embeddings.ndim != 5:
            raise RuntimeError("image encoder must return [B,C,D,H,W]")
        descriptor = image_embeddings.float().mean(dim=(-3, -2, -1))
        if descriptor.shape[1] != self.domain_adapter.descriptor_dim:
            raise RuntimeError("GAP descriptor dimension does not match D1 adapter")
        domain_logits, affinities = self.domain_adapter(descriptor)
        pseudo_ids = self.domain_registry.pseudo_domain_ids(descriptor)
        prefix = self.prompt_generator(affinities)
        decoded = decode_dynamic_binary_prompt(
            self.sam_model,
            image_embeddings,
            prefix,
            output_size=tuple(images.shape[-3:]),
        )
        return DynamicBinarySAMOutput(
            logits=decoded.logits,
            probabilities=decoded.probabilities,
            iou_predictions=decoded.iou_predictions,
            domain_logits=domain_logits,
            domain_affinities=affinities,
            pseudo_domain_ids=pseudo_ids,
            global_descriptor=descriptor,
            prompt_prefix=prefix,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": D1_SCHEMA,
            "schema_version": D1_SCHEMA_VERSION,
            "arm": "D1_dynamic_domain_prompt_reconstruction",
            "input": "T1ce_only_single_channel_whole_volume_128cubed",
            "target": {"name": TARGET_NAME, "foreground_raw_labels": [1, 2, 3]},
            "automatic_inference_reads_gt": False,
            "image_encoder": "frozen_no_encoder_lora",
            "cls_surrogate": {
                "type": "global_average_pooling",
                "source": "frozen_sam_encoder_output_embedding",
                "learnable": False,
                "descriptor_dim": self.domain_adapter.descriptor_dim,
            },
            "latent_domains": {
                "count": self.domain_registry.domain_count,
                "centroids_fitted_externally": bool(self.domain_registry.fitted),
                "assignment": "nearest_squared_euclidean_centroid",
            },
            "affinity_adapter": {
                "architecture": "linear_gelu_linear_softmax",
                "hidden_dim": self.domain_adapter.hidden_dim,
            },
            "prompt": {
                "type": "image_conditioned_rank_one_domain_prefix",
                "formula": "P_k=P_shared_hadamard_(U_k_matmul_V_k_transpose)",
                "rank": 1,
                "tokens": self.prompt_generator.prompt_tokens,
                "dimension": self.prompt_generator.prompt_dim,
                "injection": "sparse_prompt_prefix_no_points_no_boxes",
                "reconstruction_initialization": {
                    "shared_prompt": "normal_mean_0_std_0.02",
                    "domain_factors": "normal_mean_1_std_0.02",
                },
            },
            "decoder": "official_single_mask_token_zero",
            "threshold": 0.5,
            "reproduction_status": "dpdsam_minimal_reconstruction_not_bit_exact",
        }


class SpatialAutomaticPromptBinarySAM(nn.Module):
    """A2 engineering arm for testing image-conditioned spatial dense prompts."""

    def __init__(self, sam_model: nn.Module) -> None:
        super().__init__()
        _validate_single_channel_core(sam_model)
        target_embed = getattr(sam_model, "target_embed", None)
        if not isinstance(target_embed, peft.TargetEmbedding):
            raise RuntimeError("A2 residual-tumor sparse target token has not been injected")
        if target_embed.target_vocab != (TARGET_NAME,):
            raise ValueError("unexpected A2 target vocabulary")
        self.sam_model = sam_model
        self.spatial_prompt_generator = SpatialDensePromptGenerator(
            int(sam_model.prompt_encoder.embed_dim)
        )

    @property
    def image_encoder(self) -> nn.Module:
        return self.sam_model.image_encoder

    @property
    def encoder_lora_enabled(self) -> bool:
        return bool(list(peft.iter_qv_lora_modules(self.sam_model)))

    def train(self, mode: bool = True) -> "SpatialAutomaticPromptBinarySAM":
        super().train(mode)
        if self.encoder_lora_enabled:
            self.sam_model.image_encoder.train(mode)
        else:
            self.sam_model.image_encoder.eval()
        self.sam_model.prompt_encoder.eval()
        return self

    def forward(self, images: torch.Tensor) -> SpatialPromptBinarySAMOutput:
        if images.ndim != 5 or images.shape[1] != 1:
            raise ValueError(f"A2 images must be [B,1,D,H,W], got {tuple(images.shape)}")
        if self.encoder_lora_enabled:
            image_embeddings = self.sam_model.image_encoder(images)
        else:
            with torch.no_grad():
                image_embeddings = self.sam_model.image_encoder(images)
        dense_residual = self.spatial_prompt_generator(image_embeddings)
        return decode_spatial_binary_prompt(
            self.sam_model,
            image_embeddings,
            dense_residual,
            output_size=tuple(images.shape[-3:]),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": A2_SCHEMA,
            "schema_version": A2_SCHEMA_VERSION,
            "arm": "A2_spatial_automatic_prompt_engineering",
            "scientific_status": "engineering_root_cause_arm_not_dpdsam_paper",
            "hypothesis": "missing_image_conditioned_spatial_prompt_limits_B0_D1",
            "input": "T1ce_only_single_channel_whole_volume_128cubed",
            "target": {"name": TARGET_NAME, "foreground_raw_labels": [1, 2, 3]},
            "automatic_inference_reads_gt": False,
            "sparse_prompt": "one_learned_case_independent_target_token",
            "dense_prompt": {
                "base": "frozen_prompt_encoder_no_mask_embedding",
                "residual_source": "image_encoder_embedding",
                "generator": "depthwise_conv3d_k3_gelu_pointwise_conv3d_k1",
                "shape": "same_as_image_embedding",
                "output_initialization": "pointwise_zero_exact_no_mask_start",
            },
            "image_encoder": {
                "base_weights": "frozen",
                "optional_qv_lora": self.encoder_lora_enabled,
            },
            "decoder": "official_single_mask_token_zero",
            "threshold": 0.5,
            "reproduction_status": "a2_spatial_engineering_not_dpdsam_reproduction",
        }


def make_b0_adapter_config(
    *,
    model_type: str = peft.DEFAULT_MODEL_TYPE,
    embedding_dim: int = peft.DEFAULT_TARGET_EMBED_DIM,
    lora_rank: int = 0,
    lora_alpha: float = peft.DEFAULT_LORA_ALPHA,
) -> dict[str, Any]:
    return peft.make_adapter_config(
        target_token=True,
        target_vocab=(TARGET_NAME,),
        target_embedding_dim=embedding_dim,
        model_type=model_type,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )


def make_a2_adapter_config(
    *,
    model_type: str = peft.DEFAULT_MODEL_TYPE,
    embedding_dim: int = peft.DEFAULT_TARGET_EMBED_DIM,
    lora_rank: int = 0,
    lora_alpha: float = peft.DEFAULT_LORA_ALPHA,
) -> dict[str, Any]:
    return peft.make_adapter_config(
        target_token=True,
        target_vocab=(TARGET_NAME,),
        target_embedding_dim=embedding_dim,
        model_type=model_type,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )


def load_untrained_b0(
    checkpoint: str | Path,
    model_factory: Callable[[str], nn.Module],
    *,
    device: torch.device | str,
    model_type: str = peft.DEFAULT_MODEL_TYPE,
    embedding_dim: int = peft.DEFAULT_TARGET_EMBED_DIM,
    lora_rank: int = 0,
    lora_alpha: float = peft.DEFAULT_LORA_ALPHA,
) -> tuple[StaticPromptBinarySAM, dict[str, Any]]:
    requested = make_b0_adapter_config(
        model_type=model_type,
        embedding_dim=embedding_dim,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
    core, effective = peft.load_model_with_adapters(
        checkpoint,
        model_factory,
        device=device,
        requested_config=requested,
        default_model_type=model_type,
    )
    if effective != requested:
        raise RuntimeError("effective B0 adapter configuration changed during load")
    wrapper = StaticPromptBinarySAM(core)
    metadata = wrapper.metadata()
    metadata["arm"] = "B0_static_prompt" if lora_rank == 0 else f"B1_static_prompt_lora_r{lora_rank}"
    metadata["adapter_config"] = effective
    return wrapper, metadata


def load_untrained_a2(
    checkpoint: str | Path,
    model_factory: Callable[[str], nn.Module],
    *,
    device: torch.device | str,
    model_type: str = peft.DEFAULT_MODEL_TYPE,
    embedding_dim: int = peft.DEFAULT_TARGET_EMBED_DIM,
    lora_rank: int = 0,
    lora_alpha: float = peft.DEFAULT_LORA_ALPHA,
) -> tuple[SpatialAutomaticPromptBinarySAM, dict[str, Any]]:
    requested = make_a2_adapter_config(
        model_type=model_type,
        embedding_dim=embedding_dim,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
    core, effective = peft.load_model_with_adapters(
        checkpoint,
        model_factory,
        device=device,
        requested_config=requested,
        default_model_type=model_type,
    )
    if effective != requested:
        raise RuntimeError("effective A2 adapter configuration changed during load")
    wrapper = SpatialAutomaticPromptBinarySAM(core).to(device)
    metadata = wrapper.metadata()
    metadata["adapter_config"] = effective
    return wrapper, metadata


def make_d1_adapter_config(
    *, model_type: str = peft.DEFAULT_MODEL_TYPE
) -> dict[str, Any]:
    """D1 keeps the image encoder unchanged and owns prompts outside PEFT."""
    return peft.make_adapter_config(
        target_token=False,
        model_type=model_type,
        lora_rank=0,
    )


def load_untrained_d1(
    checkpoint: str | Path,
    model_factory: Callable[[str], nn.Module],
    *,
    device: torch.device | str,
    centroids: torch.Tensor | None = None,
    model_type: str = peft.DEFAULT_MODEL_TYPE,
    domain_count: int = DEFAULT_DOMAIN_COUNT,
    prompt_tokens: int = DEFAULT_PROMPT_TOKENS,
    adapter_hidden_dim: int | None = None,
) -> tuple[DynamicDomainPromptBinarySAM, dict[str, Any]]:
    requested = make_d1_adapter_config(model_type=model_type)
    core, effective = peft.load_model_with_adapters(
        checkpoint,
        model_factory,
        device=device,
        requested_config=requested,
        default_model_type=model_type,
    )
    if effective != requested:
        raise RuntimeError("effective D1 adapter configuration changed during load")
    wrapper = DynamicDomainPromptBinarySAM(
        core,
        centroids=centroids,
        domain_count=domain_count,
        prompt_tokens=prompt_tokens,
        adapter_hidden_dim=adapter_hidden_dim,
    ).to(device)
    metadata = wrapper.metadata()
    metadata["adapter_config"] = effective
    return wrapper, metadata


def configure_b0_trainable(model: StaticPromptBinarySAM) -> list[nn.Parameter]:
    """Freeze the encoder/prompt encoder; train the token and mask decoder."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.sam_model.mask_decoder.parameters():
        parameter.requires_grad_(True)
    iou_head = getattr(model.sam_model.mask_decoder, "iou_prediction_head", None)
    if isinstance(iou_head, nn.Module):
        for parameter in iou_head.parameters():
            parameter.requires_grad_(False)
    for parameter in model.sam_model.target_embed.parameters():
        parameter.requires_grad_(True)
    for adapter in peft.iter_qv_lora_modules(model.sam_model):
        for parameter in adapter.lora_parameters():
            parameter.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("B0 has no trainable parameters")
    return trainable


def configure_d1_trainable(
    model: DynamicDomainPromptBinarySAM,
) -> list[nn.Parameter]:
    """Freeze SAM encoders; train DPG, affinity adapter, and mask decoder."""
    if not isinstance(model, DynamicDomainPromptBinarySAM):
        raise TypeError("configure_d1_trainable requires DynamicDomainPromptBinarySAM")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.sam_model.mask_decoder.parameters():
        parameter.requires_grad_(True)
    iou_head = getattr(model.sam_model.mask_decoder, "iou_prediction_head", None)
    if isinstance(iou_head, nn.Module):
        for parameter in iou_head.parameters():
            parameter.requires_grad_(False)
    for module in (model.domain_adapter, model.prompt_generator):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.sam_model.image_encoder.eval()
    model.sam_model.prompt_encoder.eval()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("D1 has no trainable parameters")
    if any(
        parameter.requires_grad
        for parameter in model.sam_model.image_encoder.parameters()
    ):
        raise RuntimeError("D1 image encoder must remain frozen")
    return trainable


def configure_a2_trainable(
    model: SpatialAutomaticPromptBinarySAM,
) -> list[nn.Parameter]:
    """Train A2 prompts/decoder and optional LoRA while freezing SAM bases."""
    if not isinstance(model, SpatialAutomaticPromptBinarySAM):
        raise TypeError("configure_a2_trainable requires SpatialAutomaticPromptBinarySAM")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.sam_model.mask_decoder.parameters():
        parameter.requires_grad_(True)
    iou_head = getattr(model.sam_model.mask_decoder, "iou_prediction_head", None)
    if isinstance(iou_head, nn.Module):
        for parameter in iou_head.parameters():
            parameter.requires_grad_(False)
    for module in (model.sam_model.target_embed, model.spatial_prompt_generator):
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    lora_parameter_ids: set[int] = set()
    for adapter in peft.iter_qv_lora_modules(model.sam_model):
        for parameter in adapter.lora_parameters():
            parameter.requires_grad_(True)
            lora_parameter_ids.add(id(parameter))

    unintended_encoder_parameters = [
        name
        for name, parameter in model.sam_model.image_encoder.named_parameters()
        if parameter.requires_grad and id(parameter) not in lora_parameter_ids
    ]
    if unintended_encoder_parameters:
        raise RuntimeError(
            "A2 base image-encoder parameters became trainable: "
            + ", ".join(unintended_encoder_parameters)
        )
    model.sam_model.prompt_encoder.eval()
    if model.encoder_lora_enabled:
        model.sam_model.image_encoder.train(model.training)
    else:
        model.sam_model.image_encoder.eval()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("A2 has no trainable parameters")
    if len({id(parameter) for parameter in trainable}) != len(trainable):
        raise RuntimeError("A2 trainable parameter list contains aliases")
    return trainable


def checkpoint_payload(
    model: (
        StaticPromptBinarySAM
        | DynamicDomainPromptBinarySAM
        | SpatialAutomaticPromptBinarySAM
    ),
    *,
    adapter_config: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_adapter = peft.normalize_adapter_config(adapter_config)
    if isinstance(model, DynamicDomainPromptBinarySAM) and peft.adapters_enabled(
        normalized_adapter
    ):
        raise ValueError("D1 checkpoints cannot declare target-token or encoder-LoRA adapters")
    payload: dict[str, Any] = {
        "model_state_dict": model.sam_model.state_dict(),
        "adapter_config": normalized_adapter,
        "dpdsam_binary_config": model.metadata(),
    }
    if isinstance(model, DynamicDomainPromptBinarySAM):
        payload["dpdsam_dynamic_state_dict"] = {
            "domain_registry": model.domain_registry.state_dict(),
            "domain_adapter": model.domain_adapter.state_dict(),
            "prompt_generator": model.prompt_generator.state_dict(),
        }
    if isinstance(model, SpatialAutomaticPromptBinarySAM):
        target = normalized_adapter["target_token"]
        lora = normalized_adapter["encoder_qv_lora"]
        if (
            not target["enabled"]
            or target["vocab"] != [TARGET_NAME]
            or int(target["embedding_dim"])
            != int(model.sam_model.prompt_encoder.embed_dim)
            or bool(lora["enabled"]) != model.encoder_lora_enabled
        ):
            raise ValueError("A2 checkpoint adapter configuration does not match model")
        payload["a2_spatial_prompt_state_dict"] = (
            model.spatial_prompt_generator.state_dict()
        )
    if extra:
        overlap = set(payload).intersection(extra)
        if overlap:
            raise ValueError(f"extra checkpoint fields overwrite {sorted(overlap)}")
        payload.update(dict(extra))
    return payload


def load_a2_checkpoint(
    checkpoint: str | Path,
    model_factory: Callable[[str], nn.Module],
    *,
    device: torch.device | str,
) -> tuple[SpatialAutomaticPromptBinarySAM, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("A2 checkpoint payload must be a mapping")
    raw_config = payload.get("dpdsam_binary_config")
    spatial_state = payload.get("a2_spatial_prompt_state_dict")
    if not isinstance(raw_config, Mapping) or not isinstance(spatial_state, Mapping):
        raise ValueError("checkpoint does not contain an A2 spatial-prompt state")
    config = dict(raw_config)
    if (
        config.get("schema") != A2_SCHEMA
        or int(config.get("schema_version", -1)) != A2_SCHEMA_VERSION
    ):
        raise ValueError("unsupported A2 checkpoint schema")

    core, effective = peft.load_model_with_adapters(
        checkpoint,
        model_factory,
        device=device,
        default_model_type=peft.DEFAULT_MODEL_TYPE,
    )
    wrapper = SpatialAutomaticPromptBinarySAM(core).to(device)
    wrapper.spatial_prompt_generator.load_state_dict(dict(spatial_state), strict=True)
    if wrapper.metadata() != config:
        raise ValueError("loaded A2 architecture does not match checkpoint metadata")
    metadata = wrapper.metadata()
    metadata["adapter_config"] = effective
    return wrapper, metadata


def load_d1_checkpoint(
    checkpoint: str | Path,
    model_factory: Callable[[str], nn.Module],
    *,
    device: torch.device | str,
) -> tuple[DynamicDomainPromptBinarySAM, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("D1 checkpoint payload must be a mapping")
    raw_config = payload.get("dpdsam_binary_config")
    dynamic_state = payload.get("dpdsam_dynamic_state_dict")
    if not isinstance(raw_config, Mapping) or not isinstance(dynamic_state, Mapping):
        raise ValueError("checkpoint does not contain a D1 reconstruction state")
    config = dict(raw_config)
    if (
        config.get("schema") != D1_SCHEMA
        or int(config.get("schema_version", -1)) != D1_SCHEMA_VERSION
    ):
        raise ValueError("unsupported D1 checkpoint schema")
    latent = config.get("latent_domains")
    prompt = config.get("prompt")
    adapter = config.get("affinity_adapter")
    cls_surrogate = config.get("cls_surrogate")
    if not all(isinstance(item, Mapping) for item in (latent, prompt, adapter, cls_surrogate)):
        raise ValueError("D1 checkpoint architecture metadata is incomplete")

    core, effective = peft.load_model_with_adapters(
        checkpoint,
        model_factory,
        device=device,
        default_model_type=peft.DEFAULT_MODEL_TYPE,
    )
    wrapper = DynamicDomainPromptBinarySAM(
        core,
        domain_count=int(latent["count"]),
        prompt_tokens=int(prompt["tokens"]),
        adapter_hidden_dim=int(adapter["hidden_dim"]),
    ).to(device)
    for name, module in (
        ("domain_registry", wrapper.domain_registry),
        ("domain_adapter", wrapper.domain_adapter),
        ("prompt_generator", wrapper.prompt_generator),
    ):
        state = dynamic_state.get(name)
        if not isinstance(state, Mapping):
            raise ValueError(f"D1 checkpoint is missing {name} state")
        module.load_state_dict(dict(state), strict=True)
    if wrapper.metadata() != config:
        raise ValueError("loaded D1 architecture does not match checkpoint metadata")
    metadata = wrapper.metadata()
    metadata["adapter_config"] = effective
    return wrapper, metadata
