"""Latent-domain conditioned safe adapter for T1ce/WT refinement.

Domain-SAGE keeps the validated robust refiner frozen and conditions only the
small SAGE adapter.  Domain modulation is zero initialized, so construction is
function preserving: before optimization the output is exactly the frozen base.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from monai.networks.nets import SegResNet


@dataclass(frozen=True)
class DomainSAGEOutput:
    logits: torch.Tensor
    base_logits: torch.Tensor
    residual: torch.Tensor
    domain_logits: torch.Tensor
    domain_affinities: torch.Tensor
    film_abs_mean: torch.Tensor


class LowRankDomainFiLM(nn.Module):
    """Affinity-weighted low-rank FiLM with an exact identity initialization."""

    def __init__(self, domain_count: int, channels: int, rank: int) -> None:
        super().__init__()
        if min(domain_count, channels, rank) <= 0:
            raise ValueError("domain_count, channels, and rank must be positive")
        self.domain_count = int(domain_count)
        self.channels = int(channels)
        self.rank = int(rank)
        self.gamma_coeff = nn.Parameter(torch.zeros(domain_count, rank))
        self.beta_coeff = nn.Parameter(torch.zeros(domain_count, rank))
        self.gamma_basis = nn.Parameter(torch.empty(rank, channels))
        self.beta_basis = nn.Parameter(torch.empty(rank, channels))
        nn.init.normal_(self.gamma_basis, mean=0.0, std=0.1)
        nn.init.normal_(self.beta_basis, mean=0.0, std=0.1)

    def parameters_by_domain(self) -> tuple[torch.Tensor, torch.Tensor]:
        gamma = self.gamma_coeff @ self.gamma_basis
        beta = self.beta_coeff @ self.beta_basis
        return gamma, beta

    def forward(
        self, features: torch.Tensor, affinities: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 5 or features.shape[1] != self.channels:
            raise ValueError("features must be [B,C,D,H,W] with configured channels")
        if affinities.shape != (features.shape[0], self.domain_count):
            raise ValueError("domain affinities have an invalid shape")
        gamma_by_domain, beta_by_domain = self.parameters_by_domain()
        gamma = torch.tanh(affinities @ gamma_by_domain)
        beta = torch.tanh(affinities @ beta_by_domain)
        shape = (features.shape[0], self.channels, 1, 1, 1)
        gamma = gamma.reshape(shape)
        beta = beta.reshape(shape)
        modulated = features * (1.0 + gamma) + beta
        magnitude = 0.5 * (
            gamma.abs().flatten(1).mean(1) + beta.abs().flatten(1).mean(1)
        )
        return modulated, magnitude


class DomainAffinityMLP(nn.Module):
    """Two-layer descriptor-to-domain adapter, following the DPD-SAM principle."""

    def __init__(self, descriptor_dim: int, hidden_dim: int, domain_count: int) -> None:
        super().__init__()
        if min(descriptor_dim, hidden_dim, domain_count) <= 0:
            raise ValueError("adapter dimensions must be positive")
        self.descriptor_dim = int(descriptor_dim)
        self.domain_count = int(domain_count)
        self.network = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, domain_count),
        )

    def forward(self, descriptor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if descriptor.ndim != 2 or descriptor.shape[1] != self.descriptor_dim:
            raise ValueError("descriptor must be [B,descriptor_dim]")
        logits = self.network(descriptor)
        return logits, torch.softmax(logits, dim=1)


class DomainSAGESafeAdapter(nn.Module):
    """Frozen robust base plus a domain-conditioned residual SegResNet adapter."""

    def __init__(
        self,
        base: nn.Module,
        *,
        descriptor_dim: int,
        domain_count: int = 4,
        domain_hidden_dim: int = 64,
        film_rank: int = 4,
        init_filters: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.base.eval()
        self.domain_count = int(domain_count)
        self.descriptor_dim = int(descriptor_dim)
        self.adapter = SegResNet(
            spatial_dims=3,
            in_channels=5,
            # Keep the exact two-head SAGE adapter capacity.  The second head is
            # unused in both the matched plain arm and Domain-SAGE arm.
            out_channels=2,
            init_filters=init_filters,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=dropout,
        )
        final = self.adapter.conv_final[2].conv
        nn.init.zeros_(final.weight)
        if final.bias is not None:
            nn.init.zeros_(final.bias)
        self.domain_adapter = DomainAffinityMLP(
            descriptor_dim, domain_hidden_dim, domain_count
        )
        decoder_channels = [init_filters * 8, init_filters * 4, init_filters * 2, init_filters]
        self.film_layers = nn.ModuleList(
            [LowRankDomainFiLM(domain_count, channels, film_rank) for channels in decoder_channels]
        )

    def train(self, mode: bool = True) -> "DomainSAGESafeAdapter":
        super().train(mode)
        self.base.eval()
        return self

    @staticmethod
    def adapter_features(
        image: torch.Tensor, coarse_logits: torch.Tensor, base_logits: torch.Tensor
    ) -> torch.Tensor:
        base_probability = torch.sigmoid(base_logits)
        coarse_probability = torch.sigmoid(coarse_logits)
        base_uncertainty = 4.0 * base_probability * (1.0 - base_probability)
        disagreement = torch.abs(base_probability - coarse_probability)
        return torch.cat(
            [
                image,
                torch.tanh(coarse_logits / 5.0),
                torch.tanh(base_logits / 5.0),
                base_uncertainty,
                disagreement,
            ],
            dim=1,
        )

    def _conditioned_adapter(
        self, features: torch.Tensor, affinities: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, down_x = self.adapter.encode(features)
        magnitudes: list[torch.Tensor] = []
        x, magnitude = self.film_layers[0](x, affinities)
        magnitudes.append(magnitude)
        skips = list(reversed(down_x))
        for index, (up, up_layer) in enumerate(
            zip(self.adapter.up_samples, self.adapter.up_layers), start=1
        ):
            x = up(x) + skips[index]
            x = up_layer(x)
            x, magnitude = self.film_layers[index](x, affinities)
            magnitudes.append(magnitude)
        x = self.adapter.conv_final(x)
        residual, _unused_second_head = x.split(1, dim=1)
        return residual, torch.stack(magnitudes, dim=1).mean(dim=1)

    def forward_with_aux(
        self,
        image: torch.Tensor,
        coarse_logits: torch.Tensor,
        descriptor: torch.Tensor,
    ) -> DomainSAGEOutput:
        if image.shape != coarse_logits.shape:
            raise ValueError("image and coarse_logits must have identical shapes")
        if descriptor.shape != (image.shape[0], self.descriptor_dim):
            raise ValueError("descriptor does not match the image batch")
        with torch.no_grad():
            base_logits = self.base(image, coarse_logits)
        domain_logits, affinities = self.domain_adapter(descriptor)
        features = self.adapter_features(image, coarse_logits, base_logits)
        residual, film_abs_mean = self._conditioned_adapter(features, affinities)
        return DomainSAGEOutput(
            logits=base_logits + residual,
            base_logits=base_logits,
            residual=residual,
            domain_logits=domain_logits,
            domain_affinities=affinities,
            film_abs_mean=film_abs_mean,
        )

    def forward(
        self,
        image: torch.Tensor,
        coarse_logits: torch.Tensor,
        descriptor: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_with_aux(image, coarse_logits, descriptor).logits

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]


@torch.inference_mode()
def frozen_image_descriptor(
    base: nn.Module, image: torch.Tensor
) -> torch.Tensor:
    """Global average pooled frozen bottleneck with the coarse channel zeroed."""
    if image.ndim != 5 or image.shape[1] != 1:
        raise ValueError("image must be [B,1,D,H,W]")
    network = getattr(base, "network", None)
    if network is None or not hasattr(network, "encode"):
        raise TypeError("base does not expose a SegResNet encoder")
    base.eval()
    encoded, _ = network.encode(torch.cat([image, torch.zeros_like(image)], dim=1))
    return encoded.float().mean(dim=(-3, -2, -1))


def fit_kmeans_plus_plus(
    values: np.ndarray,
    *,
    cluster_count: int,
    seed: int,
    max_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Small dependency-free deterministic K-means++ implementation."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError("values must be a finite two-dimensional array")
    if not 1 < cluster_count <= data.shape[0]:
        raise ValueError("cluster_count must be in [2, number of rows]")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    rng = np.random.default_rng(seed)
    centers = np.empty((cluster_count, data.shape[1]), dtype=np.float64)
    first = int(rng.integers(data.shape[0]))
    centers[0] = data[first]
    nearest = np.sum((data - centers[0]) ** 2, axis=1)
    for index in range(1, cluster_count):
        total = float(nearest.sum())
        if total <= 0:
            chosen = int(np.flatnonzero(np.ones(data.shape[0], dtype=bool))[index])
        else:
            chosen = int(rng.choice(data.shape[0], p=nearest / total))
        centers[index] = data[chosen]
        nearest = np.minimum(nearest, np.sum((data - centers[index]) ** 2, axis=1))
    labels = np.zeros(data.shape[0], dtype=np.int64)
    for iteration in range(1, max_iterations + 1):
        distances = np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(distances, axis=1).astype(np.int64)
        new_centers = centers.copy()
        for cluster in range(cluster_count):
            members = data[new_labels == cluster]
            if len(members):
                new_centers[cluster] = members.mean(axis=0)
            else:
                farthest = int(np.argmax(np.min(distances, axis=1)))
                new_centers[cluster] = data[farthest]
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            labels = new_labels
            centers = new_centers
            return centers.astype(np.float32), labels, iteration
        labels = new_labels
        centers = new_centers
    return centers.astype(np.float32), labels, max_iterations


def nearest_centroid_ids(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    centers = np.asarray(centroids, dtype=np.float32)
    if data.ndim != 2 or centers.ndim != 2 or data.shape[1] != centers.shape[1]:
        raise ValueError("values and centroids must have matching feature dimensions")
    distances = np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1).astype(np.int64)
