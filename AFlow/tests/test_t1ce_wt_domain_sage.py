from __future__ import annotations

import numpy as np
import torch

from sammed3d.t1ce_wt_domain_sage import (
    DomainSAGESafeAdapter,
    LowRankDomainFiLM,
    fit_kmeans_plus_plus,
    frozen_image_descriptor,
    nearest_centroid_ids,
)
from sammed3d.t1ce_wt_safe_adapter import FrozenBaseSafeAdapter
from scripts.train_sammed3d_t1ce_wt_residual_refiner import ResidualRefiner


def test_low_rank_domain_film_is_exact_identity_at_initialization() -> None:
    layer = LowRankDomainFiLM(domain_count=4, channels=8, rank=2)
    features = torch.randn(3, 8, 2, 2, 2)
    affinities = torch.softmax(torch.randn(3, 4), dim=1)
    output, magnitude = layer(features, affinities)
    assert torch.equal(output, features)
    assert torch.equal(magnitude, torch.zeros_like(magnitude))


def test_domain_film_changes_with_domain_after_coefficients_move() -> None:
    layer = LowRankDomainFiLM(domain_count=4, channels=4, rank=2)
    with torch.no_grad():
        layer.gamma_coeff[0, 0] = 1.0
        layer.gamma_coeff[1, 0] = -1.0
        layer.gamma_basis[0].fill_(0.5)
    features = torch.ones(2, 4, 2, 2, 2)
    affinities = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    output, _ = layer(features, affinities)
    assert not torch.equal(output[0], output[1])


def test_domain_sage_identity_path_matches_plain_sage() -> None:
    torch.manual_seed(19)
    base = ResidualRefiner(init_filters=8, dropout=0.0)
    plain = FrozenBaseSafeAdapter(
        base, mode="plain", init_filters=8, dropout=0.0
    ).eval()
    domain = DomainSAGESafeAdapter(
        ResidualRefiner(init_filters=8, dropout=0.0),
        descriptor_dim=16,
        domain_count=4,
        domain_hidden_dim=8,
        film_rank=2,
        init_filters=8,
        dropout=0.0,
    ).eval()
    domain.base.load_state_dict(plain.base.state_dict(), strict=True)
    domain.adapter.load_state_dict(plain.adapter.state_dict(), strict=True)
    image = torch.randn(1, 1, 16, 16, 16)
    coarse = torch.randn_like(image)
    descriptor = torch.randn(1, 16)
    with torch.inference_mode():
        plain_output = plain.forward_with_aux(image, coarse)
        domain_output = domain.forward_with_aux(image, coarse, descriptor)
    assert torch.equal(domain_output.base_logits, plain_output.base_logits)
    assert torch.equal(domain_output.residual, plain_output.residual)
    assert torch.equal(domain_output.logits, plain_output.logits)


def test_frozen_image_descriptor_has_no_coarse_or_label_input() -> None:
    base = ResidualRefiner(init_filters=8, dropout=0.0).eval()
    image = torch.randn(2, 1, 16, 16, 16)
    descriptor = frozen_image_descriptor(base, image)
    assert descriptor.shape == (2, 64)
    assert torch.isfinite(descriptor).all()


def test_kmeans_plus_plus_is_deterministic_and_assigns_every_cluster() -> None:
    rng = np.random.default_rng(7)
    values = np.concatenate(
        [rng.normal(loc=float(index) * 5.0, scale=0.1, size=(8, 3)) for index in range(4)]
    ).astype(np.float32)
    centers_a, labels_a, iterations_a = fit_kmeans_plus_plus(
        values, cluster_count=4, seed=23
    )
    centers_b, labels_b, iterations_b = fit_kmeans_plus_plus(
        values, cluster_count=4, seed=23
    )
    assert np.array_equal(centers_a, centers_b)
    assert np.array_equal(labels_a, labels_b)
    assert iterations_a == iterations_b
    assert np.all(np.bincount(labels_a, minlength=4) > 0)
    assert np.array_equal(nearest_centroid_ids(values, centers_a), labels_a)
