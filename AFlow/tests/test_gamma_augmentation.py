from __future__ import annotations

import pytest
import torch

from scripts.gamma_augmentation import RetainedStatsGamma


def test_gamma_is_deterministic_and_preserves_statistics_and_label() -> None:
    image = torch.linspace(-2, 3, 2 * 1 * 4 * 4 * 4).reshape(2, 1, 4, 4, 4)
    label = (image > 0).float()
    first = RetainedStatsGamma(torch.device("cpu"), seed=17, probability=0.3)
    second = RetainedStatsGamma(torch.device("cpu"), seed=17, probability=0.3)
    # Force enough samples that the frozen p=0.3 stream applies at least one.
    image = image.repeat(8, 1, 1, 1, 1)
    label = label.repeat(8, 1, 1, 1, 1)
    output_a, target_a = first(image, label)
    output_b, target_b = second(image, label)
    assert torch.equal(output_a, output_b)
    assert torch.equal(target_a, label) and torch.equal(target_b, label)
    changed = (output_a != image).flatten(1).any(dim=1)
    assert bool(changed.any())
    assert torch.allclose(output_a[changed].mean(dim=(1, 2, 3, 4)), image[changed].mean(dim=(1, 2, 3, 4)), atol=1e-5)
    assert torch.allclose(output_a[changed].std(dim=(1, 2, 3, 4), unbiased=False), image[changed].std(dim=(1, 2, 3, 4), unbiased=False), atol=1e-5)


def test_gamma_uses_dedicated_rng_and_valid_range() -> None:
    image = torch.linspace(-1, 1, 16 * 1 * 3 * 3 * 3).reshape(16, 1, 3, 3, 3)
    label = torch.zeros_like(image)
    state = torch.random.get_rng_state().clone()
    transform = RetainedStatsGamma(torch.device("cpu"), seed=23)
    transform(image, label)
    assert torch.equal(state, torch.random.get_rng_state())
    audit = transform.audit()
    assert audit["applied_samples"] > 0
    assert 0.7 <= audit["gamma_min"] < 1.5
    assert 0.7 < audit["gamma_max"] <= 1.5
    assert audit["gamma_min"] != 1.0 and audit["gamma_max"] != 1.0


def test_gamma_rejects_bad_shape_and_constant_selected_input() -> None:
    transform = RetainedStatsGamma(torch.device("cpu"), seed=1)
    with pytest.raises(ValueError, match="Bx1"):
        transform(torch.zeros(1, 2, 2, 2, 2), torch.zeros(1, 2, 2, 2, 2))
    constant = torch.ones(32, 1, 2, 2, 2)
    with pytest.raises(ValueError, match="nonconstant"):
        transform(constant, torch.zeros_like(constant))


def test_gamma_rejects_nonfrozen_probability() -> None:
    with pytest.raises(ValueError, match="0.3"):
        RetainedStatsGamma(torch.device("cpu"), seed=1, probability=0.2)
