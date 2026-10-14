from __future__ import annotations

import torch

from scripts.bridge_residual_refiner_checkpoint import state_dict_sha256


def test_state_dict_digest_is_stable_and_tensor_sensitive() -> None:
    first = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}
    reordered = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
    changed = {"a": torch.tensor([1.0]), "b": torch.tensor([3.0])}
    assert state_dict_sha256(first) == state_dict_sha256(reordered)
    assert state_dict_sha256(first) != state_dict_sha256(changed)
