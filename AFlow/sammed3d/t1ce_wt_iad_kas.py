"""Training-only IAD knowledge-assimilation stage for T1ce/WT SegResNet.

This is the preregistered binary 3D adaptation of the Knowledge Assimilation
Stage (KAS) from IAD.  A registered ``1x1x1`` projection maps the deepest
SegResNet encoder feature to one raw foreground logit and resizes it to the
ordinary segmentation-logit grid.  The training-only KAS loss couples those
two raw logits with MSE, soft-Dice consistency, and a per-volume contrastive
objective.  No teacher is used in this stage, and deployment strips the
projector and remains the unchanged plain :class:`AutomaticPromptLocator`.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
import copy
from dataclasses import dataclass
import hashlib
from os import PathLike
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator


SCHEMA = "aflow.sammed3d.t1ce_wt_iad_kas"
DEEPEST_CHANNEL_MULTIPLIER = 8
KAS_GAMMA = 0.01
KAS_TEMPERATURE = 1.0
KAS_DICE_EPSILON = 1e-6
COSINE_EPSILON = 1e-8
PROJECTOR_INITIALIZATION_SEED = 20260861
PROJECTOR_INITIAL_SHA256 = "61b7c474d78ce3ea275e66e88f0ee1d02840417064cdd46fa438b33523e95fe1"


def projector_parameter_sha256(projector: nn.Conv3d) -> str:
    """Hash contiguous float32 weight bytes followed by bias bytes."""

    if not isinstance(projector, nn.Conv3d):
        raise TypeError("projector must be Conv3d")
    if projector.bias is None:
        raise ValueError("the frozen KAS projector requires bias")
    digest = hashlib.sha256()
    for value in (projector.weight, projector.bias):
        tensor = value.detach().cpu().contiguous()
        if tensor.dtype != torch.float32:
            raise TypeError("the initial projector hash requires float32")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class KASTrainingOutput:
    """Raw deploy logit and the two tensors needed only for KAS training."""

    final_logits: torch.Tensor
    deepest_feature: torch.Tensor
    projected_logits: torch.Tensor

    def retain_kas_grads(self) -> "KASTrainingOutput":
        """Retain the two non-leaf gradients required by the smoke audit."""

        for name, value in (
            ("final_logits", self.final_logits),
            ("deepest_feature", self.deepest_feature),
        ):
            if not value.requires_grad:
                raise RuntimeError(f"{name} does not require gradient")
            value.retain_grad()
        return self


@dataclass(frozen=True)
class KASLossOutput:
    """Frozen equal-weight KAS loss and its three auditable components."""

    total: torch.Tensor
    mse: torch.Tensor
    dice: torch.Tensor
    contrastive: torch.Tensor


def _validate_raw_logits(value: torch.Tensor, *, name: str) -> None:
    if value.ndim != 5 or value.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B,1,D,H,W]")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if value.shape[0] <= 0 or any(size <= 0 for size in value.shape[-3:]):
        raise ValueError(f"{name} dimensions must be positive")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _per_volume_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return one numerically stable flattened cosine value per volume."""

    if left.shape != right.shape:
        raise ValueError("cosine inputs must have identical shapes")
    return F.cosine_similarity(
        left.float().flatten(1),
        right.float().flatten(1),
        dim=1,
        eps=COSINE_EPSILON,
    )


def kas_assimilation_loss(output: KASTrainingOutput) -> KASLossOutput:
    """Compute the frozen binary KAS objective without applying ``gamma``.

    Both the projected and final raw logits remain attached to the graph.
    The caller forms the formal objective as ``segmentation + 0.01 * total``.
    """

    if not isinstance(output, KASTrainingOutput):
        raise TypeError("output must be KASTrainingOutput")
    final = output.final_logits
    projected = output.projected_logits
    _validate_raw_logits(final, name="final_logits")
    _validate_raw_logits(projected, name="projected_logits")
    if final.shape != projected.shape:
        raise ValueError("final_logits and projected_logits must have identical shapes")

    final_float = final.float()
    projected_float = projected.float()
    mse = F.mse_loss(projected_float, final_float, reduction="mean")

    final_probability = torch.sigmoid(final_float)
    projected_probability = torch.sigmoid(projected_float)
    intersection = (final_probability * projected_probability).flatten(1).sum(dim=1)
    denominator = (
        final_probability.flatten(1).sum(dim=1)
        + projected_probability.flatten(1).sum(dim=1)
    )
    dice = (
        1.0
        - (2.0 * intersection + KAS_DICE_EPSILON)
        / (denominator + KAS_DICE_EPSILON)
    ).mean()

    negative = (1.0 - projected_probability) + (1.0 - final_probability)
    positive_similarity = _per_volume_cosine(final_float, projected_float)
    negative_similarity = _per_volume_cosine(final_float, negative)
    pair_logits = torch.stack(
        (
            positive_similarity / KAS_TEMPERATURE,
            negative_similarity / KAS_TEMPERATURE,
        ),
        dim=1,
    )
    contrastive = (
        torch.logsumexp(pair_logits, dim=1)
        - positive_similarity / KAS_TEMPERATURE
    ).mean()

    total = mse + dice + contrastive
    components = (mse, dice, contrastive, total)
    if any(value.ndim != 0 or not bool(torch.isfinite(value)) for value in components):
        raise RuntimeError("KAS produced a non-finite or non-scalar loss")
    return KASLossOutput(
        total=total,
        mse=mse,
        dice=dice,
        contrastive=contrastive,
    )


class KASTrainingModel(nn.Module):
    """Plain SegResNet plus one registered training-only KAS projector."""

    def __init__(self, *, init_filters: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        if init_filters <= 0:
            raise ValueError("init_filters must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        self.init_filters = int(init_filters)
        self.dropout = float(dropout)

        plain = AutomaticPromptLocator(
            center_aware=False,
            init_filters=self.init_filters,
            dropout=self.dropout,
        )
        self.network = plain.network
        # The amendment freezes projector initialization independently of all
        # model/training RNG.  Conv3d construction invokes default
        # reset_parameters (Kaiming uniform with a=sqrt(5), then the matching
        # fan-in bias range) inside a CPU-only RNG fork.
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(PROJECTOR_INITIALIZATION_SEED)
            self.projector = nn.Conv3d(
                self.init_filters * DEEPEST_CHANNEL_MULTIPLIER,
                1,
                kernel_size=1,
                bias=True,
            )
        if self.projector.in_channels == 128:
            observed = projector_parameter_sha256(self.projector)
            if observed != PROJECTOR_INITIAL_SHA256:
                raise RuntimeError(
                    f"frozen KAS projector hash changed: {observed}"
                )

    @classmethod
    def from_plain_model(cls, plain: AutomaticPromptLocator) -> "KASTrainingModel":
        """Copy a plain locator into the KAS wrapper without storage aliasing."""

        if not isinstance(plain, AutomaticPromptLocator):
            raise TypeError("plain must be an AutomaticPromptLocator")
        if plain.center_aware:
            raise ValueError("KAS requires a plain center_aware=False locator")
        model = cls(init_filters=plain.init_filters, dropout=plain.dropout)
        model.network.load_state_dict(
            copy.deepcopy(plain.network.state_dict()), strict=True
        )
        reference = next(plain.parameters())
        model.to(device=reference.device, dtype=reference.dtype)
        model.train(plain.training)
        return model

    @classmethod
    def from_plain_checkpoint(
        cls,
        checkpoint: Mapping[str, Any] | str | PathLike[str],
        *,
        map_location: str | torch.device = "cpu",
    ) -> "KASTrainingModel":
        """Strictly load a plain deploy checkpoint into a new KAS wrapper."""

        if isinstance(checkpoint, (str, PathLike)):
            value = torch.load(
                Path(checkpoint), map_location=map_location, weights_only=False
            )
        elif isinstance(checkpoint, Mapping):
            value = checkpoint
        else:
            raise TypeError("checkpoint must be a mapping or filesystem path")
        architecture = value.get("architecture")
        state_dict = value.get("model_state_dict")
        if not isinstance(architecture, Mapping):
            raise ValueError("plain checkpoint is missing architecture metadata")
        if not isinstance(state_dict, Mapping):
            raise ValueError("plain checkpoint is missing model_state_dict")
        if architecture.get("class") != "AutomaticPromptLocator":
            raise ValueError("checkpoint is not an AutomaticPromptLocator")
        if architecture.get("center_aware") is not False:
            raise ValueError("checkpoint is not a plain center_aware=False locator")

        model = cls(
            init_filters=int(architecture.get("init_filters", -1)),
            dropout=float(architecture.get("dropout", -1.0)),
        )
        expected = model.deploy_state_dict()
        if tuple(state_dict) != tuple(expected):
            missing = sorted(set(expected) - set(state_dict))
            extra = sorted(set(state_dict) - set(expected))
            raise RuntimeError(
                f"plain checkpoint key mismatch: missing={missing}, extra={extra}"
            )
        network_state = OrderedDict(
            (name.removeprefix("network."), tensor)
            for name, tensor in state_dict.items()
        )
        model.network.load_state_dict(network_state, strict=True)
        return model

    @staticmethod
    def _validate_image(image: torch.Tensor) -> None:
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError("image must have shape [B,1,D,H,W]")
        if not image.is_floating_point():
            raise TypeError("image must be floating point")
        if any(size <= 0 or size % 8 != 0 for size in image.shape[-3:]):
            raise ValueError(
                "each image spatial dimension must be positive and divisible by 8"
            )

    def forward(self, image: torch.Tensor) -> KASTrainingOutput:
        self._validate_image(image)
        encoded, down = self.network.encode(image)
        if len(down) != 4:
            raise RuntimeError("frozen SegResNet must expose exactly four encoder stages")
        deepest = down[-1]
        expected_channels = self.init_filters * DEEPEST_CHANNEL_MULTIPLIER
        if deepest.shape[1] != expected_channels:
            raise RuntimeError(
                f"unexpected deepest channels {deepest.shape[1]}; expected {expected_channels}"
            )
        expected_spatial = tuple(size // 8 for size in image.shape[-3:])
        if tuple(deepest.shape[-3:]) != expected_spatial:
            raise RuntimeError(
                f"unexpected deepest shape {tuple(deepest.shape[-3:])}; "
                f"expected {expected_spatial}"
            )

        final = self.network.decode(encoded, list(reversed(down)))
        projected_native = self.projector(deepest)
        projected = F.interpolate(
            projected_native,
            size=tuple(final.shape[-3:]),
            mode="trilinear",
            align_corners=False,
        )
        if projected.shape != final.shape:
            raise RuntimeError("KAS projected logits do not match final-logit shape")
        return KASTrainingOutput(
            final_logits=final,
            deepest_feature=deepest,
            projected_logits=projected,
        )

    def kas_loss(self, output: KASTrainingOutput) -> KASLossOutput:
        """Compute the frozen equal-weight KAS objective."""

        return kas_assimilation_loss(output)

    def training_only_named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        """Yield the projector parameters removed from the deploy model."""

        for name, parameter in self.projector.named_parameters():
            yield f"projector.{name}", parameter

    def training_only_parameters(self) -> Iterator[nn.Parameter]:
        for _, parameter in self.training_only_named_parameters():
            yield parameter

    def deploy_state_dict(self, *, cpu: bool = False) -> OrderedDict[str, torch.Tensor]:
        """Return only the state strict-loadable by the plain locator."""

        result: OrderedDict[str, torch.Tensor] = OrderedDict()
        for name, value in self.network.state_dict().items():
            exported = value.detach().clone()
            if cpu:
                exported = exported.cpu()
            result[f"network.{name}"] = exported
        return result

    def to_deploy_model(self) -> AutomaticPromptLocator:
        """Build the exact projector-free locator used at inference."""

        deploy = AutomaticPromptLocator(
            center_aware=False,
            init_filters=self.init_filters,
            dropout=self.dropout,
        )
        reference = next(self.network.parameters())
        deploy.to(device=reference.device, dtype=reference.dtype)
        deploy.load_state_dict(self.deploy_state_dict(), strict=True)
        deploy.train(self.training)
        return deploy

    @torch.inference_mode()
    def deploy_max_abs_difference(self, image: torch.Tensor) -> float:
        """Audit exact eval-logit equivalence while preserving every RNG stream."""

        self._validate_image(image)
        was_training = self.training
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all() if image.is_cuda else None
        try:
            self.eval()
            deploy = self.to_deploy_model().eval()
            wrapper_logits = self(image).final_logits
            deploy_logits = deploy(image).segmentation_logits
            return float(
                (wrapper_logits.float() - deploy_logits.float()).abs().max().item()
            )
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
            self.train(was_training)


__all__ = [
    "COSINE_EPSILON",
    "DEEPEST_CHANNEL_MULTIPLIER",
    "KAS_DICE_EPSILON",
    "KAS_GAMMA",
    "KAS_TEMPERATURE",
    "KASLossOutput",
    "KASTrainingModel",
    "KASTrainingOutput",
    "SCHEMA",
    "PROJECTOR_INITIALIZATION_SEED",
    "PROJECTOR_INITIAL_SHA256",
    "kas_assimilation_loss",
    "projector_parameter_sha256",
]
