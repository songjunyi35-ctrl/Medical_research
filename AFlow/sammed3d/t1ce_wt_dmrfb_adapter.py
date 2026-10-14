"""Single-bottleneck, same-parameter DMRFB adapter for T1ce/WT.

The historical SegResNet remains intact.  One training-and-deployment adapter
is inserted after its deepest encoder stage.  Both frozen arms have identical
parameter tensors and differ only in the dilation/padding metadata of branch 3.
The final projection is zero initialized, so either arm is initially exactly
the plain KD200 function while remaining trainable in two gradient steps.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
import copy
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator


SCHEMA = "aflow.sammed3d.t1ce_wt_dmrfb_adapter"
DMRFB_ARMS = ("same_rf_control", "dmrfb_dilation2")
ADAPTER_CONSTRUCTION_SEED = 20260901
FORMAL_CHANNELS = 128
FORMAL_ADAPTER_PARAMETERS = 57_088
FORMAL_DEPLOY_PARAMETERS = 4_757_985


@dataclass(frozen=True)
class DMRFBAdapterOutput:
    """The unchanged final raw foreground logit interface."""

    final_logits: torch.Tensor

    @property
    def segmentation_logits(self) -> torch.Tensor:
        return self.final_logits

    @property
    def center_logits(self) -> None:
        """Match the optional center-head interface of the plain locator."""

        return None


class DMRFBBottleneckAdapter(nn.Module):
    """Three-branch depthwise DMRFB followed by zero-init projection."""

    def __init__(self, *, channels: int = FORMAL_CHANNELS, arm: str) -> None:
        super().__init__()
        if arm not in DMRFB_ARMS:
            raise ValueError(f"unknown DMRFB arm: {arm!r}")
        if int(channels) <= 0:
            raise ValueError("channels must be positive")
        self.channels = int(channels)
        self.arm = arm
        branch3_dilation = 1 if arm == "same_rf_control" else 2
        self.branch3_dilation = branch3_dilation
        self.branch3_padding = branch3_dilation

        # The dedicated CPU fork makes every same-shaped state tensor exactly
        # equal across arms and leaves the paired caller RNG untouched.
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(ADAPTER_CONSTRUCTION_SEED)
            self.branches = nn.ModuleList(
                (
                    nn.Conv3d(
                        self.channels,
                        self.channels,
                        kernel_size=1,
                        dilation=1,
                        padding=0,
                        groups=self.channels,
                        bias=False,
                    ),
                    nn.Conv3d(
                        self.channels,
                        self.channels,
                        kernel_size=3,
                        dilation=1,
                        padding=1,
                        groups=self.channels,
                        bias=False,
                    ),
                    nn.Conv3d(
                        self.channels,
                        self.channels,
                        kernel_size=3,
                        dilation=self.branch3_dilation,
                        padding=self.branch3_padding,
                        groups=self.channels,
                        bias=False,
                    ),
                )
            )
            concatenated_channels = 3 * self.channels
            self.norm = nn.GroupNorm(
                num_groups=self.channels,
                num_channels=concatenated_channels,
                eps=1e-5,
                affine=True,
            )
            self.activation = nn.GELU(approximate="none")
            self.projection = nn.Conv3d(
                concatenated_channels,
                self.channels,
                kernel_size=1,
                bias=True,
            )
            nn.init.zeros_(self.projection.weight)
            assert self.projection.bias is not None
            nn.init.zeros_(self.projection.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 5 or features.shape[1] != self.channels:
            raise ValueError(
                f"features must have shape [B,{self.channels},D,H,W]"
            )
        if not features.is_floating_point():
            raise TypeError("features must be floating point")
        if features.shape[0] <= 0 or any(size <= 0 for size in features.shape[2:]):
            raise ValueError("feature dimensions must be positive")
        mixed = torch.cat(tuple(branch(features) for branch in self.branches), dim=1)
        residual = self.projection(self.activation(self.norm(mixed)))
        return features + residual


class DMRFBAdapterLocator(nn.Module):
    """Plain SegResNet plus one retained deepest-feature DMRFB adapter."""

    def __init__(
        self,
        *,
        arm: str,
        init_filters: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if arm not in DMRFB_ARMS:
            raise ValueError(f"unknown DMRFB arm: {arm!r}")
        if int(init_filters) <= 0:
            raise ValueError("init_filters must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        self.arm = arm
        self.init_filters = int(init_filters)
        self.dropout = float(dropout)

        plain = AutomaticPromptLocator(
            center_aware=False,
            init_filters=self.init_filters,
            dropout=self.dropout,
        )
        self.network = plain.network
        self.adapter = DMRFBBottleneckAdapter(
            channels=self.init_filters * 8,
            arm=self.arm,
        )

    @classmethod
    def from_plain_model(
        cls,
        plain: AutomaticPromptLocator,
        *,
        arm: str,
    ) -> "DMRFBAdapterLocator":
        if not isinstance(plain, AutomaticPromptLocator):
            raise TypeError("plain must be an AutomaticPromptLocator")
        if plain.center_aware:
            raise ValueError("DMRFB requires center_aware=False")
        model = cls(
            arm=arm,
            init_filters=plain.init_filters,
            dropout=plain.dropout,
        )
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
        arm: str,
        map_location: str | torch.device = "cpu",
    ) -> "DMRFBAdapterLocator":
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
            raise ValueError("checkpoint is not center_aware=False")
        model = cls(
            arm=arm,
            init_filters=int(architecture.get("init_filters", -1)),
            dropout=float(architecture.get("dropout", -1.0)),
        )
        expected_plain = OrderedDict(
            (f"network.{name}", tensor)
            for name, tensor in model.network.state_dict().items()
        )
        if tuple(state_dict) != tuple(expected_plain):
            missing = sorted(set(expected_plain) - set(state_dict))
            extra = sorted(set(state_dict) - set(expected_plain))
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
        if image.shape[0] <= 0 or any(
            size <= 0 or size % 8 != 0 for size in image.shape[2:]
        ):
            raise ValueError("spatial dimensions must be positive and divisible by8")

    def forward(self, image: torch.Tensor) -> DMRFBAdapterOutput:
        self._validate_image(image)
        encoded, down = self.network.encode(image)
        adapted = self.adapter(encoded)
        final = self.network.decode(adapted, list(reversed(down)))
        return DMRFBAdapterOutput(final_logits=final)

    def base_named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        for name, parameter in self.network.named_parameters():
            yield f"network.{name}", parameter

    def base_parameters(self) -> Iterator[nn.Parameter]:
        for _, parameter in self.base_named_parameters():
            yield parameter

    def adapter_named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        for name, parameter in self.adapter.named_parameters():
            yield f"adapter.{name}", parameter

    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        for _, parameter in self.adapter_named_parameters():
            yield parameter

    def deploy_state_dict(
        self, *, cpu: bool = False
    ) -> OrderedDict[str, torch.Tensor]:
        """Return the complete retained single-model deployment state."""

        result: OrderedDict[str, torch.Tensor] = OrderedDict()
        for name, value in self.state_dict().items():
            exported = value.detach().clone()
            if cpu:
                exported = exported.cpu()
            result[name] = exported
        return result

    def to_deploy_model(self) -> "DMRFBAdapterLocator":
        deploy = DMRFBAdapterLocator(
            arm=self.arm,
            init_filters=self.init_filters,
            dropout=self.dropout,
        )
        reference = next(self.parameters())
        deploy.to(device=reference.device, dtype=reference.dtype)
        deploy.load_state_dict(self.deploy_state_dict(), strict=True)
        deploy.train(self.training)
        return deploy

    @torch.inference_mode()
    def deploy_max_abs_difference(self, image: torch.Tensor) -> float:
        self._validate_image(image)
        was_training = self.training
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all() if image.is_cuda else None
        try:
            self.eval()
            deploy = self.to_deploy_model().eval()
            left = self(image).final_logits
            right = deploy(image).final_logits
            return float((left.float() - right.float()).abs().max().item())
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
            self.train(was_training)


__all__ = [
    "ADAPTER_CONSTRUCTION_SEED",
    "DMRFBAdapterLocator",
    "DMRFBAdapterOutput",
    "DMRFBBottleneckAdapter",
    "DMRFB_ARMS",
    "FORMAL_ADAPTER_PARAMETERS",
    "FORMAL_CHANNELS",
    "FORMAL_DEPLOY_PARAMETERS",
    "SCHEMA",
]
