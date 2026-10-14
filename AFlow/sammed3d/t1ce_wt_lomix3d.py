"""Training-only three-stage LoMix adaptation for the plain T1ce/WT locator.

The deployment path remains the historical :class:`AutomaticPromptLocator`.
This module exposes the three real SegResNet decoder stages during training,
builds the preregistered 19 raw-logit supervision maps, and can export a state
dict containing only the unchanged plain locator.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
import copy
from os import PathLike
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator


SCHEMA = "aflow.sammed3d.t1ce_wt_lomix3d"
DECODER_RESOLUTIONS = (32, 64, 128)
STAGE_NAMES = tuple(f"s{resolution}" for resolution in DECODER_RESOLUTIONS)
SUBSETS = ((0, 1), (0, 2), (1, 2), (0, 1, 2))
OPERATORS = ("add", "multiply", "concat", "awf")
ORIGINAL_MAP_NAMES = tuple(f"original_{resolution}" for resolution in DECODER_RESOLUTIONS)


def _subset_name(subset: tuple[int, ...]) -> str:
    return "_".join(STAGE_NAMES[index] for index in subset)


MUTANT_MAP_NAMES = tuple(
    f"{operator}_{_subset_name(subset)}"
    for subset in SUBSETS
    for operator in OPERATORS
)
MAP_NAMES = ORIGINAL_MAP_NAMES + MUTANT_MAP_NAMES

if len(MAP_NAMES) != 19 or len(set(MAP_NAMES)) != 19:
    raise RuntimeError("the frozen LoMix3D map catalog must contain 19 unique maps")


@dataclass(frozen=True)
class LoMix3DTrainingOutput:
    """All raw-logit maps produced by a training forward pass.

    ``native_original_logits`` retains the real decoder resolutions.
    ``original_logits`` contains the same three maps resized to the final
    output shape, as required by the frozen loss.  Its last item and
    ``final_logits`` are the exact original SegResNet final-head tensor.
    """

    final_logits: torch.Tensor
    original_logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    mutant_logits: OrderedDict[str, torch.Tensor]
    native_original_logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    def all_logits(self) -> OrderedDict[str, torch.Tensor]:
        maps: OrderedDict[str, torch.Tensor] = OrderedDict(
            zip(ORIGINAL_MAP_NAMES, self.original_logits, strict=True)
        )
        maps.update(self.mutant_logits)
        if tuple(maps) != MAP_NAMES:
            raise RuntimeError("LoMix3D output map order differs from the frozen catalog")
        return maps


@dataclass(frozen=True)
class LoMix3DLossOutput:
    """Weighted total loss and its auditable 19-map decomposition."""

    total: torch.Tensor
    per_map: OrderedDict[str, torch.Tensor]
    effective_weights: OrderedDict[str, torch.Tensor]


class SubsetAttentionWeightedFusion(nn.Module):
    """Subset-specific voxelwise attention over raw-logit input maps."""

    def __init__(self, map_count: int) -> None:
        super().__init__()
        if map_count < 2:
            raise ValueError("attention fusion requires at least two maps")
        self.map_count = int(map_count)
        self.attention_logits = nn.Conv3d(
            self.map_count, self.map_count, kernel_size=1, bias=True
        )

    def forward(self, maps: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(maps) != self.map_count:
            raise ValueError("attention fusion received an unexpected map count")
        reference_shape = maps[0].shape
        if any(value.shape != reference_shape for value in maps):
            raise ValueError("attention fusion maps must have identical shapes")
        if len(reference_shape) != 5 or reference_shape[1] != 1:
            raise ValueError("attention fusion maps must have shape [B,1,D,H,W]")
        stacked = torch.cat(maps, dim=1)
        weights = torch.softmax(self.attention_logits(stacked), dim=1)
        return (weights * stacked).sum(dim=1, keepdim=True)


class LoMix3DTrainingModel(nn.Module):
    """Plain SegResNet plus preregistered training-only LoMix3D modules."""

    def __init__(self, *, init_filters: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        if init_filters <= 0:
            raise ValueError("init_filters must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        self.init_filters = int(init_filters)
        self.dropout = float(dropout)

        # Retain the exact attribute name and module structure used by
        # AutomaticPromptLocator, so deploy_state_dict is strict-loadable.
        plain = AutomaticPromptLocator(
            center_aware=False,
            init_filters=self.init_filters,
            dropout=self.dropout,
        )
        self.network = plain.network

        # The final 128-stage prediction uses network.conv_final unchanged.
        # Only the two coarse real decoder stages need auxiliary heads.
        self.auxiliary_heads = nn.ModuleDict(
            {
                "s32": nn.Conv3d(self.init_filters * 4, 1, kernel_size=1),
                "s64": nn.Conv3d(self.init_filters * 2, 1, kernel_size=1),
            }
        )
        self.concat_fusions = nn.ModuleDict()
        self.attention_fusions = nn.ModuleDict()
        for subset in SUBSETS:
            name = _subset_name(subset)
            self.concat_fusions[name] = nn.Conv3d(
                len(subset), 1, kernel_size=1, bias=True
            )
            self.attention_fusions[name] = SubsetAttentionWeightedFusion(
                len(subset)
            )
        # ParameterDict sorts keys from a plain dict.  An OrderedDict keeps the
        # frozen 19-map order available to reports and gradient audits.
        self.loss_alphas = nn.ParameterDict(
            OrderedDict(
                (name, nn.Parameter(torch.zeros(()))) for name in MAP_NAMES
            )
        )

    @classmethod
    def from_plain_model(
        cls, plain: AutomaticPromptLocator
    ) -> "LoMix3DTrainingModel":
        """Copy a plain locator into a new LoMix training wrapper, strictly."""

        if not isinstance(plain, AutomaticPromptLocator):
            raise TypeError("plain must be an AutomaticPromptLocator")
        if plain.center_aware:
            raise ValueError("LoMix3D requires a plain center_aware=False locator")
        model = cls(init_filters=plain.init_filters, dropout=plain.dropout)
        # deepcopy prevents later mutations of either model from aliasing the
        # other model's storage, while strict=True audits every backbone key.
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
    ) -> "LoMix3DTrainingModel":
        """Strictly load a historical plain checkpoint or checkpoint mapping."""

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
        init_filters = int(architecture.get("init_filters", -1))
        dropout = float(architecture.get("dropout", -1.0))
        model = cls(init_filters=init_filters, dropout=dropout)
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
        # A shape mismatch must still fail strictly.  Constructing only this
        # wrapper also freezes the preregistered auxiliary-module RNG order.
        model.network.load_state_dict(network_state, strict=True)
        return model

    @staticmethod
    def _validate_image(image: torch.Tensor) -> None:
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError("image must have shape [B,1,D,H,W]")
        if not image.is_floating_point():
            raise TypeError("image must be floating point")
        if any(size <= 0 or size % 8 != 0 for size in image.shape[-3:]):
            raise ValueError("each image spatial dimension must be positive and divisible by 8")

    def _decoder_features(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded, down = self.network.encode(image)
        skips = list(reversed(down))
        decoded = encoded
        features: list[torch.Tensor] = []
        for index, (up_sample, up_layer) in enumerate(
            zip(self.network.up_samples, self.network.up_layers, strict=True)
        ):
            decoded = up_sample(decoded) + skips[index + 1]
            decoded = up_layer(decoded)
            features.append(decoded)
        if len(features) != 3:
            raise RuntimeError("frozen SegResNet must expose exactly three decoder stages")
        expected_shapes = tuple(
            tuple(size // divisor for size in image.shape[-3:])
            for divisor in (4, 2, 1)
        )
        observed_shapes = tuple(tuple(value.shape[-3:]) for value in features)
        if observed_shapes != expected_shapes:
            raise RuntimeError(
                f"unexpected decoder shapes {observed_shapes}; expected {expected_shapes}"
            )
        return features[0], features[1], features[2]

    def forward(self, image: torch.Tensor) -> LoMix3DTrainingOutput:
        self._validate_image(image)
        feature32, feature64, feature128 = self._decoder_features(image)
        native32 = self.auxiliary_heads["s32"](feature32)
        native64 = self.auxiliary_heads["s64"](feature64)
        final = self.network.conv_final(feature128)
        final_size = tuple(final.shape[-3:])
        original = (
            F.interpolate(
                native32, size=final_size, mode="trilinear", align_corners=False
            ),
            F.interpolate(
                native64, size=final_size, mode="trilinear", align_corners=False
            ),
            final,
        )

        mutants: OrderedDict[str, torch.Tensor] = OrderedDict()
        for subset in SUBSETS:
            subset_maps = tuple(original[index] for index in subset)
            subset_name = _subset_name(subset)
            mutants[f"add_{subset_name}"] = torch.stack(subset_maps, dim=0).sum(dim=0)
            mutants[f"multiply_{subset_name}"] = torch.stack(subset_maps, dim=0).prod(dim=0)
            concatenated = torch.cat(subset_maps, dim=1)
            mutants[f"concat_{subset_name}"] = self.concat_fusions[subset_name](
                concatenated
            )
            mutants[f"awf_{subset_name}"] = self.attention_fusions[subset_name](
                subset_maps
            )
        if tuple(mutants) != MUTANT_MAP_NAMES:
            raise RuntimeError("mutant map order differs from the frozen catalog")
        return LoMix3DTrainingOutput(
            final_logits=final,
            original_logits=original,
            mutant_logits=mutants,
            native_original_logits=(native32, native64, final),
        )

    def lomix_loss(
        self,
        output: LoMix3DTrainingOutput,
        label: torch.Tensor,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> LoMix3DLossOutput:
        """Apply independently weighted segmentation loss to all 19 maps."""

        if not isinstance(output, LoMix3DTrainingOutput):
            raise TypeError("output must be LoMix3DTrainingOutput")
        if label.shape != output.final_logits.shape:
            raise ValueError("label shape must match final_logits")
        if not label.is_floating_point():
            raise TypeError("label must be floating point")
        maps = output.all_logits()
        per_map: OrderedDict[str, torch.Tensor] = OrderedDict()
        effective_weights: OrderedDict[str, torch.Tensor] = OrderedDict()
        weighted: list[torch.Tensor] = []
        for name, logits in maps.items():
            value = loss_function(logits, label)
            if value.ndim != 0:
                raise ValueError("per-map loss function must return a scalar")
            weight = F.softplus(self.loss_alphas[name])
            per_map[name] = value
            effective_weights[name] = weight
            weighted.append(weight * value)
        total = torch.stack(weighted).sum()
        return LoMix3DLossOutput(
            total=total,
            per_map=per_map,
            effective_weights=effective_weights,
        )

    def training_only_named_parameters(
        self,
    ) -> Iterator[tuple[str, nn.Parameter]]:
        """Yield every parameter that is removed from the deploy-only model."""

        groups: tuple[tuple[str, nn.Module], ...] = (
            ("auxiliary_heads", self.auxiliary_heads),
            ("concat_fusions", self.concat_fusions),
            ("attention_fusions", self.attention_fusions),
            ("loss_alphas", self.loss_alphas),
        )
        for prefix, module in groups:
            for name, parameter in module.named_parameters():
                yield f"{prefix}.{name}", parameter

    def training_only_parameters(self) -> Iterator[nn.Parameter]:
        for _, parameter in self.training_only_named_parameters():
            yield parameter

    def deploy_state_dict(self, *, cpu: bool = False) -> OrderedDict[str, torch.Tensor]:
        """Return only strict-loadable plain-locator parameters and buffers."""

        result: OrderedDict[str, torch.Tensor] = OrderedDict()
        for name, value in self.network.state_dict().items():
            exported = value.detach().clone()
            if cpu:
                exported = exported.cpu()
            result[f"network.{name}"] = exported
        return result

    def to_deploy_model(self) -> AutomaticPromptLocator:
        """Build the exact one-forward plain locator used at inference."""

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
        """Audit exact eval-logit equivalence before and after stripping LoMix."""

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
            # Constructing the temporary deploy model initializes parameters
            # before strict loading.  Restore RNG so checkpoint audits cannot
            # perturb the next epoch's matched-control dropout stream.
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
            self.train(was_training)
