"""Training-only DCD detail distillation for the plain T1ce/WT locator.

The implementation follows the frozen AFlow adaptation of Detail Consistent
Distillation (DCD): four SegResNet encoder stages, a three-level 3D db4
transform, and directional bands 1--6.  PyTorch ``conv3d`` is a
cross-correlation and ``conv_transpose3d`` is its adjoint, so the same frozen
decomposition bank is deliberately used for analysis and synthesis.  The
wavelet path is training-only; deployment remains the unchanged plain locator.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import copy
import hashlib
from os import PathLike
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator


SCHEMA = "aflow.sammed3d.t1ce_wt_dcd"
ENCODER_CHANNELS = (16, 32, 64, 128)
ENCODER_RESOLUTIONS = (128, 64, 32, 16)
SUBBAND_NAMES = ("LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH")
SELECTED_BANDS = (1, 2, 3, 4, 5, 6)
LEVELS = 3
FILTER_LENGTH = 8
PADDING = FILTER_LENGTH - 1
STRIDE = 2

# Canonical PyWavelets db4 decomposition taps.  Values are converted to
# float32 before the separable 3D outer products, matching the frozen hash.
DB4_DEC_LO = (
    -0.010597401785069032,
    0.0328830116668852,
    0.030841381835560764,
    -0.18703481171909309,
    -0.027983769416859854,
    0.6308807679298587,
    0.7148465705529157,
    0.2303778133088965,
)
DB4_DEC_HI = (
    -0.2303778133088965,
    0.7148465705529157,
    -0.6308807679298587,
    -0.027983769416859854,
    0.18703481171909309,
    0.030841381835560764,
    -0.0328830116668852,
    -0.010597401785069032,
)
DB4_FILTER_BANK_SHA256 = "1cc1550c3ca56547e427b3568d2d1e2bc5bd52d2a4a39305b7890dc16918240a"


def build_db4_filter_bank_3d() -> torch.Tensor:
    """Return the frozen float32 ``[8,1,8,8,8]`` db4 analysis bank.

    Loop order is z, y, x with low before high on each axis, giving exactly
    ``LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH``.
    """

    low = torch.tensor(DB4_DEC_LO, dtype=torch.float32)
    high = torch.tensor(DB4_DEC_HI, dtype=torch.float32)
    kernels = [
        z[:, None, None] * y[None, :, None] * x[None, None, :]
        for z in (low, high)
        for y in (low, high)
        for x in (low, high)
    ]
    bank = torch.stack(kernels, dim=0).unsqueeze(1).contiguous()
    observed = hashlib.sha256(bank.numpy().tobytes()).hexdigest()
    if observed != DB4_FILTER_BANK_SHA256:
        raise RuntimeError(
            f"canonical db4 filter bank hash changed: {observed}"
        )
    return bank


def _validate_features(features: torch.Tensor, *, name: str) -> None:
    if features.ndim != 5:
        raise ValueError(f"{name} must have shape [B,C,D,H,W]")
    if not features.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if features.shape[0] <= 0 or features.shape[1] <= 0:
        raise ValueError(f"{name} batch and channel dimensions must be positive")
    if any(size <= 0 for size in features.shape[-3:]):
        raise ValueError(f"{name} spatial dimensions must be positive")


def _validate_filter_bank(filter_bank: torch.Tensor) -> None:
    if tuple(filter_bank.shape) != (8, 1, FILTER_LENGTH, FILTER_LENGTH, FILTER_LENGTH):
        raise ValueError("filter_bank must have shape [8,1,8,8,8]")
    if not filter_bank.is_floating_point():
        raise TypeError("filter_bank must be floating point")


def dwt3(features: torch.Tensor, filter_bank: torch.Tensor) -> torch.Tensor:
    """Apply the frozen per-channel 3D wavelet analysis transform."""

    _validate_features(features, name="features")
    _validate_filter_bank(filter_bank)
    batch, channels, depth, height, width = features.shape
    filters = filter_bank.to(device=features.device, dtype=features.dtype)
    flattened = features.reshape(batch * channels, 1, depth, height, width)
    coefficients = F.conv3d(
        flattened, filters, stride=STRIDE, padding=PADDING
    )
    return coefficients.reshape(
        batch,
        channels,
        len(SUBBAND_NAMES),
        *coefficients.shape[-3:],
    )


def idwt3(
    coefficients: torch.Tensor,
    filter_bank: torch.Tensor,
    output_shape: Sequence[int],
) -> torch.Tensor:
    """Apply the adjoint 3D transform and crop exactly to ``output_shape``."""

    if coefficients.ndim != 6 or coefficients.shape[2] != len(SUBBAND_NAMES):
        raise ValueError("coefficients must have shape [B,C,8,D,H,W]")
    if not coefficients.is_floating_point():
        raise TypeError("coefficients must be floating point")
    _validate_filter_bank(filter_bank)
    shape = tuple(int(value) for value in output_shape)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("output_shape must contain three positive integers")
    batch, channels, _, depth, height, width = coefficients.shape
    filters = filter_bank.to(device=coefficients.device, dtype=coefficients.dtype)
    flattened = coefficients.reshape(
        batch * channels, len(SUBBAND_NAMES), depth, height, width
    )
    reconstruction = F.conv_transpose3d(
        flattened, filters, stride=STRIDE, padding=PADDING
    )
    if any(observed < expected for observed, expected in zip(reconstruction.shape[-3:], shape)):
        raise RuntimeError(
            "adjoint reconstruction is smaller than the requested output shape; "
            "the frozen DCD path requires even level inputs"
        )
    reconstruction = reconstruction[..., : shape[0], : shape[1], : shape[2]]
    if tuple(reconstruction.shape[-3:]) != shape:
        raise RuntimeError("DCD reconstruction crop did not recover the level input shape")
    return reconstruction.reshape(batch, channels, *shape)


@dataclass(frozen=True)
class DCDStageLossOutput:
    """One stage's level-mean loss and auditable per-level values."""

    total: torch.Tensor
    level_losses: tuple[torch.Tensor, ...]


def multilevel_detail_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    filter_bank: torch.Tensor,
    *,
    levels: int = LEVELS,
    selected_bands: Sequence[int] = SELECTED_BANDS,
) -> DCDStageLossOutput:
    """Return the mean spatial-detail MSE over recursive db4 levels."""

    _validate_features(student_features, name="student_features")
    _validate_features(teacher_features, name="teacher_features")
    if student_features.shape != teacher_features.shape:
        raise ValueError("student and teacher feature shapes must match exactly")
    if teacher_features.requires_grad:
        raise ValueError("detail teacher features must be frozen/no-grad")
    if levels < 1:
        raise ValueError("levels must be positive")
    bands = tuple(sorted(set(int(index) for index in selected_bands)))
    if not bands or any(index < 0 or index >= len(SUBBAND_NAMES) for index in bands):
        raise ValueError("selected_bands must contain valid indices in [0,7]")

    mask = student_features.new_zeros((1, 1, len(SUBBAND_NAMES), 1, 1, 1))
    mask[:, :, bands] = 1
    current_student = student_features
    current_teacher = teacher_features
    losses: list[torch.Tensor] = []
    for _ in range(levels):
        input_shape = tuple(current_student.shape[-3:])
        student_coefficients = dwt3(current_student, filter_bank)
        teacher_coefficients = dwt3(current_teacher, filter_bank)
        student_detail = idwt3(
            student_coefficients * mask, filter_bank, input_shape
        )
        teacher_detail = idwt3(
            teacher_coefficients * mask, filter_bank, input_shape
        )
        loss = F.mse_loss(student_detail.float(), teacher_detail.float())
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise RuntimeError("DCD produced a non-finite or non-scalar level loss")
        losses.append(loss)
        current_student = student_coefficients[:, :, 0]
        current_teacher = teacher_coefficients[:, :, 0]
    total = torch.stack(losses).mean()
    return DCDStageLossOutput(total=total, level_losses=tuple(losses))


@dataclass(frozen=True)
class DCDLossOutput:
    """Raw four-stage DCD sum and its complete decomposition."""

    total: torch.Tensor
    stage_losses: tuple[torch.Tensor, ...]
    level_losses: tuple[tuple[torch.Tensor, ...], ...]


class DCDDetailLoss(nn.Module):
    """Frozen four-stage, three-level directional-detail loss."""

    def __init__(
        self,
        *,
        levels: int = LEVELS,
        selected_bands: Sequence[int] = SELECTED_BANDS,
    ) -> None:
        super().__init__()
        if levels != LEVELS:
            raise ValueError(f"frozen DCD protocol requires exactly {LEVELS} levels")
        bands = tuple(int(value) for value in selected_bands)
        if bands != SELECTED_BANDS:
            raise ValueError(f"frozen DCD protocol requires bands {SELECTED_BANDS}")
        self.levels = levels
        self.selected_bands = bands
        self.register_buffer("filter_bank", build_db4_filter_bank_3d(), persistent=True)

    def forward(
        self,
        student_features: Sequence[torch.Tensor],
        teacher_features: Sequence[torch.Tensor],
    ) -> DCDLossOutput:
        students = tuple(student_features)
        teachers = tuple(teacher_features)
        if len(students) != len(ENCODER_CHANNELS) or len(teachers) != len(ENCODER_CHANNELS):
            raise ValueError("DCD requires exactly four student and teacher encoder stages")
        stage_outputs = tuple(
            multilevel_detail_loss(
                student,
                teacher,
                self.filter_bank,
                levels=self.levels,
                selected_bands=self.selected_bands,
            )
            for student, teacher in zip(students, teachers, strict=True)
        )
        stage_losses = tuple(value.total for value in stage_outputs)
        total = torch.stack(stage_losses).sum()
        if not bool(torch.isfinite(total)):
            raise RuntimeError("DCD produced a non-finite raw stage sum")
        return DCDLossOutput(
            total=total,
            stage_losses=stage_losses,
            level_losses=tuple(value.level_losses for value in stage_outputs),
        )


@dataclass(frozen=True)
class DCDTrainingOutput:
    """Final deploy logit plus the four real SegResNet encoder features."""

    final_logits: torch.Tensor
    encoder_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

    def retain_encoder_feature_grads(self) -> "DCDTrainingOutput":
        for feature in self.encoder_features:
            if not feature.requires_grad:
                raise RuntimeError("student encoder feature does not require gradient")
            feature.retain_grad()
        return self


def encoder_feature_gradient_audit(
    output_or_features: DCDTrainingOutput | Sequence[torch.Tensor],
) -> OrderedDict[str, dict[str, float | bool]]:
    """Audit retained DCD gradients at every encoder stage after backward."""

    features = (
        output_or_features.encoder_features
        if isinstance(output_or_features, DCDTrainingOutput)
        else tuple(output_or_features)
    )
    if len(features) != len(ENCODER_CHANNELS):
        raise ValueError("gradient audit requires exactly four encoder features")
    report: OrderedDict[str, dict[str, float | bool]] = OrderedDict()
    for index, feature in enumerate(features):
        gradient = feature.grad
        finite = gradient is not None and bool(torch.isfinite(gradient).all())
        maximum = 0.0 if gradient is None else float(gradient.detach().abs().max())
        report[f"stage{index}"] = {
            "finite": finite,
            "nonzero": gradient is not None and maximum > 0.0,
            "max_abs": maximum,
        }
    return report


class DCDTrainingModel(nn.Module):
    """Plain SegResNet with training-only wavelet-detail loss buffers."""

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
        self.dcd_loss = DCDDetailLoss()

    @classmethod
    def from_plain_model(cls, plain: AutomaticPromptLocator) -> "DCDTrainingModel":
        if not isinstance(plain, AutomaticPromptLocator):
            raise TypeError("plain must be an AutomaticPromptLocator")
        if plain.center_aware:
            raise ValueError("DCD requires a plain center_aware=False locator")
        model = cls(init_filters=plain.init_filters, dropout=plain.dropout)
        model.network.load_state_dict(copy.deepcopy(plain.network.state_dict()), strict=True)
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
    ) -> "DCDTrainingModel":
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
            raise ValueError("each image spatial dimension must be positive and divisible by 8")

    def _encode(
        self, image: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        self._validate_image(image)
        encoded, down = self.network.encode(image)
        features = tuple(down)
        if len(features) != len(ENCODER_CHANNELS):
            raise RuntimeError("frozen SegResNet must expose exactly four encoder stages")
        expected_channels = tuple(
            self.init_filters * multiplier for multiplier in (1, 2, 4, 8)
        )
        observed_channels = tuple(int(value.shape[1]) for value in features)
        if observed_channels != expected_channels:
            raise RuntimeError(
                f"unexpected encoder channels {observed_channels}; expected {expected_channels}"
            )
        expected_shapes = tuple(
            tuple(size // divisor for size in image.shape[-3:])
            for divisor in (1, 2, 4, 8)
        )
        observed_shapes = tuple(tuple(value.shape[-3:]) for value in features)
        if observed_shapes != expected_shapes:
            raise RuntimeError(
                f"unexpected encoder shapes {observed_shapes}; expected {expected_shapes}"
            )
        return encoded, features  # type: ignore[return-value]

    def encode_features(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run only the encoder, for the frozen KD200 detail teacher."""

        _, features = self._encode(image)
        return features

    def forward(self, image: torch.Tensor) -> DCDTrainingOutput:
        """Run one encoder and one decoder pass for the trainable student."""

        encoded, features = self._encode(image)
        final = self.network.decode(encoded, list(reversed(features)))
        return DCDTrainingOutput(final_logits=final, encoder_features=features)

    def detail_loss(
        self,
        student_features: Sequence[torch.Tensor],
        teacher_features: Sequence[torch.Tensor],
    ) -> DCDLossOutput:
        return self.dcd_loss(student_features, teacher_features)

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
    "DB4_DEC_HI",
    "DB4_DEC_LO",
    "DB4_FILTER_BANK_SHA256",
    "DCDDetailLoss",
    "DCDLossOutput",
    "DCDStageLossOutput",
    "DCDTrainingModel",
    "DCDTrainingOutput",
    "ENCODER_CHANNELS",
    "ENCODER_RESOLUTIONS",
    "FILTER_LENGTH",
    "LEVELS",
    "SCHEMA",
    "SELECTED_BANDS",
    "SUBBAND_NAMES",
    "build_db4_filter_bank_3d",
    "dwt3",
    "encoder_feature_gradient_audit",
    "idwt3",
    "multilevel_detail_loss",
]
