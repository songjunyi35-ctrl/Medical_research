"""Reliable anatomical reference segmentation; no label-dependent inference.

Coordinates are input tensor voxel coordinates (D,H,W), not grid_sample (x,y,z).
The caller supplies an image-derived affine reflection and a geometric valid mask.
"""
from dataclasses import dataclass
import itertools

import torch
from torch import nn
from torch.nn import functional as F

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator


def voxel_grid(shape, device):
    return torch.stack(torch.meshgrid(*(torch.arange(n, device=device, dtype=torch.float32)
                                       for n in shape), indexing="ij"), dim=-1)


def feature_reference_grid(matrix, input_shape, feature_shape, offset=(0, 0, 0)):
    """Strided SegResNet features have centers at input coordinates 8*k.

    align_corners=True maps those centers exactly; input sizes needn't be divisible
    by eight for this utility, though the backbone itself requires that padding.
    """
    if matrix.ndim != 3 or matrix.shape[-2:] != (4, 4):
        raise ValueError("reflection must be [B,4,4]")
    if any(n < 2 for n in feature_shape):
        raise ValueError("reference feature axes must have at least two locations")
    if tuple((n-1)//8+1 for n in input_shape) != tuple(feature_shape):
        raise ValueError("feature lattice does not match stride8 input centers")
    stride = torch.tensor([8., 8., 8.], device=matrix.device)
    points = voxel_grid(feature_shape, matrix.device) * stride
    mapped = torch.einsum("bij,dhwj->bdhwi", matrix[:, :3, :3].float(), points)
    mapped = mapped + matrix[:, None, None, None, :3, 3].float()
    mapped = mapped / stride + torch.tensor(offset, device=matrix.device)
    limits = torch.tensor(feature_shape, device=matrix.device, dtype=torch.float32) - 1
    valid = ((mapped >= 0) & (mapped <= limits)).all(-1).unsqueeze(1)
    grid = (2 * mapped / limits - 1).flip(-1)
    return grid, valid


def conjugate_reflection(matrix, transform):
    """If augmentation maps old -> new coordinates, M_new = A M_old A^-1."""
    return transform @ matrix @ torch.linalg.inv(transform)


def spatial_augment(image, target, classes, valid, matrix, *, flips, shifts):
    shape = image.shape[-3:]
    transform = torch.eye(4, device=image.device, dtype=torch.float32)
    arrays = [image, target, classes, valid]
    for axis, flip in enumerate(flips):
        if flip:
            arrays = [a.flip(axis + 1) for a in arrays]
            one = torch.eye(4, device=image.device)
            one[axis, axis] = -1
            one[axis, 3] = shape[axis] - 1
            transform = one @ transform
    for axis, shift in enumerate(shifts):
        if shift:
            arrays = [a.roll(shift, axis + 1) for a in arrays]
            slices = [slice(None)] * 4
            slices[axis + 1] = slice(0, shift) if shift > 0 else slice(shift, None)
            for a in arrays:
                a[tuple(slices)] = 0
            transform[axis, 3] += shift
    changed = conjugate_reflection(matrix, transform)
    return (*arrays, changed)


class LocalReference(nn.Module):
    def __init__(self, channels, dim=16):
        super().__init__()
        self.query = nn.Conv3d(channels, dim, 1)
        self.key = nn.Conv3d(channels, dim, 1)
        self.value = nn.Conv3d(channels, dim, 1)
        self.offsets = tuple(itertools.product((-1, 0, 1), repeat=3))
        self.dim = dim

    def forward(self, features, reflection, valid, *, corrupt=False):
        size = features.shape[-3:]
        q = F.normalize(self.query(features).float(), dim=1)
        k = F.normalize(self.key(features).float(), dim=1)
        current = self.value(features).float()
        v = current
        # Sampling at backbone centers, rather than interpolate's half-pixel centers.
        support = valid[:, :, ::8, ::8, ::8].float()
        if support.shape[-3:] != size:
            raise ValueError("valid mask and backbone feature centers disagree")
        if corrupt:
            # Known invalid correspondence: move reference keys/values by >= half
            # the AP feature axis. Main image, segmentation labels stay unchanged.
            shift = size[1] // 2
            k, v = k.roll(shift, 3), v.roll(shift, 3)
        scores, values, masks = [], [], []
        for offset in self.offsets:
            grid, inside = feature_reference_grid(reflection, valid.shape[-3:], size, offset)
            ref_k = F.grid_sample(k, grid, align_corners=True, padding_mode="zeros")
            ref_v = F.grid_sample(v, grid, align_corners=True, padding_mode="zeros")
            ref_mask = F.grid_sample(support, grid, mode="nearest", align_corners=True) > .5
            mask = inside & ref_mask & (support > .5)
            score = (q * ref_k).sum(1, keepdim=True) / .2
            score = score - .1 * sum(x*x for x in offset)
            scores.append(score)
            values.append(ref_v)
            masks.append(mask)
        logits = torch.cat(scores, 1)
        mask = torch.cat(masks, 1)
        any_valid = mask.any(1, keepdim=True)
        # Finite sentinel + explicit normalization avoids all-invalid NaNs.
        weight = torch.softmax(logits.masked_fill(~mask, -1e4), 1) * mask.float()
        weight = weight / weight.sum(1, keepdim=True).clamp_min(1e-8)
        reference = sum(weight[:, j:j+1] * value for j, value in enumerate(values))
        entropy = -(weight * weight.clamp_min(1e-8).log()).sum(1, keepdim=True) / 3.296
        # Values on both sides share one projection; query/key only select matches.
        return current, reference, entropy, any_valid


@dataclass
class RaceOutput:
    segmentation_logits: torch.Tensor
    base_logits: torch.Tensor
    embedding: torch.Tensor
    gate: torch.Tensor
    has_reference: torch.Tensor
    bad_logits: torch.Tensor | None = None
    bad_gate: torch.Tensor | None = None


class RaceSeg(nn.Module):
    def __init__(self, width=32, dropout=.1, dim=16, enabled=True):
        super().__init__()
        self.backbone = AutomaticPromptLocator(center_aware=False, init_filters=width, dropout=dropout)
        self.reference = LocalReference(width * 8, dim)
        self.evidence = nn.Sequential(nn.Conv3d(dim*4, 32, 1), nn.GroupNorm(8, 32), nn.SiLU())
        self.project = nn.Conv3d(32, dim, 1)
        self.gate_head = nn.Conv3d(33, 1, 1)
        self.correction = nn.Conv3d(32, 1, 1)
        # A zero scalar with nonzero correction initialization allows the scalar
        # to learn immediately; never zero both factors (a dead residual).
        self.strength = nn.Parameter(torch.zeros(()))
        self.enabled = enabled
        self.prototypes = nn.Parameter(torch.randn(3, 2, dim) * .1)

    def apply_reference(self, features, matrix, valid, base, corrupt):
        query, reference, entropy, available = self.reference(features, matrix, valid, corrupt=corrupt)
        delta = query - reference
        evidence = self.evidence(torch.cat([query, reference, delta, delta.abs()], 1))
        gate = torch.sigmoid(self.gate_head(torch.cat([evidence, entropy], 1))) * available
        low = gate * torch.tanh(self.correction(evidence))
        correction = F.interpolate(low.float(), size=base.shape[-3:], mode="trilinear", align_corners=True)
        logits = base.float() + 2 * torch.tanh(self.strength) * correction
        return logits, F.normalize(self.project(evidence).float(), dim=1), gate, available

    def forward(self, image, reflection=None, valid=None, *, corruption=False):
        network = self.backbone.network
        features, skip = network.encode(image)
        base = network.decode(features, list(reversed(skip)))
        if not self.enabled:
            return RaceOutput(base, base, features[:, :1], features[:, :1]*0, features[:, :1].bool() & False)
        if reflection is None or valid is None:
            raise ValueError("RACE requires image-derived geometry and support")
        logits, embedding, gate, available = self.apply_reference(features, reflection, valid, base, False)
        bad_logits = bad_gate = None
        if corruption:
            bad_logits, _, bad_gate, _ = self.apply_reference(features, reflection, valid, base, True)
        return RaceOutput(logits, base, embedding, gate, available, bad_logits, bad_gate)


def typed_contrast(embedding, classes, prototypes, valid, max_per_class=128):
    """Class-balanced supervised multi-prototype contrast, training labels only."""
    labels = classes[:, :, ::8, ::8, ::8].long()[:, 0]
    support = valid[:, :, ::8, ::8, ::8][:, 0] > .5
    anchors = F.normalize(prototypes.float(), dim=-1)
    terms = []
    for b in range(len(labels)):
        flat = embedding[b].flatten(1).T
        for category in range(3):
            indices = ((labels[b].flatten() == category) & support[b].flatten()).nonzero().flatten()
            if not len(indices):
                continue
            if len(indices) > max_per_class:
                # Isolate auxiliary sampling from the matched backbone dropout RNG.
                devices=[indices.device.index] if indices.is_cuda else []
                with torch.random.fork_rng(devices=devices):
                    indices = indices[torch.randperm(len(indices), device=indices.device)[:max_per_class]]
            logits = torch.einsum("nd,ckd->nck", flat[indices].float(), anchors) / .2
            logits = torch.logsumexp(logits, -1)
            terms.append(F.cross_entropy(logits, torch.full((len(indices),), category, device=indices.device)))
    return torch.stack(terms).mean() if terms else embedding.sum()*0
