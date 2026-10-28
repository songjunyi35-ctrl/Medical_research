"""Label-free transforms and decoders, separated from all performance scoring."""
import math
import numpy as np
from scipy import ndimage as ndi
import torch
import torch.nn.functional as F


def move_zero(x, axis, amount):
    """Translate without wraparound; axis indexes the last three dimensions."""
    result = torch.zeros_like(x)
    source = [slice(None)] * x.ndim
    target = source.copy()
    dim = x.ndim - 3 + axis
    if amount > 0:
        source[dim] = slice(0, -amount)
        target[dim] = slice(amount, None)
    elif amount < 0:
        source[dim] = slice(-amount, None)
        target[dim] = slice(0, amount)
    else:
        return x.clone()
    result[tuple(target)] = x[tuple(source)]
    return result


def rotation_theta(shape, axis, degrees, device):
    angle = math.radians(degrees)
    rotation = np.eye(3)
    # Coordinates in affine_grid are W,H,D, unlike the stored D,H,W.
    pair = [(0, 1), (0, 2), (1, 2)][axis]
    a, b = pair
    rotation[a, a] = rotation[b, b] = math.cos(angle)
    rotation[a, b] = -math.sin(angle)
    rotation[b, a] = math.sin(angle)
    scales = np.diag(np.asarray(shape[::-1], dtype=float))
    theta = np.zeros((3, 4), np.float32)
    theta[:, :3] = np.linalg.inv(scales) @ rotation @ scales
    return torch.tensor(theta, device=device)[None]


def resample(x, theta):
    grid = F.affine_grid(theta.expand(len(x), -1, -1), x.shape, align_corners=False)
    return F.grid_sample(x.float(), grid, mode='bilinear', padding_mode='zeros', align_corners=False)


def predict_atom(model, image, atom, baseline=None):
    def forward(x):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(x).segmentation_logits
        return output.float()

    kind = atom['kind']
    if kind == 'scale_flip':
        size = tuple(max(8, int(round(s * atom['scale'] / 8)) * 8) for s in image.shape[-3:])
        x = image if size == image.shape[-3:] else F.interpolate(image, size=size, mode='trilinear', align_corners=False)
        axes = tuple(i + 2 for i in range(3) if atom['flip'] & (1 << i))
        if axes:
            x = torch.flip(x, axes)
        z = forward(x)
        if axes:
            z = torch.flip(z, axes)
        return z if size == image.shape[-3:] else F.interpolate(z, image.shape[-3:], mode='trilinear', align_corners=False)
    if kind == 'shift':
        assert baseline is not None
        z = move_zero(forward(move_zero(image, atom['axis'], atom['amount'])), atom['axis'], -atom['amount'])
        coverage = move_zero(move_zero(torch.ones_like(image), atom['axis'], atom['amount']), atom['axis'], -atom['amount'])
        return torch.where(coverage > .999, z, baseline)
    if kind == 'rotate':
        assert baseline is not None
        theta = rotation_theta(image.shape[-3:], atom['axis'], atom['degrees'], image.device)
        inverse = rotation_theta(image.shape[-3:], atom['axis'], -atom['degrees'], image.device)
        z = resample(forward(resample(image, theta)), inverse)
        coverage = resample(resample(torch.ones_like(image), theta), inverse)
        return torch.where(coverage > .999, z, baseline)
    if kind in ('gain', 'offset', 'gamma'):
        if kind == 'gain':
            x = image * atom['value']
        elif kind == 'offset':
            x = torch.where(image != 0, image + atom['value'], image)
        else:
            x = image.sign() * image.abs().pow(atom['value'])
        return forward(x)
    if kind == 'sliding':
        from monai.inferers import sliding_window_inference
        return sliding_window_inference(image, (128, 128, 128), 2, forward,
                                        overlap=atom['overlap'], mode=atom['blend'],
                                        sigma_scale=.125, padding_mode='constant', cval=0.)
    raise ValueError(kind)


def fuse_score(arrays, recipe):
    result = torch.zeros_like(arrays[recipe['atoms'][0]])
    for name, weight in zip(recipe['atoms'], recipe['weights']):
        z = arrays[name]
        result.add_(torch.sigmoid(z) if recipe['fusion'] == 'probability' else z, alpha=weight)
    return result


def fuse(arrays, recipe):
    score = fuse_score(arrays, recipe)
    return score if recipe['fusion'] == 'probability' else torch.sigmoid(score)


def threshold_counts(probability, target, thresholds):
    """All strict > thresholds in a single histogram, preserving exact tie behavior."""
    bins = torch.bucketize(probability.contiguous().flatten(), thresholds, right=False)
    width = len(thresholds) + 1
    bins.add_(target.flatten().long() * width)
    hist = torch.bincount(bins, minlength=2 * width).reshape(2, width)
    tails = hist.flip(1).cumsum(1).flip(1)[:, 1:]
    tp = tails[1]
    fp = tails[0]
    positive = target.sum(dtype=torch.int64)
    denominator = tp + fp + positive
    dice = torch.where(denominator > 0, 2 * tp.float() / denominator.float(), 1.)
    return dice, tp, fp


def component_metrics(probability, target, threshold, rules):
    """All component filters share one label pass. Labels never enter filtering."""
    mask = probability > threshold
    labels, count = ndi.label(mask, structure=np.ones((3, 3, 3), np.uint8))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    overlap = np.bincount(labels[target].ravel(), minlength=count + 1)
    sizes[0] = 0
    overlap[0] = 0
    maxprob = None
    if count:
        maxprob = ndi.maximum(probability, labels, np.arange(count + 1))
    out = []
    for rule in rules:
        keep = np.ones(count + 1, bool)
        kind = rule['kind']
        if kind == 'largest':
            keep[:] = False
            if count:
                keep[int(np.argmax(sizes))] = True
        elif kind == 'volume':
            keep = sizes >= rule['value']
        elif kind == 'relative':
            keep = sizes >= sizes.max(initial=0) * rule['value']
        elif kind == 'seed' and count:
            keep = maxprob >= rule['value']
        elif kind not in ('none', 'seed'):
            raise ValueError(kind)
        keep[0] = False
        tp = int(overlap[keep].sum())
        predicted = int(sizes[keep].sum())
        positive = int(target.sum())
        den = predicted + positive
        dice = float(np.float32(2 * tp) / np.float32(den)) if den else 1.
        out.append((dice, tp, predicted - tp))
    return out


def component_mask(probability, threshold, rule):
    labels, count = ndi.label(probability > threshold, structure=np.ones((3, 3, 3), np.uint8))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0
    keep = np.ones(count + 1, bool)
    if rule['kind'] == 'largest':
        keep[:] = False
        if count:
            keep[int(np.argmax(sizes))] = True
    elif rule['kind'] == 'volume':
        keep = sizes >= rule['value']
    elif rule['kind'] == 'relative':
        keep = sizes >= sizes.max(initial=0) * rule['value']
    elif rule['kind'] == 'seed' and count:
        keep = ndi.maximum(probability, labels, np.arange(count + 1)) >= rule['value']
    keep[0] = False
    return keep[labels]


def volume_decode(probability, temperature):
    """Maximize 2*sum(top-k p)/(k+sum(p)), a ratio-of-expectations approximation.

    Not exact RankDice. A single global temperature is a frozen decoder parameter.
    """
    p = torch.as_tensor(probability, device='cuda').flatten().clamp(1e-7, 1 - 1e-7)
    if temperature != 1:
        p = torch.sigmoid(torch.logit(p) / temperature)
    ordered, indices = p.sort(descending=True, stable=True)
    # CUDA float32 cumsum can vary in its reduction order. The objective is very
    # flat near its maximum, so one ulp may move the selected k by a few voxels.
    # Stable GPU sorting plus a sequential CPU float64 prefix makes k repeatable.
    prefix = np.cumsum(ordered.cpu().numpy(), dtype=np.float64)
    k = np.arange(1, len(p) + 1, dtype=np.float64)
    score = 2 * prefix / (k + prefix[-1])
    count = int(np.argmax(score)) + 1
    mask = torch.zeros_like(p, dtype=torch.bool)
    mask[indices[:count]] = True
    return mask.reshape(probability.shape).cpu().numpy(), count
