"""One change to the native multiscale recipe: geometry-derived valid-region losses."""
from pathlib import Path
import torch
import torch.nn.functional as F
from scripts.segresnet_pretrained_native_common import NativeCanvasDataset, CANVAS, canvas_slices
from scripts.train_t1ce_wt_capg3d import shift_without_wrap

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'archive/research_records/2026-10-10_segresnet_valid_region_supervision'
OUT = ROOT / 'workspace/BraTS/segresnet_t1ce_wt/valid_region_20261010'
CACHE = Path('/tmp/aflow_valid_region_20261010')


class ValidRegionDataset(NativeCanvasDataset):
    def __getitem__(self, i):
        item = super().__getitem__(i)
        mask = torch.zeros((1, *CANVAS), dtype=torch.uint8)
        mask[(slice(None),) + canvas_slices(self.geometry[item['case_id']])] = 1
        item['valid_mask'] = mask
        return item


def augment_with_mask(image, label, mask, *, max_shift=8):
    """Same draws/order as old safe_augment; the mask gets no intensity changes."""
    if image.shape != label.shape or mask.shape != label.shape:
        raise ValueError('image/label/mask shape mismatch')
    if not bool((label.flatten(1).sum(1) > 0).all()):
        raise RuntimeError('empty original training target')
    original_image = image.clone(); original_label = label.clone(); original_mask = mask.clone()
    for batch_index in range(image.shape[0]):
        for axis in (-3, -2, -1):
            if bool(torch.rand((), device=image.device) < .5):
                image[batch_index] = torch.flip(image[batch_index], dims=(axis,))
                label[batch_index] = torch.flip(label[batch_index], dims=(axis,))
                mask[batch_index] = torch.flip(mask[batch_index], dims=(axis,))
        shifts = tuple(int(torch.randint(-max_shift, max_shift + 1, (), device=image.device)) for _ in range(3))
        image[batch_index] = shift_without_wrap(image[batch_index], shifts)
        label[batch_index] = shift_without_wrap(label[batch_index], shifts)
        mask[batch_index] = shift_without_wrap(mask[batch_index], shifts)
    shape = (image.shape[0], 1, 1, 1, 1)
    scale = torch.empty(shape, device=image.device).uniform_(.9, 1.1)
    offset = torch.empty(shape, device=image.device).uniform_(-.1, .1)
    noise = torch.empty(shape, device=image.device).uniform_(0., .05)
    image = image * scale + offset + torch.randn_like(image) * noise
    missing = label.flatten(1).sum(1) <= 0
    if bool(missing.any()):
        image[missing] = original_image[missing]
        label[missing] = original_label[missing]
        mask[missing] = original_mask[missing]
    return image, label, mask, int(missing.sum())


def valid_dice_bce(logits, target, weights, criterion):
    """Per-case weighted original MONAI Dice(squared=False,soft_label=False)+.1BCE.

    Keep the original sigmoid and actual reduction dtype (including autocast), and
    the original tp/fp/fn order and smoothing. No additional loss coefficient.
    """
    if logits.shape != target.shape or weights.shape != target.shape:
        raise ValueError('weighted loss shape mismatch')
    if criterion.dice.batch or criterion.dice.squared_pred or criterion.dice.soft_label:
        raise ValueError('unexpected historical Dice definition')
    dims = tuple(range(2, target.ndim))
    w = weights.float()
    p = torch.sigmoid(logits)
    tp = (p * target * w).sum(dim=dims)
    pred_sum = (p * w).sum(dim=dims).to(torch.sum(p, dim=dims).dtype)
    target_sum = (target * w).sum(dim=dims)
    fp = pred_sum - tp; fn = target_sum - tp
    numerator = 2 * tp + criterion.dice.smooth_nr
    denominator = 2 * (tp + .5 * fp + .5 * fn) + criterion.dice.smooth_dr
    dice = (1 - numerator / denominator).mean()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    mass = w.sum(dim=dims)
    if not bool((mass > 0).all()):
        raise RuntimeError('empty valid region')
    bce = ((bce * w).sum(dim=dims) / mass).mean()
    return criterion.lambda_dice * dice + criterion.lambda_ce * bce


def valid_multiscale_loss(outputs, target, mask, criterion):
    main, half, quarter = outputs
    if main.shape != target.shape or mask.shape != target.shape:
        raise ValueError('main mask/target shape mismatch')
    if half.shape[2:] != tuple(n // 2 for n in target.shape[2:]) or quarter.shape[2:] != tuple(n // 4 for n in target.shape[2:]):
        raise ValueError('auxiliary output scale mismatch')
    mask = mask.float()
    if bool(((mask != 0) & (mask != 1)).any()) or bool((target * (1 - mask) != 0).any()):
        raise RuntimeError('main geometry mask is not binary or excludes retained target')
    parts = [valid_dice_bce(main, target, mask, criterion)]
    for factor, logits in [(2, half), (4, quarter)]:
        coverage = F.avg_pool3d(mask, factor, factor)
        occupied = F.avg_pool3d(target * mask, factor, factor)
        # Nonempty coarse cells contain at least one valid original voxel.
        coarse_target = occupied / coverage.clamp_min(1 / factor**3)
        parts.append(valid_dice_bce(logits, coarse_target, coverage, criterion))
    return parts[0] + .25 * parts[1] + .125 * parts[2], parts
