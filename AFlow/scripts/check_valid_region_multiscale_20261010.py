"""Meaningful synthetic checks: original equivalence, independent crop oracle and gradients."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss
from scripts.valid_region_multiscale_20261010_common import (
    R, augment_with_mask, valid_dice_bce, valid_multiscale_loss,
)
from scripts.segresnet_pretrained_native_common import safe_augment
from scripts.segresnet_plain972_common import write, now


def main():
    torch.set_num_threads(2); torch.manual_seed(928)
    criterion = DiceCELoss(sigmoid=True, squared_pred=False, lambda_dice=1., lambda_ce=.1)
    shape = (3, 1, 16, 20, 24)
    target = (torch.rand(shape) > .8).float()
    logits = torch.randn(shape, requires_grad=True)
    a = criterion(logits, target)
    b = valid_dice_bce(logits, target, torch.ones_like(target), criterion)
    grad_a = torch.autograd.grad(a, logits, retain_graph=True)[0]
    grad_b = torch.autograd.grad(b, logits)[0]
    errors = {'all_ones_value_error': float(abs(a - b)),
              'all_ones_gradient_error': float((grad_a - grad_b).abs().max())}
    assert errors['all_ones_value_error'] < 2e-6 and errors['all_ones_gradient_error'] < 1e-7
    crops = [(slice(1, 12), slice(2, 17), slice(3, 20)),
             (slice(0, 9), slice(5, 19), slice(2, 23)),
             (slice(3, 16), slice(0, 15), slice(6, 24))]
    mask = torch.zeros(shape, dtype=torch.uint8); target = torch.zeros(shape)
    for i, sl in enumerate(crops):
        mask[(i, 0) + sl] = 1
        target[(i, 0) + sl] = (torch.rand_like(target[(i, 0) + sl]) > .85).float()
    logits = torch.randn(shape, requires_grad=True)
    cropped = torch.stack([criterion(logits[(slice(i, i + 1), slice(None)) + sl],
                                      target[(slice(i, i + 1), slice(None)) + sl]) for i, sl in enumerate(crops)]).mean()
    weighted = valid_dice_bce(logits, target, mask, criterion)
    errors['independent_crop_value_error'] = float(abs(cropped - weighted))
    assert errors['independent_crop_value_error'] < 2e-6
    gradient = torch.autograd.grad(weighted, logits)[0]
    assert bool((gradient[mask == 0] == 0).all())
    modified = logits.detach().clone(); modified[mask == 0] = 123.
    assert abs(float(valid_dice_bce(modified, target, mask, criterion) - weighted)) < 1e-7
    # Multiscale all-ones equivalence and zero gradient outside coarse coverage.
    dense = (torch.rand(shape) > .8).float()
    outputs = [torch.randn((3, 1, *(n // f for n in shape[2:])), requires_grad=True) for f in [1, 2, 4]]
    parts = [criterion(z, dense if f == 1 else F.avg_pool3d(dense, f, f)) for f, z in zip([1, 2, 4], outputs)]
    old = parts[0] + .25 * parts[1] + .125 * parts[2]
    current, _ = valid_multiscale_loss(outputs, dense, torch.ones_like(dense), criterion)
    errors['multiscale_all_ones_value_error'] = float(abs(old - current))
    assert errors['multiscale_all_ones_value_error'] < 2e-6
    current, _ = valid_multiscale_loss(outputs, target, mask, criterion)
    gradients = torch.autograd.grad(current, outputs)
    for f, gradient in zip([1, 2, 4], gradients):
        coverage = mask if f == 1 else F.avg_pool3d(mask.float(), f, f)
        assert bool((gradient[coverage == 0] == 0).all())
    # Same random draw order and exact old image/label output, including fallback.
    original_image = torch.randn(shape); original_label = torch.zeros(shape)
    original_label[:, :, 0, 0, 0] = 1
    original_mask = torch.zeros(shape, dtype=torch.uint8); original_mask[:, :, :9, :10, :11] = 1
    fallback_count = 0
    for seed in range(12):
        torch.manual_seed(seed)
        x1, y1, n1 = safe_augment(original_image.clone(), original_label.clone())
        torch.manual_seed(seed)
        x2, y2, m2, n2 = augment_with_mask(original_image.clone(), original_label.clone(), original_mask.clone())
        assert torch.equal(x1, x2) and torch.equal(y1, y2) and n1 == n2
        assert not bool((y2 * (1 - m2.float()) != 0).any())
        fallback_count += n2
    assert fallback_count > 0
    # Check ordinary non-fallback mask motion, not just points triggering fallback.
    for seed in range(5):
        torch.manual_seed(seed)
        full_label = original_mask.float()
        _, y, m, _ = augment_with_mask(original_image.clone(), full_label.clone(), original_mask.clone())
        assert torch.equal(y, m.float())
    result = {'utc': now(), 'status': 'passed', 'synthetic_only': True, 'patient_model_forward': False,
              'errors': errors, 'zero_gradients_and_invariance_outside_valid_regions': True,
              'fractional_auxiliary_coverage_checked': True,
              'old_augmentation_outputs_and_rng_identical': True, 'fallback_examples': fallback_count,
              'image_geometry_mask_not_gt_or_nonzero_intensity_mask': True}
    write(R / 'CPU_CHECKS.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
