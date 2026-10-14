"""Bounded image/label paired deformation with an isolated CPU random stream."""
import hashlib
import math
import torch
import torch.nn.functional as F


def displacement_bound(field):
    """Global infinity-norm Lipschitz bound for piecewise trilinear fields."""
    return sum(field.diff(dim=d).abs().flatten(2).amax(2) for d in (2, 3, 4)).amax(1)


def dense_field(coefficients, shape, device, max_displacement=2.0):
    if len(shape) != 3 or min(shape) < 3:
        raise ValueError('three spatial axes >= 3 required')
    coefficients = coefficients.to(device=device, dtype=torch.float32)
    t = torch.arange(-2, 3, device=device, dtype=torch.float32)
    k = torch.exp(-t.square()/2); k = k/k.sum()
    kernel = (k[:, None, None]*k[None, :, None]*k[None, None, :])[None, None]
    smooth = F.conv3d(coefficients, kernel.expand(3, 1, 5, 5, 5), padding=2, groups=3)
    field = F.interpolate(smooth, size=shape, mode='trilinear', align_corners=True)
    envelope = torch.ones(shape, device=device)
    for axis, n in enumerate(shape):
        x = torch.linspace(0, 1, n, device=device)
        dims = [1, 1, 1]; dims[axis] = n
        envelope = envelope * (4*x*(1-x)).reshape(dims)
    field = field * envelope[None, None]
    amplitude = field.abs().flatten(1).amax(1).clamp_min(1e-12)
    field = field * (max_displacement/amplitude).reshape(-1, 1, 1, 1, 1)
    bound = displacement_bound(field)
    field = field * torch.clamp(.25/bound.clamp_min(1e-12), max=1).reshape(-1, 1, 1, 1, 1)
    return field


def field_grid(field):
    shape = field.shape[-3:]
    axes = [(torch.arange(n, device=field.device, dtype=torch.float32)+.5)*2/n-1 for n in shape]
    base = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)
    scale = torch.tensor([2/n for n in shape], device=field.device)
    # Input axes D/H/W correspond to grid z/y/x, not x/y/z.
    return (base[None]+field.permute(0, 2, 3, 4, 1)*scale)[..., [2, 1, 0]]


class WeakElastic:
    def __init__(self, seed, probability=.2, max_displacement=2.):
        self.generator = torch.Generator(device='cpu').manual_seed(seed+100000)
        self.probability = probability
        self.max_displacement = max_displacement
        self.steps = self.samples = self.selected = self.incoming_empty = 0
        self.max_bound = self.max_observed_displacement = 0.
        self.random_digest = hashlib.sha256()

    def __call__(self, image, labels):
        if image.ndim != 5 or labels.ndim != 5 or image.shape[0] != labels.shape[0] or image.shape[-3:] != labels.shape[-3:]:
            raise ValueError('paired B,C,D,H,W tensors required')
        n = len(image)
        flags = torch.rand(n, generator=self.generator) < self.probability
        coefficients = torch.rand((n, 3, 7, 7, 7), generator=self.generator)*2-1
        self.random_digest.update(flags.numpy().tobytes()); self.random_digest.update(coefficients.numpy().tobytes())
        self.steps += 1; self.samples += n; self.selected += int(flags.sum())
        if not flags.any():
            return image, labels
        selected = flags.nonzero().flatten().to(image.device)
        field = dense_field(coefficients[flags], image.shape[-3:], image.device, self.max_displacement)
        bound = float(displacement_bound(field).max())
        if not math.isfinite(bound) or bound > .250001:
            raise RuntimeError('displacement Lipschitz safety gate')
        self.max_bound = max(self.max_bound, bound)
        self.max_observed_displacement = max(self.max_observed_displacement, float(field.abs().max()))
        grid = field_grid(field)
        # Pixel-center range on each axis: boundary fixed, no source extrapolation.
        for axis, size in enumerate(reversed(image.shape[-3:])):
            if float(grid[..., axis].abs().max()) > 1-1/size+1e-6:
                raise RuntimeError('source coordinate out of bounds')
        original = labels[selected]
        if not bool(((original == 0) | (original == 1)).all()):
            raise RuntimeError('nonbinary incoming labels')
        warped_labels = F.grid_sample(original, grid, mode='nearest', padding_mode='zeros', align_corners=False)
        before = original[:, 0].flatten(1).sum(1)
        after = warped_labels[:, 0].flatten(1).sum(1)
        self.incoming_empty += int((before == 0).sum())
        if bool(((before > 0) & (after == 0)).any()):
            raise RuntimeError('new deformation erased nonempty WT')
        if labels.shape[1] == 2 and bool((warped_labels[:, 0]*warped_labels[:, 1] != 0).any()):
            raise RuntimeError('overlapping WT/RC labels')
        warped_image = F.grid_sample(image[selected], grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        if not bool(torch.isfinite(warped_image).all()):
            raise RuntimeError('nonfinite augmented image')
        result_image, result_labels = image.clone(), labels.clone()
        result_image[selected] = warped_image; result_labels[selected] = warped_labels
        return result_image, result_labels

    def receipt(self):
        return {'steps':self.steps, 'samples':self.samples, 'selected':self.selected,
                'incoming_empty_selected':self.incoming_empty, 'max_lipschitz_bound':self.max_bound,
                'max_axis_displacement_voxels':self.max_observed_displacement,
                'independent_random_stream_sha256':self.random_digest.hexdigest(),
                'probability':self.probability, 'max_displacement_voxels':self.max_displacement}
