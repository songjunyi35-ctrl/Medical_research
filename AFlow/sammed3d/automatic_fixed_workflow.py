"""Fixed automatic-locator workflows; no label access or workflow selection."""
from contextlib import nullcontext
import torch

AXES = ((), (-1,), (-2,), (-2, -1), (-3,), (-3, -1), (-3, -2), (-3, -2, -1))
WORKFLOWS = {'identity': (0,), 'opposite2': (0, 7),
             'single_axis4': (0, 1, 2, 4), 'full8': tuple(range(8))}


def validate_views(views):
    values = tuple(views)
    if not values or any(type(v) is not int or not 0 <= v < 8 for v in values):
        raise ValueError('nonempty integer view IDs in [0,7] required')
    if tuple(sorted(set(values))) != values or values[0] != 0:
        raise ValueError('unique canonical ordered views including identity required')
    return values


@torch.inference_mode()
def restored_atom(model, image, view):
    axes = AXES[view]
    transformed = torch.flip(image, dims=axes) if axes else image
    context = torch.autocast('cuda', dtype=torch.bfloat16) if image.is_cuda else nullcontext()
    with context:
        logits = model(transformed).segmentation_logits
    return (torch.flip(logits, dims=axes) if axes else logits).float()


@torch.inference_mode()
def online_logits(model, image, views):
    views = validate_views(views)
    total = None
    for view in views:
        atom = restored_atom(model, image, view)
        if total is None:
            total = torch.zeros_like(atom, dtype=torch.float32)
        total.add_(atom)
    return total / len(views)


def fuse_atoms(atoms, views):
    views = validate_views(views)
    total = torch.zeros_like(atoms[views[0]], dtype=torch.float32)
    for view in views:
        total.add_(atoms[view].float())
    return total / len(views)
