"""A single fixed two-model raw-logit rule, independent of images and labels."""
import torch


def equal_recipe_logits(sampling,occupancy):
    if sampling.shape!=occupancy.shape:raise ValueError('member shapes differ')
    result=(sampling.float()+occupancy.float())*.5
    if not torch.isfinite(result).all():raise ValueError('nonfinite fusion')
    return result
