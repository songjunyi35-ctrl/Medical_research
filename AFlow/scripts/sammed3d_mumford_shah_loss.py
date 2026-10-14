"""Explicit 3D adaptation of the author's region/TV energy, no patient access."""
import torch
from monai.losses import DiceCELoss
REFERENCE_WEIGHT = 1e-6 * 4 * 256**2
TV_WEIGHT = 0.001


def intensity01(image):
    x = image.float()
    dims = tuple(range(2, x.ndim))
    low = x.amin(dims, keepdim=True)
    span = x.amax(dims, keepdim=True) - low
    return (x-low) / span.clamp_min(1e-6)


def region_tv(probabilities, image):
    """Sum over classes/channels; divide by batch*spatial voxels, not class count."""
    if probabilities.ndim not in (4, 5) or image.ndim != probabilities.ndim:
        raise ValueError('expected 2D or 3D batch/channel/spatial tensors')
    if image.shape[0] != probabilities.shape[0] or image.shape[2:] != probabilities.shape[2:]:
        raise ValueError('image/probability geometry mismatch')
    p = probabilities.float().clamp(1e-10, 1.)
    x = image.float()
    dims = tuple(range(2,p.ndim));denom = p.sum(dims,keepdim=True)
    region = p.new_zeros(())
    for channel in range(x.shape[1]):
        values = x[:,channel:channel+1]
        centroid = (values*p).sum(dims,keepdim=True)/denom
        region = region + (p*(values-centroid).square()).sum()
    tv = sum(p.diff(dim=axis).abs().sum() for axis in dims)
    normalizer = p.shape[0]
    for size in p.shape[2:]:normalizer *= size
    return region/normalizer,tv/normalizer


def mumford_shah(logits,image):
    p = torch.sigmoid(logits.float())
    region,tv = region_tv(torch.cat([1-p,p],dim=1), intensity01(image))
    return REFERENCE_WEIGHT*(region+TV_WEIGHT*tv),region,tv


class FirstPromptMumfordShah:
    def __init__(self,image):
        self.image=image;self.calls=0
        self.seg=DiceCELoss(sigmoid=True,squared_pred=True,reduction='mean',lambda_dice=.5,lambda_ce=.5)
        self.seg_sum=image.new_zeros((),dtype=torch.float32)
        self.weighted=None;self.region=None;self.tv=None
    def __call__(self,logits,target):
        if self.calls==0 and bool((target.flatten(1).sum(1)<=0).any()):
            raise ValueError('empty target: no dropped/replaced case')
        loss=self.seg(logits,target)
        self.seg_sum=self.seg_sum+loss.detach().float()
        if self.calls==0:
            self.weighted,self.region,self.tv=mumford_shah(logits,self.image)
            loss=loss+self.weighted
        self.calls+=1
        return loss
