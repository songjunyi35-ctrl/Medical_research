"""Train-only conditional RC supervision with an unchanged WT inference head."""
import torch
from torch import nn
from torch.nn import functional as F
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator


class SeparateFinalChannels(nn.Module):
    """Keep primary arithmetic identical to a one-channel deployment convolution."""
    def __init__(self, conv):
        super().__init__()
        self.conv = conv

    def forward(self, features):
        conv = self.conv
        return torch.cat([
            F.conv3d(features, conv.weight[i:i+1], conv.bias[i:i+1], stride=conv.stride,
                     padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
            for i in range(2)
        ], dim=1)


class StructuredBackgroundLocator(AutomaticPromptLocator):
    def __init__(self):
        # Same two-output network in both arms; no center targets or routing used.
        super().__init__(center_aware=True, init_filters=16, dropout=0.1)
        self.network.conv_final[2] = SeparateFinalChannels(self.network.conv_final[2].conv)

    def forward(self, image):
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError('expected B,1,D,H,W image')
        logits = self.network(image)
        return logits[:, :1], logits[:, 1:2]


def conditional_rc_loss(logits, rc, wt):
    if logits.shape != rc.shape or logits.shape != wt.shape:
        raise ValueError('auxiliary shape mismatch')
    mask = 1.0 - wt.float()
    probability = logits.float().sigmoid()
    rc = rc.float()
    axes = tuple(range(1, logits.ndim))
    intersection = (mask * probability * rc).sum(axes)
    denominator = (mask * probability).sum(axes) + (mask * rc).sum(axes)
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    bce = (mask * F.binary_cross_entropy_with_logits(logits.float(), rc, reduction='none')).sum(axes) / mask.sum(axes).clamp_min(1)
    return (dice + 0.1 * bce).mean()


def export_plain_state(model):
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for suffix in ('weight', 'bias'):
        key = 'network.conv_final.2.conv.' + suffix
        state[key] = state[key][:1].clone()
    return state
