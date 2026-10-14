"""Four mutually exclusive classes, with an algebraic WT union for inference."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from monai.losses import DiceCELoss
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator, LocatorOutput


def wt_logit(logits):
    if logits.ndim != 5 or logits.shape[1] != 4:
        raise ValueError('four class logits [B,4,D,H,W] required')
    z = logits.float()
    return torch.logsumexp(z[:, 1:], dim=1, keepdim=True) - z[:, :1]


def map_classes(original):
    if not bool(((original >= 0) & (original <= 4) & (original == original.round())).all()):
        raise ValueError('original integral labels 0..4 required')
    return torch.where(original == 4, 0, original).long()


class SemanticSegResNet(AutomaticPromptLocator):
    def __init__(self):
        super().__init__(center_aware=False, init_filters=16, dropout=.1)
        old = self.network.conv_final[2].conv
        self.network.conv_final[2].conv = nn.Conv3d(old.in_channels, 4, 1, bias=True)

    def initialize_from_binary(self, state):
        wkey = 'network.conv_final.2.conv.weight'
        bkey = 'network.conv_final.2.conv.bias'
        expected = self.state_dict()
        if set(state) != set(expected):
            raise ValueError('initial backbone keys mismatch')
        new = {k: v.detach().clone() for k, v in state.items()}
        new[wkey] = torch.cat([torch.zeros_like(state[wkey]), state[wkey].repeat(3, 1, 1, 1, 1)])
        new[bkey] = torch.cat([torch.zeros_like(state[bkey]), state[bkey].repeat(3) - math.log(3)])
        self.load_state_dict(new, strict=True)

    def forward(self, image):
        return LocatorOutput(wt_logit(self.network(image)), None)


class SemanticLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.whole = DiceCELoss(sigmoid=True, squared_pred=False, lambda_dice=1., lambda_ce=.1)

    def forward(self, logits, classes):
        z = logits.float()
        c = classes[:, 0].long()
        if not bool(((c >= 0) & (c <= 3)).all()):
            raise ValueError('training classes must already map cavity to background')
        whole_target = (classes > 0).float()
        if not bool((whole_target.flatten(1).sum(1) > 0).all()):
            raise ValueError('empty WT target')
        primary = self.whole(wt_logit(z), whole_target)
        prob = torch.softmax(z, dim=1)[:, 1:]
        target = torch.stack([c == k for k in (1, 2, 3)], dim=1).float()
        mass = target.flatten(2).sum(-1)
        intersection = (prob * target).flatten(2).sum(-1)
        denominator = prob.flatten(2).sum(-1) + mass
        component = 1 - (2 * intersection + 1e-5) / (denominator + 1e-5)
        present = mass > 0
        # Equal weight per patient image, then per present foreground class.
        component = ((component * present).sum(1) / present.sum(1).clamp_min(1)).mean()
        return primary + .5 * component + .1 * F.cross_entropy(z, c)
