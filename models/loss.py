import torch
import torch.nn.functional as F
from torch import nn


def define_loss(config):
    loss_name = config['loss_name']
    if loss_name == 'focal_loss':
        return FocalLoss(config['focal_alpha'], config['focal_gamma'])
    elif loss_name == 'ce':
        loss_weight = torch.tensor([1, config['bce_loss_weight']]).float()
        return nn.CrossEntropyLoss(weight=loss_weight)
    else:
        raise NotImplementedError("only implemented for focal_loss, ce.")


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.1, gamma=2):
        super(FocalLoss, self).__init__()
        self.register_buffer('alpha', torch.tensor([alpha, 1 - alpha]))
        self.register_buffer('gamma', torch.tensor(gamma))

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-ce_loss)
        focal_loss = at * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
