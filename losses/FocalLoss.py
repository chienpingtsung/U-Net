import torch
from torch import nn
from torch.nn import functional


class FocalLoss(nn.Module):
    """
    Loss function from
    https://arxiv.org/pdf/1708.02002.pdf
    """

    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        """

        :param input: shape as (B, 1, H, W)
        :param target: shape as (B, 1, H, W)
        """
        p = torch.sigmoid(input)
        bce = functional.binary_cross_entropy(p, target, reduction='none')
        p_t = target * p + (1 - target) * (1 - p)
        loss = bce * ((1 - p_t) ** self.gamma)

        if self.alpha:
            alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
            loss = loss * alpha_t

        return loss.mean()
