import torch
from torch import nn


class FocalLoss(nn.Module):
    """
    Loss function from
    https://arxiv.org/pdf/1708.02002.pdf
    """

    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        """Specifically using equation (5) from paper

        :param inputs: shape as (B, C, H, W)
        :param targets: shape as (B, 1, H, W), where C as index of class number.
        """
        logpt = nn.functional.log_softmax(inputs, dim=1)
        logpt = torch.gather(logpt, 1, targets)
        pt = torch.exp(logpt)

        loss = -1 * (1 - pt) ** self.gamma * logpt

        return loss.mean()
