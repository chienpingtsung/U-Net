from torchvision import ops


class FocalLoss:
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def __call__(self, inputs, targets):
        return ops.sigmoid_focal_loss(inputs, targets, self.alpha, self.gamma, self.reduction)
