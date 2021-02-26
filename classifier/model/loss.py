import torch.nn as nn
import torch.functional as F
import torch

from torchvision.ops import sigmoid_focal_loss
class FocalLoss(nn.Module):
    def __init__(self, alpha = 0.25, gamma = 2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        assert inputs.shape[1] <= 1
        print(type(inputs), type(targets))
        inputs = inputs[:, 0]

        return sigmoid_focal_loss(inputs, targets, self.alpha, self.gamma, reduction="mean")
