import torch.nn as nn
import torch.nn.functional as F
import torch

from torchvision.ops import sigmoid_focal_loss
class FocalLoss(nn.Module):
    def __init__(self, alpha = 0.25, gamma = 2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        assert inputs.shape[1] <= 1
        # print(type(inputs), inputs.dtype, type(targets), targets.dtype)
        inputs = inputs[:, 0]
        # print(inputs.shape, targets.shape)

        return sigmoid_focal_loss(inputs, targets.float(), self.alpha, self.gamma, reduction="mean")


class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        outputs = torch.sigmoid(outputs)
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        eps = 1e-8
        non_ignored = targets.view(-1) != self.ignore_index
        targets = targets.view(-1)[non_ignored].float()
        outputs = outputs.contiguous().view(-1)[non_ignored]
        outputs = torch.clamp(outputs, eps, 1. - eps)
        targets = torch.clamp(targets, eps, 1. - eps)
        pt = (1 - targets) * (1 - outputs) + targets * outputs
        return (-(1. - pt) ** self.gamma * torch.log(pt)).mean()

class BCE(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, outputs, targets):
        return F.binary_cross_entropy_with_logits(outputs, targets)