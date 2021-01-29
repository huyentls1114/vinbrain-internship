import torch.nn as nn
import torch
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, activation = nn.Sigmoid(), epsilon = 1e-4):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.activation = activation

    def forward(self, predict, ground_truth):
        if self.activation is not None:
            predict = self.activation(predict)
        predict = predict.view(predict.shape[0], -1)
        ground_truth = ground_truth.view(ground_truth.shape[0], -1)
        intersection = torch.sum(predict*ground_truth, 1)
        union = torch.sum(predict, 1) + torch.sum(ground_truth,1 )
        dice = 1 - (2*intersection + self.epsilon)/( union + self.epsilon)
        return torch.mean(dice)    

from torchvision.ops import sigmoid_focal_loss
class FocalLoss(nn.Module):
    def __init__(self, alpha = 0.25, gamma = 2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        return sigmoid_focal_loss(inputs, targets, self.alpha, self.gamma, reduction="mean")

class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha = 0.25, gamma = 2):
        super(FocalLossWithLogits, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        pt = probs*targets+(1-probs)*(1-targets)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        loss = ((1-pt)**self.gamma)*bce

        alpha = self.alpha*targets+(1-self.alpha)*(1-targets)
        loss = alpha*loss
        return loss.mean()

class CombineLoss(nn.Module):
    def __init__(self, weights, **args):
        super(CombineLoss, self).__init__()
        weight_dice = weights["dice"]
        weight_focal = weights["focal"]
        alpha = args["alpha"]
        beta = args["beta"]

        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha, gamma)
    def forward(self, logits, ground_truth):
        return self.dice(logits, ground_truth)*self.weight_dice + self.focal(logits, ground_truth)*self.weight_focal
        
# class FocalLoss(nn.Module):
#     def __init__(self, alpha = 0.25, gamma = 2):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.activation = nn.Sigmoid()
#     def forward(self, logits, ground_truth):
#         with torch.no_grad():
#             alpha = torch.empty_like(logits).fill_(1 - self.alpha)
#             alpha[ground_truth == 1] = self.alpha
#         probs = self.activation(logits)
#         pt = probs*ground_truth+(1-probs)*(1-ground_truth)
#         ce_loss = F.binary_cross_entropy_with_logits(logits, ground_truth, reduction="none")
#         loss = self.alpha*torch.pow(1-pt, self.gamma)*ce_loss
#         return loss.mean()
        