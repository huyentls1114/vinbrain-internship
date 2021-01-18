import torch.nn as nn
import torch
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, activation = nn.Sigmoid()):
        super(DiceLoss, self).__init__()
        
        self.activation = activation

    def forward(self, predict, ground_truth):
        if activation is not None:
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
        