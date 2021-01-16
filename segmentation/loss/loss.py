import torch.nn as nn
import torch

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


