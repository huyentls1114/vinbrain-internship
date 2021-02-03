import torch
import numpy as np

class Dice_Score:
    def __init__(self, epsilon = 1e-4, threshold = 0.5):
        self.epsilon = epsilon
        self.threshold = threshold

    def __call__(self, predict, ground_truth):
        predict = _threshold(predict, self.threshold)
        predict = predict.view(predict.shape[0], -1)
        ground_truth = ground_truth.view(ground_truth.shape[0], -1)
        intersection = torch.sum(predict*ground_truth, 1)
        union = torch.sum(predict, 1) + torch.sum(ground_truth,1 )
        n = (2*intersection + self.epsilon)/( union + self.epsilon)
        return torch.mean((2*intersection + self.epsilon)/( union + self.epsilon))    

def _threshold(x, threshold = 0.5):
    return (x > threshold).type(x.dtype)


class DiceMetric:
    def __init__(self, threshold = 0.5, per_image = True, per_channel = False):
        self.per_image = per_image
        self.per_channel = per_channel
        self.threshold = threshold
        self.eps = 1e-6
    def __call__(self, intersection, union):
        loss = torch.mean((2 * intersection + self.eps) / (union+self.eps))
        return loss