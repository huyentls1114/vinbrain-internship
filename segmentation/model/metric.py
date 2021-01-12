import torch

class Dice_Score:
    def __init__(self, epsilon = 1e-4, threshold = 0.5):
        self.epsilon = epsilon
        self.threshold = threshold

    def __call__(self, predict, ground_truth):
        predict = _threshold(predict, self.threshold)
        intersection = torch.sum(predict*ground_truth)
        union = torch.sum(predict) + torch.sum(ground_truth)
        return (2*intersection + self.epsilon)/( union + self.epsilon)    

def _threshold(x, threshold = 0.5):
    return (x > threshold).type(x.dtype)
