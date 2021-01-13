import torch

class Dice_Score:
    def __init__(self, epsilon = 1e-4, threshold = 0.5):
        self.epsilon = epsilon
        self.threshold = threshold

    def __call__(self, predict, ground_truth):
        predict = _threshold(predict, self.threshold)
        predict = torch.view(predict.shape[0], -1)
        ground_truth = torch.view(ground_truth.shape[0], -1)
        intersection = torch.sum(predict*ground_truth, 1)
        union = torch.sum(predict, 1) + torch.sum(ground_truth,1 )
        return torch.mean((2*intersection + self.epsilon)/( union + self.epsilon))    

def _threshold(x, threshold = 0.5):
    return (x > threshold).type(x.dtype)
