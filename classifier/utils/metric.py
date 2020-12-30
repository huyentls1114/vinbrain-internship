import numpy as np
import torch

class Accuracy:
    def __init__(self, threshold = 0.5, from_logits:bool = True):
        self.threshold = threshold
        self.from_logis = from_logits
    
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.from_logis:
            y_pred, y_true = self._conversion(y_pred, y_true, self.threshold)
        return torch.mean((y_pred == y_true).float())
    
    def _conversion(self, y_pred, y_true, threshold):
        if len(y_pred.shape) == len(y_true.shape) + 1:
            y_pred = torch.argmax(y_pred, dim = 1)
        if len(y_pred.shape) == len(y_true.shape) and y_pred.dtype == torch.float:
            y_pred = (y_pred > threshold).float()
        return y_pred, y_true

class Precision:
    def __init__(self, epsilon = 1e-10):
        self.epsilon = epsilon

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        true_positive = torch.sum(torch.round(torch.clamp(y_pred*y_true, 0, 1)))
        num_positive_predict = torch.sum(torch.round(torch.clamp(y_pred, 0, 1)))
        return true_positive/(num_positive_predict+self.epsilon)

class Recall:
    def __init__(self, epsilon = 1e-10):
        self.epsilon = epsilon

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        true_positive = torch.sum(torch.round(torch.clamp(y_pred*y_true, 0, 1)))
        num_positive_true = torch.sum(torch.round(torch.clamp(y_true, 0, 1)))
        return true_positive/(num_positive_true+self.epsilon)

class F1_Score:
    def __init__(self, epsilon = 1e-10):
        self.epsilon = epsilon
        self.precision = Precision()
        self.recall = Recall()
    def __call__(self, y_pred, y_true):
        precision = self.precision(y_pred, y_true)
        recal = self.recall(y_pred, y_true)
        return (2*precision*recal)/(precision+recal+self.epsilon)

class Fbeta_Score:
    def __init__(self, beta = 0.5):
        self.beta = beta
        self.precision = Precision()
        self.recall = Recall()
    def __call__(self, y_pred, y_true):
        precision = self.precision(y_pred, y_true)
        recall = self.recall(y_pred, y_true)
        beta_2 = self.beta**2
        return (1-beta_2)*(precision*recall)/(beta_2*precision+recall)
    
class IOU:
    def __init__(self, threshold = 0.5, from_logis: bool = True, epsilon = 1e-6):
        self.from_fogits = from_logis
        self.threshold = threshold
        self.epsilon = epsilon
    
    def __call__(self, y_pred, y_true):
        #convert y_pred have the same dimention and type as y_true BATCH x H x W
        if self.from_fogits:
            y_pred, y_true = self._conversion(y_pred, y_true, self.threshold)

        intersection = (y_pred&y_true).float().sum((1, 2))
        union = (y_pred | y_true).float().sum((1, 2))
        iou = (intersection + self.epsilon)/(union+self.epsilon)
        return iou.mean()
        
    def _conversion(self, y_pred, y_true, threshold):
        if len(y_pred.shape) == len(y_true.shape) + 1:
            y_pred = torch.argmax(y_pred, dim = 1)
        if len(y_pred.shape) == len(y_true.shape) and y_pred.dtype == torch.float:
            y_pred = (y_pred > threshold).float()
        return y_pred, y_true