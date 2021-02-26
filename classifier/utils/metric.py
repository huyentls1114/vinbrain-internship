import numpy as np
import torch
from torch.nn.functional import one_hot
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
            if y_pred.shape[1] > 1:
                y_pred = torch.argmax(y_pred, dim = 1)
            else:
                y_pred = y_pred.view(-1)
        if len(y_pred.shape) == len(y_true.shape) and y_pred.dtype == torch.float:
            y_pred = (y_pred > threshold).float()
        return y_pred, y_true

class Precision:
    def __init__(self, num_classes, epsilon = 1e-10, from_logis = True, threshold = 0.5):
        self.epsilon = epsilon
        self.from_logis = from_logis
        self.num_classes = num_classes 
        self.threshold = threshold

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.from_logis:
            y_pred, y_true = self._conversion(y_pred, y_true, self.threshold)

        y_pred = one_hot(y_pred, num_classes = self.num_classes)
        y_true = one_hot(y_true, num_classes = self.num_classes)

        true_positive = torch.sum(y_pred * y_true, 0)
        num_positive_predict = torch.sum(y_pred, 0)
        return true_positive/(num_positive_predict+self.epsilon)
    
    def _conversion(self, y_pred, y_true, threshold):
        if len(y_pred.shape) == len(y_true.shape) + 1:
            y_pred = torch.argmax(y_pred, dim = 1)
        if len(y_pred.shape) == len(y_true.shape) and y_pred.dtype == torch.float:
            y_pred = (y_pred > threshold).float()
        return y_pred, y_true

class Recall:
    def __init__(self, num_classes, epsilon = 1e-10, from_logis = True, threshold = 0.5):
        self.epsilon = epsilon
        self.from_logis = from_logis
        self.num_classes = num_classes 
        self.threshold = threshold

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.from_logis:
            y_pred, y_true = self._conversion(y_pred, y_true, self.threshold)

        y_pred = one_hot(y_pred, num_classes = self.num_classes)
        y_true = one_hot(y_true, num_classes = self.num_classes)

        true_positive = torch.sum(y_pred * y_true, 0)
        num_positive_groundtruth = torch.sum(y_true, 0)
        return true_positive/(num_positive_groundtruth+self.epsilon)
    
    def _conversion(self, y_pred, y_true, threshold):
        if len(y_pred.shape) == len(y_true.shape) + 1:
            y_pred = torch.argmax(y_pred, dim = 1)
        if len(y_pred.shape) == len(y_true.shape) and y_pred.dtype == torch.float:
            y_pred = (y_pred > threshold).float()
        return y_pred, y_true

class F1_Score:
    def __init__(self, num_classes, epsilon = 1e-10):
        self.epsilon = epsilon
        self.precision = Precision(num_classes)
        self.recall = Recall(num_classes)
    def __call__(self, y_pred, y_true):
        precision = self.precision(y_pred, y_true)
        recal = self.recall(y_pred, y_true)
        return (2*precision*recal)/(precision+recal+self.epsilon)

class Fbeta_Score:
    def __init__(self, num_classes, beta = 0.5):
        self.beta = beta
        self.precision = Precision(num_classes)
        self.recall = Recall(num_classes)
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