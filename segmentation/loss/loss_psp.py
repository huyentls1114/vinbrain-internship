import torch
import torch.nn as nn
from torch.nn import functional as F
class BCEWithLogits_Compose(nn.Module):
    def __init__(self, ignore_label=-1, weight=None, aux_weight = None):
        super(BCEWithLogits_Compose, self).__init__()
        self.ignore_label = ignore_label
        self.aux_weight = aux_weight

        self.main_criterion = nn.BCEWithLogitsLoss(weight=weight)
        self.aux_criterion = nn.BCEWithLogitsLoss(weight=weight)
        
    def forward(self, output, target):
        if isinstance(output, tuple):
            output0 = torch.sigmoid(output[0])
            output1 = torch.sigmoid(output[1])
            main_loss = self.main_criterion(output0, target)
            aux_loss = self.aux_criterion(output1, target)
            loss = main_loss + self.aux_weight*aux_loss
        else:
            output = torch.sigmoid(output)
            loss = self.main_criterion(output, target)
        return loss