import torch
import torch.nn as nn
from torch.nn import functional as F

class WeightedBCEv2(nn.Module):
    def __init__(self):
        super(WeightedBCEv2, self).__init__()
    #
    def forward(self, y_pred, y_true, reduction='mean'):
        import pdb; pdb.set_trace()
        y_pred = y_pred[:,1].view(-1)
        y_true = y_true.view(-1)
        assert(y_pred.shape==y_true.shape)

        loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')

        pos = (y_true>0.5).float()
        neg = (y_true<0.5).float()
        pos_weight = (pos.sum().item() + 1) / len(y_true)
        neg_weight = (neg.sum().item() + 1) / len(y_true)
        pos_weight = 1 / pos_weight
        neg_weight = 1 / neg_weight 
        pos_weight = np.log(pos_weight) + 1
        neg_weight = np.log(neg_weight) + 1
        pos_weight = pos_weight / (pos_weight + neg_weight)
        neg_weight = neg_weight / (pos_weight + neg_weight)
        loss = (pos*loss*pos_weight + neg*loss*neg_weight).mean()
        return loss

class BCEWithLogits_Compose(nn.Module):
    def __init__(self, ignore_label=-1, weight=None, aux_weight = None):
        super(BCEWithLogits_Compose, self).__init__()
        self.ignore_label = ignore_label
        self.aux_weight = aux_weight

        self.main_criterion = WeightedBCEv2()
        self.aux_criterion = WeightedBCEv2()
        
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