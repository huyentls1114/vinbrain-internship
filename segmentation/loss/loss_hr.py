import torch
import torch.nn as nn
from torch.nn import functional as F
class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.BCEWithLogitsLoss(weight=weight)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(2), target.size(3)
        # print(score.shape, target.shape, ph, pw, h, w)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')
        # score = score/255.0
        # print(score.dtype, target.dtype)
        # print(score.shape, target.shape)
        loss = self.criterion(score, target)

        return loss

class CrossEntropyOCR(nn.Module):
    def __init__(self, ignore_label=-1, weight=None, align_corners = None, num_outputs = None, balance_weights = None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

        self.align_corners = align_corners
        self.num_outputs = num_outputs
        self.balance_weights = balance_weights

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(2), target.size(3)
        if ph != h or pw != w:
            # score = F.interpolate(input=score, size=(
            #     h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=self.align_corners)
        loss = self.criterion(score, target)

        return loss

    def forward(self, score, target):

        # if config.MODEL.NUM_OUTPUTS == 1:
        if self.num_outputs == 1:
            score = [score]

        # weights = config.LOSS.BALANCE_WEIGHTS
        weights = self.balance_weights
        assert len(weights) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])
