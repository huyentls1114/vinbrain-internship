import torch
from torchvision.models.segmentation.fcn import FCNHead
import torch.nn as nn


def get_fcn(num_classes):
    model = torch.hub.load('pytorch/vision:v0.6.0', 'fcn_resnet101', pretrained=True)
    model.classifier = FCNHead(in_channels = 2048, channels = num_classes)
    model.aux_classifier = None
    return model


class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()
        model = torch.hub.load('pytorch/vision:v0.6.0', 'fcn_resnet101', pretrained=True)
        model.classifier = FCNHead(in_channels = 2048, channels = num_classes)
        model.aux_classifier = None
        self.model = model
    def forward(self, x):
        return self.model(x)["out"]
