import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import BackboneOriginal
from .block import UpBlock, VGG16Block, Out

class Unet(nn.Module):
    def __init__(self, 
                backbone_class = BackboneOriginal,
                basenet_args = {},
                bilinear = True):
        super(Unet, self).__init__()

        self.bilinear = bilinear
        self.backbone = backbone_class(basenet_args)
        self.features_name = self.backbone.features_name
        self.initial_decoder()

    def initial_decoder(self):
        list_channels = self.backbone.list_channels
        self.blocks = nn.ModuleList()
        for i in range(len(list_channels)-2):
            input_channel = list_channels[i]
            output_channel = list_channels[i+1]
            up_block = self.backbone.up_class(input_channel,
                                            output_channel,
                                            bilinear = self.bilinear,
                                            **self.backbone.net_args)
            self.blocks.append(up_block)
        self.out_conv = self.backbone.out_conv_class(list_channels[-2], list_channels[-1])

    def forward(self, x):
        if x.shape[1] != self.backbone.input_channel:
            x = torch.cat([x, x, x], 1)
        x, features_value = self.forward_backbone(x)
        for i, block in enumerate(self.blocks):
            name = self.features_name[i]
            x = block(x, features_value[name])
            # print(name, x.shape)
        x = self.out_conv(x)
        return x

    def forward_backbone(self, x):
        features_value = {}
        for name, child in self.backbone.base_model.named_children():
            x = child(x)
            # print(name, x.shape)
            if name in self.features_name:
                features_value[name] = x
            if name == self.backbone.last_layer:
                break
        return x, features_value