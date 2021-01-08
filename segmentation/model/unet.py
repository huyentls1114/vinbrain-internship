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

        self.encoder = self.backbone.encoder
        self.features_name = self.backbone.features_name
        self.initial_decoder()

    def initial_decoder(self):
        list_channels = self.backbone.list_channels
        self.blocks = []
        for i in range(len(list_channels)-2):
            input_channel = list_channels[i]
            output_channel = list_channels[i+1]
            up_block = UpBlock([input_channel, output_channel, output_channel],
                                block_class=self.backbone.block_class, 
                                net_args = self.backbone.net_args,
                                bilinear = self.bilinear)
            self.blocks.append(up_block)
        self.out_conv = Out(list_channels[-2], list_channels[-1])

    def forward(self, x):
        x, features_value = self.forward_backbone(x)

        for i, block in enumerate(self.blocks):
            x = block(x, features_value[self.features_name[i]])
        x = self.out_conv(x)
        return x

    def forward_backbone(self, x):
        features_value = {}
        for name, child in self.encoder.named_children():
            x = child(x)
            if name in self.features_name:
                features_value[name] = x
        return x, features_value