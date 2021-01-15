import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import BackboneOriginal
from .block import UpBlock, VGG16Block, Out

class Unet(nn.Module):
    def __init__(self, 
                backbone_class = BackboneOriginal,
                encoder_args = {},
                decoder_args = {},
                pretrained = False):
        super(Unet, self).__init__()

        self.backbone = backbone_class(encoder_args, decoder_args)
        self.base_model = self.backbone.base_model
        self.features_name = self.backbone.features_name
        self.blocks = self.backbone.blocks
        self.out_conv = self.backbone.out_conv

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
        for name, child in self.base_model.named_children():
            x = child(x)
            # print(name, x.shape)
            if name in self.features_name:
                features_value[name] = x
            if name == self.backbone.last_layer:
                break
        return x, features_value

class UnetDynamic(Unet):
    def __init__(self, 
                backbone_class = BackboneOriginal,
                encoder_args = {},
                decoder_args = {}):
        super(UnetDynamic, self).__init__(backbone_class, encoder_args, decoder_args)
        self.unet = self.backbone.unet
    def forward(self, x):
        if x.shape[1] != self.backbone.input_channel:
            x = torch.cat([x, x, x], 1)
        return self.unet(x)
