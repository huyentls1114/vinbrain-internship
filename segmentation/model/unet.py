import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import BackboneOriginal
from .block import UpBlock, VGG16Block, Out
from crfseg import CRF

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
        x, features_value = self.forward_backbone(x)
        # print(self.features_name)
        for i, block in enumerate(self.blocks):
            name = self.features_name[i]
            x = block(x, features_value[name])
            # print(name, x.shape)
        x = self.out_conv(x)
        return x

    def forward_backbone(self, x):
        features_value = {}
        features_value["x"] = x
        if x.shape[1] != self.backbone.input_channel:
            x = torch.cat([x, x, x], 1)
        for name, child in self.base_model.named_children():
            x = child(x)
            # print(name, x.shape)
            if name in self.features_name:
                features_value[name] = x
            if name == self.backbone.last_layer:
                break
        return x, features_value
    
class UnetCRF(nn.Module):
    def __init__(self, 
                checkpoint_path,
                backbone_class = BackboneOriginal,
                encoder_args = {},
                decoder_args = {},
                current_epoch = 0,
                device = "cpu",
                gpu_id = 0):
        super(UnetCRF, self).__init__()
        self.unet = Unet(backbone_class, encoder_args, decoder_args)
        self.crf = CRF(n_spatial_dims=2)
        self.device = torch.device( device if device == "cpu" else "cuda:"+str(gpu_id))
        self.load_checkpoint(checkpoint_path)
        self.freeze_unet()

    def forward(self, x):
        return self.crf(self.unet(x))

    def load_checkpoint(self, filepath = None):
        self.unet.load_state_dict(torch.load(filepath, map_location = self.device))
    def freeze_unet(self):
        for param in self.unet.parameters():
            param.requires_grad = False

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
