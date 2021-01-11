import torch.nn as nn
from torchvision.models import resnet18, resnet101

from .unet_vgg import UnetVGG
from .block import VGG16Block, UpBlock, Out
from .block import Resnet18BlocksUp, UpLayer
from .block import Resnet101BlockUp
class Backbone:
    def __init__(self, encoder_args, decoder_args):
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args

        self.base_model = None
        self.features_name = None
        self.last_layer = None
        self.input_channel = None
        self.list_channels = None
        self.up_class = None
        self.out_conv_class = None

    def initial_decoder(self):
        list_channels = self.list_channels
        self.blocks = nn.ModuleList()
        for i in range(len(list_channels)-2):
            input_channel = list_channels[i]
            output_channel = list_channels[i+1]
            up_block = self.up_class(input_channel,
                                     output_channel,
                                     **self.decoder_args)
            self.blocks.append(up_block)
        self.out_conv = self.out_conv_class(list_channels[-2], list_channels[-1])

class BackboneOriginal(Backbone):
    def __init__(self, encoder_args, decoder_args):
        super(BackboneOriginal, self).__init__(encoder_args, decoder_args)        
        self.base_model = UnetVGG(input_channel=1, output_channel=1, **encoder_args).encoder    
        self.features_name = ["down3", "down2", "down1", "inc"]
        self.last_layer = "down4"
        self.input_channel = 1
        self.list_channels = [1024, 512, 256, 128, 64, 1]
        self.up_class = UpBlock
        self.out_conv_class = Out
        self.initial_decoder()

class BackBoneResnet18(Backbone):
    def __init__(self, encoder_args, decoder_args):
        super(BackBoneResnet18, self).__init__(encoder_args, decoder_args)
        self.base_model = resnet18(**encoder_args)
        self.features_name = ["layer3", "layer2", "layer1", "relu"]
        self.last_layer = "layer4"
        self.input_channel = 3
        self.list_channels = [512, 256, 128, 64, 64, 1]
        self.up_class = Resnet18BlocksUp
        self.out_conv_class = UpLayer
        self.initial_decoder()


class BackBoneResnet101(Backbone):
    def __init__(self, encoder_args, decoder_args):
        super(BackBoneResnet101, self).__init__(encoder_args, decoder_args)
        self.encoder_args = encoder_args
        self.base_model = resnet101(**encoder_args)
        self.features_name = ["layer3", "layer2", "layer1", "relu"]
        self.last_layer = "layer4"
        self.input_channel = 3
        self.list_channels = [2048, 1024, 512, 256, 64, 1]
        self.num_bottleneck = [23, 4, 3]
        self.up_class = Resnet101BlockUp
        self.out_conv_class = UpLayer
        self.initial_decoder()

    def initial_decoder(self):
        list_channels = self.list_channels
        assert len(list_channels) > 3
        self.blocks = nn.ModuleList()
        for i in range(len(list_channels)-3):
            input_channel = list_channels[i]
            output_channel = list_channels[i+1]
            up_block = self.up_class(input_channel,
                                     output_channel,
                                     num_bottleneck = self.num_bottleneck[i],
                                     **self.decoder_args)
            self.blocks.append(up_block)
        self.blocks.append(
            self.up_class(
                list_channels[-3], list_channels[-2],
                last_block=True
            )
        )
        self.out_conv = self.out_conv_class(list_channels[-2], list_channels[-1])