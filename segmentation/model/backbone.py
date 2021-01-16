import torch.nn as nn
from torchvision.models import resnet18, resnet101, densenet161, densenet121

from .unet_vgg import UnetVGG
from .block import VGG16Block, UpBlock, Out
from .block import Resnet18BlocksUp, UpLayer
from .block import Resnet101BlockUp

from fastai.vision.models.unet import DynamicUnet
from fastai.layers import PixelShuffle_ICNR
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
        self.blocks = None
        self.out_conv = None

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

class BackboneResnet18VGG(Backbone):
    def __init__(self, encoder_args, decoder_args):
        super(BackboneResnet18VGG, self).__init__(encoder_args, decoder_args)        
        self.base_model = resnet18()
        self.features_name = ["layer3", "layer2", "layer1","relu"]
        self.last_layer = "layer4"
        self.input_channel = 3
        self.list_channels = [512, 256, 128, 64, 64, 1]
        self.up_class = UpBlock
        self.out_conv_class = Out
        self.initial_decoder()
        self.out_conv = nn.Sequential(
            PixelShuffle_ICNR(64),
            nn.Conv2d(64, 1, kernel_size = 1, stride = 1)
        )
class BackboneResnet18VGG(Backbone):
    def __init__(self, encoder_args, decoder_args):
        super(BackboneResnet18VGG, self).__init__(encoder_args, decoder_args)        
        self.base_model = resnet18()
        self.features_name = ["layer3", "layer2", "layer1","relu"]
        self.last_layer = "layer4"
        self.input_channel = 3
        self.list_channels = [512, 256, 128, 64, 64, 1]
        self.up_class = UpBlock
        self.out_conv_class = Out
        self.initial_decoder()
        self.out_conv = nn.Sequential(
            PixelShuffle_ICNR(64),
            nn.Conv2d(64, 1, kernel_size = 1, stride = 1)
        )
class BackboneDensenet161VGG(Backbone):
    def __init__(self, encoder_args, decoder_args):
        super(BackboneDensenet161VGG, self).__init__(encoder_args, decoder_args)        
        self.base_model = densenet161(**encoder_args).features
        self.features_name = ["denseblock3","denseblock2","denseblock1","relu0"]
        self.last_layer = "denseblock4"
        self.input_channel = 3
        self.list_channels = [2208, 2112, 768, 384, 96, 1]
        self.up_class = UpBlock
        self.out_conv_class = Out
        self.initial_decoder()
        self.out_conv = nn.Sequential(
            PixelShuffle_ICNR(96),
            nn.Conv2d(96, 1, kernel_size = 1, stride = 1)
        )
class BackboneDensenet121VGG(Backbone):
    def __init__(self, encoder_args, decoder_args):
        super(BackboneDensenet121VGG, self).__init__(encoder_args, decoder_args)        
        self.base_model = densenet121(**encoder_args).features
        self.features_name = ["denseblock3","denseblock2","denseblock1","relu0"]
        self.last_layer = "denseblock4"
        self.input_channel = 3
        self.list_channels = [1024, 1024, 512, 256, 64, 1]
        self.up_class = UpBlock
        self.out_conv_class = Out
        self.initial_decoder()
        self.out_conv = nn.Sequential(
            PixelShuffle_ICNR(self.list_channels[-2]),
            nn.Conv2d(self.list_channels[-2], self.list_channels[-1], kernel_size = 1, stride = 1)
        )

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

class BackBoneResnet101Dynamic(Backbone):
    def __init__(self, encoder_args, decoder_args):
        super(BackBoneResnet101Dynamic, self).__init__(encoder_args, decoder_args)
        self.encoder_args = encoder_args
        self.base_model = resnet101(**encoder_args)
        self.input_channel = 3
        self.initial_decoder()
    def initial_decoder(self):
        m = nn.Sequential(*list(self.base_model.children())[:-2])
        img_size = self.decoder_args["img_size"]
        self.unet = DynamicUnet(m, 1, (img_size, img_size), norm_type=None)

class BackBoneResnet18Dynamic(Backbone):
    def __init__(self, encoder_args, decoder_args):
        super(BackBoneResnet18Dynamic, self).__init__(encoder_args, decoder_args)
        self.encoder_args = encoder_args
        self.base_model = resnet18(**encoder_args)
        self.input_channel = 3
        self.initial_decoder()
    def initial_decoder(self):
        m = nn.Sequential(*list(self.base_model.children())[:-2])
        img_size = self.decoder_args["img_size"]
        self.unet = DynamicUnet(m, 1, (img_size, img_size), norm_type=None)