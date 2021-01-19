import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.layers import PixelShuffle_ICNR
from collections import OrderedDict
from .backbone import Backbone
from torchvision.models import densenet121

class _DenseLayer(nn.Module):
    def __init__(self, input_channel,
                       growth_rate,
                       bottleneck_size,
                       drop_rate):
        super(_DenseLayer, self).__init__()
        self.norm1 : nn.BatchNorm2d
        self.add_module('norm1', nn.BatchNorm2d(input_channel))
        self.relu1 : nn.ReLU
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.conv1 : nn.Conv2d
        self.add_module("conv1", nn.Conv2d(input_channel, growth_rate*bottleneck_size, kernel_size=1, stride=1, bias=False))

        self.norm2 : nn.BatchNorm2d
        self.add_module('norm2', nn.BatchNorm2d(bottleneck_size*growth_rate))
        self.relu2 : nn.ReLU
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.conv2 : nn.Conv2d
        self.add_module("conv2", nn.Conv2d(growth_rate*bottleneck_size, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))

        self.drop_rate = drop_rate
    def bottleneck_function(self, inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output
    
    def forward(self, input):
        if isinstance(input, torch.Tensor):
            preview_features = [input]
        else:
            preview_features = input
        bottleneck_output = self.bottleneck_function(preview_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p = self.drop_rate, training= self.training)
        return new_features
class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers,
                       input_channel,
                       bottleneck_size,
                       growth_rate,
                       drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_channel+i*growth_rate,
                                growth_rate = growth_rate,
                                bottleneck_size = bottleneck_size,
                                drop_rate = drop_rate)
            self.add_module('denselayer%d'%(i+1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)
class _Up(nn.Module):
    def __init__(self, input_channel, output_channel, type_ = "pixel_shuffle", stride = 2):
        super(_Up, self).__init__()
        if stride == 1:
            self.up = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1, bias=False)
        if type_ == "pixel_shuffle":
            if input_channel != output_channel:
                self.up = nn.Sequential(
                    nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1, bias=False),
                    PixelShuffle_ICNR(output_channel)
                )
            else:
                self.up = PixelShuffle_ICNR(output_channel)
        elif type_ == "billinear":
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode = "bilinear", align_corners=True),
                nn.Conv2d(input_channel, output_channel, 3, padding=1)
            )
        else:
            self.up = nn.ConvTranspose2d(input_channel, output_channel, kernel_size = 2, stride = stride)
    def forward(self, x):
        return self.up(x)

class _DenseBlockUp(nn.Module):
    def __init__(self, num_layers,
                       input_channel,
                       output_channel,
                       bottleneck_size = 4,
                       growth_rate = 32,
                       drop_rate = 0,
                       type_up = "pixel_shuffle",
                       last_layer = False):
        super(_DenseBlockUp, self).__init__()
        self.up = _Up(input_channel, output_channel, type_ = type_up)
        self.conv = nn.Conv2d(output_channel*2, output_channel, kernel_size = 1, stride=1, bias=False)
        if last_layer:
            self.dense_block = _Up(output_channel, output_channel, type_=type_up)
        else:
            self.dense_block = _DenseBlock(num_layers = num_layers, 
                                       input_channel = output_channel,
                                       bottleneck_size = bottleneck_size,
                                       growth_rate = growth_rate,
                                       drop_rate = drop_rate)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x1, x2], 1)
        x1 = self.conv(x1)
        return self.dense_block(x1)

class BackboneDense(Backbone):
    def __init__(self, encoder_args={}, decoder_args = {}):
        super(BackboneDense, self).__init__(encoder_args, decoder_args)
        self.features_name = ["layer3","layer2","layer1","maxpool","relu"]
        self.last_layer = "layer4"

        self.bottleneck_size = 4
        self.growth_rate = 32
        self.drop_rate = 0
        self.input_channel_dense1 = 64
        self.type_up = "pixel_shuffle"
        self.block_configs = [6, 12, 24, 16]

    def initial_decoder(self):
        input_channel = self.input_channel_dense1

        output_channel_encoder = []
        for num_layer in self.block_configs:
            output_channel = (input_channel + self.growth_rate*num_layer)
            input_channel = output_channel//2
            output_channel_encoder.append(output_channel)
        self.blocks = nn.ModuleList()
        input_channel = output_channel_encoder[-1]
        for i in range(len(self.block_configs)-2, -1, -1):
            num_layer = self.block_configs[i]
            output_channel = output_channel_encoder[i]
            block = _DenseBlockUp(num_layer,
                                  input_channel,
                                  output_channel,
                                  bottleneck_size = self.bottleneck_size ,
                                  growth_rate = self.growth_rate,
                                  drop_rate = self.drop_rate,
                                  type_up = self.type_up)
            self.blocks.append(block)
            input_channel = output_channel + num_layer*self.growth_rate
        up1 = _DenseBlockUp(num_layer, 
                            input_channel = input_channel, 
                            output_channel = self.input_channel_dense1,
                            type_up= self.type_up,
                            last_layer= True)
        self.blocks.append(up1)
        self.out_conv = nn.Conv2d(self.input_channel_dense1, 1, kernel_size=1, stride=1)
            
class BackboneDense121(BackboneDense):
    def __init__(self, encoder_args={}, decoder_args = {}):
        super(BackboneDense121, self).__init__(encoder_args, decoder_args)
        self.base_model = densenet121(**encoder_args).features

        self.features_name = ["denseblock3","denseblock2","denseblock1","relu0"]
        self.last_layer = "denseblock4"

        self.bottleneck_size = 4
        self.growth_rate = 32
        self.drop_rate = 0
        self.input_channel_dense1 = 64
        self.type_up = "pixel_shuffle"
        self.block_configs = [6, 12, 24, 16]

        self.initial_decoder()

