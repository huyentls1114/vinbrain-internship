import torch
import torch.nn as nn
import torch.nn.functional as functional

from .backbone import Backbone
from fastai.layers import PixelShuffle_ICNR
from torchvision.models import resnet18, resnet34, resnet50, resnet101

def up(input_channel, output_channel, type_ = "pixel_shuffle", stride = 2, kernel_size = 3):
    '''
    type_ is one of [billinear, deconv, pixel_shuffle]
    '''
    if stride == 1:
        if kernel_size == 3:
            return conv3x3(input_channel, output_channel)
        else:
            return conv1x1(input_channel, output_channel)
    if type_ == "pixel_shuffle":
        if input_channel!= output_channel:
            return nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size = 1, padding = 0),
                PixelShuffle_ICNR(output_channel)
            )
        else:
            return PixelShuffle_ICNR(output_channel)
    elif type_ == "billinear":
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode = "bilinear", align_corners=True),
            nn.Conv2d(input_channel, output_channel, 3, padding=1)
        )
    else:
        return nn.ConvTranspose2d(input_channel, output_channel, kernel_size = 2, stride = stride)

def conv3x3(input_channel, output_channel, stride = 1, groups =1, padding = 1):
    return nn.Conv2d(input_channel, output_channel,kernel_size=3, stride = stride, padding = padding, groups= groups, bias= False)
def conv1x1(input_channel, output_channel, stride = 1):
    return nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=stride)

class BasicBlockUp(nn.Module):
    expansion = 1
    def __init__(self,
                 input_channel,
                 output_channel,
                 stride = 1,
                 upsample = None,
                 type_up = "pixel_shuffle",
                 padding = 1,
                 middle_channel = None):
        if middle_channel == None:
            middle_channel = output_channel
        super(BasicBlockUp, self).__init__()
        if upsample is None:
            self.conv1 = conv3x3(input_channel, middle_channel, stride, padding= padding)
        else:
            self.conv1 = up(input_channel, middle_channel, type_ = type_up, stride= stride)
        self.bn1 = nn.BatchNorm2d(middle_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(middle_channel, output_channel)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.upsample = upsample
        self.stride = stride
    def forward(self, x):
        residual = x
        if self.upsample is not None:
            residual = self.upsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x+=residual
        return self.relu(x)
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, input_channel, 
                       output_channel, 
                       stride = 1,
                       upsample = None,
                       type_up = "pixel_shuffle",
                       middle_channel = None):
        super(Bottleneck, self).__init__()
        if middle_channel is None:
            middle_channel = input_channel//self.expansion
        self.conv1 = conv1x1(input_channel, middle_channel)
        self.bn1 = nn.BatchNorm2d(middle_channel)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        if upsample is None:
            self.conv2 = conv3x3(middle_channel, middle_channel, stride=stride)
        else:
            self.conv2 = up(middle_channel, middle_channel, stride=stride, type_=type_up)
        self.bn2 = nn.BatchNorm2d(middle_channel)
        self.conv3 = conv1x1(middle_channel, output_channel)
        self.bn3 = nn.BatchNorm2d(output_channel)
    def forward(self, x):
        residual = x
        if self.upsample is not None:
            residual = self.upsample(x)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        # import pdb; pdb.set_trace()
        # print(x.shape, residual.shape)
        x+=residual
        return self.relu(x)

class ResnetLayerUp(nn.Module):
    def __init__(self, block,
                       input_channel, 
                       output_channel,
                       num_block,
                       type_up = "pixel_shuffle",
                       stride = 1,
                       last_layer = False):
        super(ResnetLayerUp, self).__init__()
        self.type_up = type_up
        self.last_layer = last_layer
        if last_layer:
            self.upsample = nn.Sequential(up(input_channel, output_channel, type_=type_up), nn.BatchNorm2d(64))
            self.first_block = block(input_channel*2, output_channel, stride = 1, upsample = conv1x1(input_channel*2, output_channel))
        else:
            self._make_layers(block, input_channel, output_channel, num_block, stride)
    def forward(self, x1, x2):
        if self.last_layer:
            x1 = self.upsample(x1)
            # print(x1.shape)
            x1 = torch.cat([x1, x2], 1)
            # print(x1.shape, x2.shape)
            return self.first_block(x1)
        else:
            # import pdb; pdb.set_trace()
            x1 = self.up_block(x1)
            x1 = torch.cat([x1, x2], 1)
            x1 = self.first_block(x1)
            return self.layers(x1)

    def _make_layers(self, block,
                           input_channel,
                           output_channel,
                           num_block,
                           stride = 1):
        upsample = nn.Sequential(
            up(input_channel, output_channel, stride= stride, type_=self.type_up),
            nn.BatchNorm2d(output_channel)
        )
        layers = []
        self.up_block = block(input_channel, output_channel, stride = stride, upsample = upsample, type_up = self.type_up)
        self.first_block = block(output_channel*2, output_channel, upsample = conv1x1(output_channel*2, output_channel), middle_channel = output_channel//block.expansion)
        for i in range(2, num_block):
            layer = block(output_channel, output_channel, type_up = self.type_up)
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

class ResnetUp(nn.Module):
    def __init__(self, block = BasicBlockUp,
                       layers = [2, 2, 2, 2],
                       type_up = "pixel_shuffle"):
        super(ResnetUp, self).__init__()
        self.type_up = type_up
        
        self.layer1 = ResnetLayerUp(block, 128*block.expansion, 64*block.expansion, layers[-4], stride = 2, type_up =type_up)
        self.layer2 = ResnetLayerUp(block, 256*block.expansion, 128*block.expansion, layers[-3], stride = 2, type_up =type_up)
        self.layer3 = ResnetLayerUp(block, 512*block.expansion, 256*block.expansion, layers[-2], stride = 2, type_up =type_up)
        
        self.up1 = ResnetLayerUp(block, 64*block.expansion, 64, 2, stride = 1, type_up =type_up)
        self.up2 = ResnetLayerUp(block, 64, 64, 2, stride = 2, type_up =type_up, last_layer= True)
        self.out_conv = nn.Sequential(
            up(64, 64),
            conv1x1(64, 1)
        )   
        # self.init_weight()
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
            elif isinstance(m, BasicBlockUp):
                nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

class BackboneResnet(Backbone):
    def __init__(self, encoder_args, decoder_args):
        super(BackboneResnet, self).__init__(encoder_args, decoder_args)
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args
        self.resnet_up = None
        self.features_name = ["layer3","layer2","layer1","maxpool","relu"]
        self.last_layer = "layer4"

    def initial_decoder(self):
        self.blocks = nn.ModuleList()
        self.blocks.append(self.resnet_up.layer3)
        self.blocks.append(self.resnet_up.layer2)
        self.blocks.append(self.resnet_up.layer1)
        self.blocks.append(self.resnet_up.up1)
        self.blocks.append(self.resnet_up.up2)
        self.out_conv = self.resnet_up.out_conv

class BackboneResnet18(BackboneResnet):
    def __init__(self, encoder_args, decoder_args):
        super(BackboneResnet18, self).__init__(encoder_args, decoder_args)
        self.resnet_up = ResnetUp(BasicBlockUp, layers = [2, 2, 2, 2], **decoder_args)
        self.base_model = resnet18(**encoder_args)
        self.initial_decoder()

class BackboneResnet101(BackboneResnet):
    def __init__(self, encoder_args, decoder_args):
        super(BackboneResnet101, self).__init__(encoder_args, decoder_args)
        self.resnet_up = ResnetUp(Bottleneck, layers = [3, 4, 23, 3], **decoder_args)
        self.base_model = resnet101(**encoder_args)
        self.initial_decoder()

class BackboneResnet34(BackboneResnet):
    def __init__(self, encoder_args, decoder_args):
        super(BackboneResnet34, self).__init__(encoder_args, decoder_args)
        self.resnet_up = ResnetUp(BasicBlockUp, layers = [3, 4, 6, 3], **decoder_args)
        self.base_model = resnet34(**encoder_args)
        self.initial_decoder()
class BackboneResnet50(BackboneResnet):
    def __init__(self, encoder_args, decoder_args):
        super(BackboneResnet50, self).__init__(encoder_args, decoder_args)
        self.resnet_up = ResnetUp(Bottleneck, layers = [3, 4, 6, 3], **decoder_args)
        self.base_model = resnet50(**encoder_args)
        self.initial_decoder()