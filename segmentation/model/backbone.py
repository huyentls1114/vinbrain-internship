from torchvision.models import resnet18

from .unet_vgg import UnetVGG
from .block import VGG16Block, UpBlock, Out
from .block import Resnet18BlocksUp, UpLayer

class BackboneOriginal:
    def __init__(self, net_args={}, bilinear = True):
        self.net_args = net_args
        
        self.base_model = UnetVGG(input_channel=1, output_channel=1, **net_args)
        self.encoder = self.base_model.encoder    
        self.features_name = ["down3", "down2", "down1", "inc"]
        self.list_channels = [1024, 512, 256, 128, 64, 1]
        self.up_class = UpBlock

class BackBoneResnet18:
    def __init__(self, net_args):
        self.net_args = net_args
        self.base_model = resnet18()
        self.features_name = ["layer3", "layer2", "layer1", "relu"]
        self.last_layer = "layer4"
        self.input_channel = 3
        self.list_channels = [512, 256, 128, 64, 64, 1]
        self.up_class = Resnet18BlocksUp
        self.out_conv_class = UpLayer