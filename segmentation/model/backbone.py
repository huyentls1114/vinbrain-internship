from .unet_vgg import UnetVGG
from .block import VGG16Block

class BackboneOriginal():
    def __init__(self, net_args={}):
        self.net_args = net_args
        
        self.base_model = UnetVGG(input_channel=1, output_channel=1, **net_args)
        self.block_class = VGG16Block
        self.encoder = self.base_model.encoder    
        self.features_name = ["down3", "down2", "down1", "inc"]
        self.list_channels = [1024, 512, 256, 128, 64, 1]
