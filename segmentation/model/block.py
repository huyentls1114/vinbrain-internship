
import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG16Block(nn.Module):
    def __init__(self, list_channels, batch_norm = True, padding = 0):
        super().__init__()
        assert len(list_channels) > 1

        list_layers = []
        for i in range(len(list_channels) - 1):
            input_channel = list_channels[i]
            output_channel = list_channels[i+1]
            list_layers.append(nn.Conv2d(input_channel, output_channel, 3, padding = padding))
            if batch_norm:
                list_layers.append(nn.BatchNorm2d(output_channel))
            list_layers.append(nn.ReLU(inplace=True))
        self.multi_conv = nn.Sequential(*list_layers)
        
    def forward(self, x):
        return self.multi_conv(x)

class Out(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.out_conv = nn.Conv2d(input_channel, output_channel, kernel_size = 1)

    def forward(self, x):
        return self.out_conv(x)

class UpBlock(nn.Module):
    def __init__(self, list_channels, 
                       block_class = VGG16Block, 
                       net_args ={},
                       bilinear = True):
        super(UpBlock, self).__init__()
        assert len(list_channels) > 1
        input_channel = list_channels[0]

        
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2,
                            mode = "bilinear",
                            align_corners=True),
                nn.Conv2d(input_channel, input_channel//2, 3, padding=1)
            )
        else:
            self.up = nn.ConvTranspose2d(input_channel, 
                                        input_channel//2,
                                        kernel_size = 2,
                                        stride = 2)
        self.conv_block = block_class(list_channels, **net_args)
    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[1]
        diffX = x2.size()[3] - x1.size()[3]
        x2 = F.pad(x2, [
            -diffX//2, -diffX//2,
            -diffY//2, -diffY//2
        ])
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x2 = F.pad(x2, [
            -diffX, 0,
            -diffY, 0
        ])
        x = torch.cat([x2, x1], dim = 1)
        return self.conv_block(x)
