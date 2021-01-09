
import torch
import torch.nn as nn
import torch.nn.functional as F

def crop_combine(x1, x2):
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
    return x2
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
        x =  torch.cat([x2, x1], dim = 1)
        return self.conv_block(x)

class UpLayer(nn.Module):
    def __init__(self, input_channel, output_channel, bilinear = True, kernel_size =3, padding = 1):
        super(UpLayer, self).__init__()
        if bilinear:
            self.up_layer = nn.Sequential(
                nn.Upsample(scale_factor=2,
                            mode = "bilinear",
                            align_corners=True),
                nn.Conv2d(input_channel, output_channel, kernel_size = kernel_size, padding = padding)
            )
        else:
            self.up_layer = nn.ConvTranspose2d(input_channel, 
                                                input_channel,
                                                kernel_size = 2,
                                                stride = 2)
    def forward(self, x):
        return self.up_layer(x)

class Resnet18Block(nn.Module):
    def __init__(self, input_channel, output_channel, up_sample = False, padding = 0, bilinear = True):
        super(Resnet18Block, self).__init__()
        self.bilinear = bilinear
        self.up_sample = up_sample
        if up_sample:
            self.conv1 = UpLayer(input_channel, output_channel, kernel_size=3, padding= padding, bilinear = bilinear)
            self.up = nn.Sequential(
                UpLayer(input_channel, output_channel, kernel_size=1, padding= 0, bilinear = bilinear),
                nn.BatchNorm2d(output_channel)
            )
        else:
            self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding = padding)
            self.up  = nn.Conv2d(input_channel, output_channel, 3, padding = padding)           
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(output_channel, output_channel, 3, padding= padding)
        self.bn2 = nn.BatchNorm2d(output_channel)
    def forward(self, x):
        residual = self.up(x)
        # print(residual.shape,x.shape)
        x = self.relu(self.bn1(self.conv1(x)))
        # print(residual.shape,x.shape)
        x = self.bn2(self.conv2(x))
        x = torch.add(residual, x)
        # print(residual.shape,x.shape)
        return self.relu(x)

class Resnet18BlocksUp(nn.Module):
    def __init__(self, input_channel, output_channel, padding = 0, bilinear = True):
        super(Resnet18BlocksUp, self).__init__()
        self.block1 = Resnet18Block(input_channel, output_channel, up_sample = True, padding = padding, bilinear = bilinear)
        self.block2 = Resnet18Block(output_channel*2, output_channel, up_sample = False, padding = padding, bilinear = bilinear)
    def forward(self, x1, x2):
        x1 = self.block1(x1)
        x2 = crop_combine(x1, x2)
        x = torch.cat([x1, x2], 1)
        return self.block2(x)
