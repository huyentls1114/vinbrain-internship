import torch.nn as nn
import torch.nn.functional as F
import torch

class UnetVGG(nn.Module):
    def __init__( self, input_channel = 1, 
                        output_channel = 1, 
                        pretrained_weights = None,
                        batch_norm = True,
                        padding = 0,
                        bilinear = True):
        super(UnetVGG, self).__init__()
        self.pretrained_weights = pretrained_weights

        self.features_name = ['inc','down1', 'down2', 'down3']
        self.encoder = Encoder(input_channel, batch_norm, padding, bilinear)
        self.decoder = Decoder(output_channel, batch_norm, padding, bilinear)
    def forward(self, x):
        x, features = self.forward_backbone(x)
        x = self.decoder(x, features)
        return x
    def forward_backbone(self, x):
        features = {}
        for name, child in self.encoder.named_children():
            x = child(x)
            if name in self.features_name:
                features[name] = x
        return x, features
        
class Encoder(nn.Module):
    def __init__( self, input_channel = 1, 
                        batch_norm = True,
                        padding = 0,
                        bilinear = True):
        super(Encoder, self).__init__()
        self.inc = MultiConv([input_channel, 64, 64], batch_norm, padding)
        self.down1 = Down([64, 128, 128], batch_norm, padding)
        self.down2 = Down([128, 256, 256], batch_norm, padding)
        self.down3 = Down([256, 512, 512], batch_norm, padding)
        self.down4 = Down([512, 1024, 1024], batch_norm, padding)
    def forward(self, x):
        x1 = self.inc(x)
        # print('x1', x1.shape)
        x2 = self.down1(x1)
        # print('x2',x2.shape)
        x3 = self.down2(x2)
        # print('x3',x3.shape)
        x4 = self.down3(x3)
        # print('x4',x4.shape)
        x5 = self.down4(x4)
        return x5

class Decoder(nn.Module):
    def __init__( self, output_channel = 1, 
                        batch_norm = True,
                        padding = 0,
                        bilinear = True):
        super(Decoder, self).__init__()
        self.up1 = Up([1024, 512, 512], batch_norm, padding, bilinear)
        self.up2 = Up([512, 256, 256], batch_norm, padding, bilinear)
        self.up3 = Up([256, 128, 128], batch_norm, padding, bilinear)
        self.up4 = Up([128, 64, 64], batch_norm, padding, bilinear)
        self.outc = OutConv(64, output_channel)
    def forward(self, x, features):
        x = self.up1(x, features["down3"])
        x = self.up2(x, features["down2"])
        x = self.up3(x, features["down1"])
        x = self.up4(x, features["inc"])
        x = self.outc(x)
        # print(x.shape)
        return x

class MultiConv(nn.Module):
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


class Down(nn.Module):
    def __init__(self, list_channels, batch_norm = True, padding = 0):
        '''
        combine maxpooling and multi_convolution
        '''
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            MultiConv(list_channels, batch_norm, padding)
        )
    
    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, list_channels, batch_norm = True, padding = 0, bilinear = True):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                        nn.Upsample(scale_factor = 2, 
                                    mode = 'bilinear', 
                                    align_corners= True),
                        nn.Conv2d(list_channels[0], list_channels[0]//2, 3, padding= 1)
            )
            self.conv = MultiConv(list_channels, batch_norm, padding)
        else:
            self.up = nn.ConvTranspose2d(list_channels[0], 
                                        list_channels[0]//2, 
                                        kernel_size = 2, 
                                        stride=2)
            self.conv = MultiConv(list_channels, batch_norm, padding)
    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
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
        return self.conv(x)
class OutConv(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.out_conv = nn.Conv2d(input_channel, output_channel, kernel_size = 1)

    def forward(self, x):
        return self.out_conv(x)