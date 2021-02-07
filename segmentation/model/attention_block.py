import torch
import torch.nn as nn
import torch.nn.functional as F
class SENet(nn.Module):
    def __init__(self, input_channel, reduction):
        super(SENet, self).__init__()

        self.global_average = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_channel, input_channel//reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(input_channel//reduction, input_channel)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):        
        x1 = self.global_average(x)
        x1 = self.flatten(x1)
        x1 = self.relu(self.fc1(x1))
        x1 = self.sigmoid(self.fc2(x1))
        return x * x1.view(x1.shape[:2]+(1, 1))


class CBAM_ChannelBlock(nn.Module):
    def __init__(self, channel, reduction):
        super(CBAM_ChannelBlock, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c = x.shape[:2]
        x_max = self.max_pool(x)
        x_avg = self.avg_pool(x)

        x_max = x_max.view(b, c)
        x_avg = x_avg.view(b, c)
        x_max = self.mlp(x_max)
        x_avg = self.mlp(x_avg)

        attention_channel = self.sigmoid(x_max + x_avg)
        return x * attention_channel.view((b, c, 1, 1))
class CBAM_SpatialBlock(nn.Module):
    def __init__(self, channel):
        super(CBAM_SpatialBlock, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size = 7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x_max = x.max(dim = 1, keepdim = True).values
        x_avg = x.mean(dim = 1, keepdim = True)

        attention = torch.cat([x_max, x_avg], dim = 1)
        # print(attention.shape)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        # print(x.shape, attention.shape)
        return x*attention
class CBAM(nn.Module):
    def __init__(self, channel, reduction):
        super(CBAM, self).__init__()
        self.channel_block = CBAM_ChannelBlock(channel, reduction)
        self.spatial_block = CBAM_SpatialBlock(channel)
    
    def forward(self, x):
        x = self.channel_block(x)
        x = self.spatial_block(x)
        return x
        