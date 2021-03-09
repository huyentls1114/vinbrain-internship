import torch
import torch.nn as nn
import torch.nn.functional as F
class SENet(nn.Module):
    def __init__(self, input_channel, reduction):
        super(SENet, self).__init__()

        self.global_average = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        # import pdb; pdb.set_trace()
        self.fc1 = nn.Linear(input_channel, input_channel//reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(input_channel//reduction, input_channel)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):        
        x1 = self.global_average(x)
        x1 = self.flatten(x1)
        x1 = self.relu(self.fc1(x1))
        x1 = self.sigmoid(self.fc2(x1))
        return x * x1[...,None,None]


class CBAMChannelBlock(nn.Module):
    def __init__(self, channel, reduction):
        super(CBAMChannelBlock, self).__init__()
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

class CBAMSpatialBlock(nn.Module):
    def __init__(self, channel):
        super(CBAMSpatialBlock, self).__init__()

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
    def __init__(self, input_channel, reduction):
        super(CBAM, self).__init__()
        self.channel_block = CBAMChannelBlock(input_channel, reduction)
        self.spatial_block = CBAMSpatialBlock(input_channel)
    
    def forward(self, x):
        x = self.channel_block(x)
        x = self.spatial_block(x)
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self, input_channel, pooling = True, reduction = None):
        super(SelfAttentionBlock, self).__init__()
        self.theta = nn.Conv2d(input_channel, input_channel//2, kernel_size = 1, stride= 1)
        self.phi = nn.Conv2d(input_channel, input_channel//2, kernel_size = 1, stride= 1)
        self.g = nn.Conv2d(input_channel, input_channel//2, kernel_size = 1, stride=1)
        if pooling:
            self.pool1 = nn.MaxPool2d(2)
            self.pool2 = nn.MaxPool2d(2)
        else:
            self.pool1 = None
            self.pool2 = None
        self.conv = nn.Conv2d(input_channel//2, input_channel, kernel_size =1, stride=1)
    def forward(self, x):
        _theta = self.theta(x)
        _phi = self.phi(x)
        if self.pool1 is not None: 
            _phi = self.pool1(_phi)
        c, w, h = _theta.shape[1:]
        c2, w2, h2 = _phi.shape[1:]
        # print(w, h, c)
        _theta = _theta.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        _phi = _phi.flatten(start_dim=2, end_dim=3)
        # print(_theta.shape, _phi.shape)
        _mul1 = torch.matmul(_theta, _phi)
        _sofmax = torch.softmax(_mul1, 2)
        # print(_mul1.shape)
        _g = self.g(x)
        if self.pool2 is not None: 
            _g = self.pool2(_g)
        _g = _g.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        # print(_g.shape)

        _mult2 = torch.matmul(_sofmax, _g)
        _mult2 = _mult2.reshape(-1, w, h, c).permute(0, 3, 1, 2)
        _mult2 = self.conv(_mult2)
        # print((x+_mult2).shape)
        return x+ _mult2
        # print(_mult2.shape)

class SESelfAttentionBlock(nn.Module):
    def __init__(self, input_channel, reduction = None):
        super(SESelfAttentionBlock, self).__init__()
        self.global_average = nn.AdaptiveAvgPool2d(1)
        self.self_attention = SelfAttentionBlock(input_channel, pooling= False)
    
    def forward(self, x):        
        x1 = self.global_average(x)
        x1 = self.self_attention(x1)
        # print(x1.shape)
        # import pdb; pdb.set_trace()
        return x * x1