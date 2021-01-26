import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcite(nn.Module):
    def __init__(self, input_channel,
                       se_ratio = 0.25,
                       act_layer = nn.ReLU,
                       gate_fn = nn.Sigmoid()):
        super(SqueezeExcite, self).__init__()

class DepthWiseSeparableConv(nn.Module):
    def __init__(self, input_channel,
                       output_channel,
                       dw_kernel_size = 3,
                       pw_kernel_size = 1,
                       stride = 1,
                       padding = 1,
                       act_layer = nn.ReLU,
                       pw_act = False,
                       norm_layer = nn.BatchNorm2d):
        super(DepthWiseSeparableConv, self).__init__()

        