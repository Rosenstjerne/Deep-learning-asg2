import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Based on https://auro-227.medium.com/writing-a-custom-layer-in-pytorch-14ab6ac94b77
# TODO Plagiat pass
class CustomLinear(nn.module):
    def __init__(self, in_features, out_features):
        super(CustomLinear, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        # init weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init
    
    def forward(self, x):
        w_times_x = torch.matmul(x, self.weights.t()) # matmul is used instead of mm because it supports broadcasting
        return torch.add(w_times_x, self.bias)  # w * x + b

class CustomConv2d(nn.module):
    # input tensor [batch_size, in_channels, height, width]

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CustomConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = [padding] * 4

        self.kernels = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.bias = nn.Parameter(bias = torch.Tensor(out_channels))

        # init weights and biases (måske bare sæt til 0 eller 1 med 'all_ones_()')
        nn.init.kaiming_uniform_(self.kernels, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init
        
    def forward(self, x):

        # Shape of output convolution (tror ikke det her er korrekt)
        out_height = int(((x.size(dim=2) - self.kernels.size(dim=2) + 2 * self.padding) / self.stride) + 1)
        out_width = int(((x.size(dim=3) - self.kernels.size(dim=3) + 2 * self.padding) / self.stride) + 1)
        output = nn.Tensor.new_zeros((x.size(dim=0), self.out_channels, out_height, out_width))

        # Apply padding
        x = F.pad(x, self.padding)

        # Ikke forbudt af opgaven
        x = F.conv2d(x, self.kernels, self.bias, self.stride, self.padding)

        return x

# Tilføj padding og stride (muligvis også dilation :))
def convolution2d(input, kernels, bias):
    out_height = (input.size(2) - kernels.size(2)) + 1
    out_width = (input.size(3) - kernels.size(3)) + 1

    uinput = F.unfold(input, (kernels.size(2), kernels.size(3))) # Unfolded tensor [batch_size, patch, block]
    uoutput = uinput.transpose(1, 2).matmul(kernels.view(kernels.size(0), -1).t()).transpose(1, 2) # Expand?
    output = F.fold(uoutput, (out_height, out_width), (1, 1)) # use .view() instead?
    output += bias.view(bias.size(0), 1, 1) # Add bias
    return output
