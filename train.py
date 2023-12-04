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
        w_times_x= torch.matmul(x, self.weights.t()) # matmul is used instead of mm because it supports broadcasting
        return torch.add(w_times_x, self.bias)  # w * x + b

class CustomConv2d(nn.module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CustomConv2d, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(bias = torch.Tensor(out_channels))
        
        
