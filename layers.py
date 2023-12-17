import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Based on https://auro-227.medium.com/writing-a-custom-layer-in-pytorch-14ab6ac94b77
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomLinear, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        # init weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init | # TODO use more elementary distribution?
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init
    
    def forward(self, x):
        w_times_x = torch.matmul(x, self.weights.t()) # matmul is used instead of mm because it supports broadcasting
        return torch.add(w_times_x, self.bias)  # w * x + b

class CustomConv2d(nn.Module):
    # input tensor [batch_size, in_channels, height, width]

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CustomConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.kernels = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        # init weights and biases (måske bare sæt til 0 eller 1 med 'all_ones_()')
        nn.init.kaiming_uniform_(self.kernels, a=math.sqrt(5)) # TODO use more elementary distribution?
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernels)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init
        
    def forward(self, x):
        x = convolution2d(x, self.kernels, self.bias, self.padding, self.stride)

        return x

# Muligvis tilføj 'dilation' parameter???
def convolution2d(input, kernels, bias, padding=0, stride=1):
    kernel_height, kernel_width = kernels.size(2), kernels.size(3)
    out_channels = kernels.size(0)

    # Adding int casting, since tensors are expected to be matrices of ints
    out_height = int((input.size(2) - kernels.size(2) + 2 * padding) / stride) + 1 # ( ( I - K + 2P ) / S ) + 1
    out_width = int((input.size(3) - kernels.size(3) + 2 * padding) / stride) + 1

    # Unfold the input tensor
    # This practically convolves over the values of the input tensor without multiplying by the kernels
    uinput = F.unfold(input, (kernel_height, kernel_width), padding=padding, stride=stride) # Unfolded tensor [batch_size, patch, block]
    reshaped_kernels = kernels.view(out_channels, -1).t()
    # Calculate output
    # For this we can simply ignore the dimention of batch_size, as it will be handled by broadcasting
    uoutput = uinput.transpose(1, 2).matmul(reshaped_kernels) # Transpose the tensor
    #uoutput = uoutput.matmul(kernels.view(kernels.size(0), -1).t()) # Perform the matrix multiplication with kernels(.matmul() supports broadcasting)
    uoutput = uoutput.transpose(1, 2) # Transpose the tensor back
    output = F.fold(uoutput, (out_height, out_width), (1, 1)) + bias.view(bias.size(0), 1, 1) # Fold back the tensor to complete the convolution

    # Add bias and return
    #output += bias.view(bias.size(0), 1, 1)
    return output
