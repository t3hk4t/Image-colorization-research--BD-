import copy
import torch
from torch import nn
import torch.utils.data
import math

import modules.torch_utils as torch_utils


# pre-activated relu
# modified version of modules.block_resnet_2d_std2_new with different skip conv, because stride set to 1
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_conv_bias=False, kernel_size=3, stride=1):
        super().__init__()

        kernel_size_first = kernel_size
        padding = int(kernel_size/2)
        if kernel_size % 2 == 0:
            kernel_size_first = kernel_size - 1
            padding -= 1

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size_first,
                               stride=1,
                               padding=int(kernel_size_first/2), bias=is_conv_bias)
        self.gn1 = nn.GroupNorm(num_channels=in_channels, num_groups=math.ceil(in_channels/2))


        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride,
                               padding=padding, bias=is_conv_bias)
        self.gn2 = nn.GroupNorm(num_channels=out_channels, num_groups=math.ceil(out_channels/2))

        self.is_projection = False
        if stride > 1 or in_channels != out_channels:
            self.is_projection = True
            self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                                      padding=1, bias=is_conv_bias)

    def forward(self, x):
        # Batch, Channel, W
        residual = x

        out = self.conv1(x)
        out = torch.relu(out)
        out = self.gn1(out)

        out = self.conv2(out)

        if self.is_projection:
            residual = self.conv_res(x)

        out += residual
        out = torch.relu(out)
        out = self.gn2(out)

        return out


# unet++ with resblocks
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args

        self.deep_supervision = self.args.is_deep_supervision
        self.depth = self.args.unet_depth
        self.first_layer_channels = self.args.first_conv_channel_count
        self.expand_rate = self.args.expansion_rate
        self.class_count = 1

        self.output_count = 1
        if self.deep_supervision:
            self.output_count = self.depth - 1

        self.channels = [1, self.first_layer_channels]
        for d in range(self.depth-1):
            self.channels.append(self.channels[-1]*self.expand_rate)
        # channels = [1, 16, 32, 64, 128, 256]

        # up and down layers
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.module_depth_list = torch.nn.ModuleList()
        for idx_depth in range(self.depth):
            module_list = torch.nn.ModuleList()
            for idx in range(self.depth-idx_depth):
                channels_in = self.channels[idx]
                if idx_depth > 0:
                    channels_in = self.channels[idx+1] * idx_depth + self.channels[idx+2]
                module_list.append(ResBlock(channels_in, self.channels[idx+1]))
            self.module_depth_list.append(module_list)

        self.output_modules_list = torch.nn.ModuleList()
        for _ in range(self.output_count):
            self.output_modules_list.append(nn.Conv2d(self.channels[1], self.class_count, kernel_size=1, bias=False))

        torch_utils.init_parameters(self)

    def forward(self, input):
        input = input.unsqueeze(1) #Batch, Channel, Width, Height

        # container for output values
        intermediate_outputs = [[]for _ in range(self.depth)]

        # calculate unet++ intermediate values
        for idx_depth in range(self.depth):
            depth_modules = self.module_depth_list[idx_depth]
            for idx in range(len(depth_modules)):
                # input for first conv
                if idx_depth == 0 and idx == 0:
                    x = input
                # input for first depth (model backbone)
                elif idx_depth == 0 and idx > 0:
                    x = self.pool(intermediate_outputs[idx-1][0])
                # input for rest of the modules
                else:
                    x = copy.copy(intermediate_outputs[idx])
                    x.append(self.up(intermediate_outputs[idx+1][idx_depth-1]))
                    x = torch.cat(x, 1)
                module = depth_modules[idx]
                out = module(x)
                intermediate_outputs[idx].append(out)

        # calculate output values
        output = 0
        for idx in range(self.output_count):
            x = intermediate_outputs[0][-(idx+1)]
            module = self.output_modules_list[idx]
            output += module(x)
        output /= self.output_count
        # added for more stable output
        output = torch.sigmoid(output)

        return output.squeeze(1)
