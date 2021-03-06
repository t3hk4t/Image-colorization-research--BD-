import copy
import torch
from torch import nn
import torch.utils.data
import math

import modules.torch_utils as torch_utils



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_conv_bias=False, kernel_size=(3,3,3), stride=1, is_first_layer=False, is_backbone = False):
        super().__init__()

        kernel_size_first = kernel_size
        padding = (0, int(kernel_size[1]/2), int(kernel_size[1]/2))
        if kernel_size[1] % 2 == 0:
            kernel_size_first = (1, kernel_size[1] - 1, kernel_size[1] - 1)
            padding = (0, padding[1] -1, padding[2] -1 )

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size,
                               stride=1,
                               padding=(0, int(kernel_size_first[1]/2) ,int(kernel_size_first[1]/2)), bias=is_conv_bias)

        # added hack for images with channel count not dividable by 2
        num_groups = math.ceil(in_channels/2)
        if is_first_layer:
            num_groups = in_channels
        self.gn1 = nn.GroupNorm(num_channels=in_channels, num_groups=num_groups)


        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride,
                               padding=padding, bias=is_conv_bias)
        self.gn2 = nn.GroupNorm(num_channels=out_channels, num_groups=math.ceil(out_channels/2))

        self.is_projection = False
        if stride > 1 or in_channels != out_channels:
            self.is_projection = True
            self.conv_res = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=(0,1,1), bias=is_conv_bias)

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
                    module_list.append(ResBlock(channels_in, self.channels[idx+1], kernel_size = (3,3,3)))
                else:
                    module_list.append(ResBlock(channels_in, self.channels[idx + 1], kernel_size = (3,3,3)))
            self.module_depth_list.append(module_list)

        self.output_modules_list = torch.nn.ModuleList()
        for _ in range(self.output_count):
            self.output_modules_list.append(nn.Conv3d(self.channels[1], self.class_count, kernel_size=(1, 1, 1), bias=False))

        torch_utils.init_parameters(self)

    def forward(self, input):
        if len(input.size()) == 3:
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
                    x = intermediate_outputs[idx-1][0] # BxCxDxHxW
                    B, C, D, H, W = x.shape
                    x = x.permute(0,2,1,3,4)
                    x = x.reshape(B*D, *x.shape[-3:])
                    x = self.pool(x)
                    x = x.view(B, D, *x.shape[-3:])
                    x = x.permute(0,2,1,3,4)
                # input for rest of the modules
                else:
                    x = copy.copy(intermediate_outputs[idx])

                    x1 = intermediate_outputs[idx+1][idx_depth-1]
                    B, C, D, H, W = x1.shape
                    x1 = x1.permute(0, 2, 1, 3, 4)
                    x1 = x1.reshape(B * D, *x1.shape[-3:])
                    x1 = self.up(x1)
                    x1 = x1.view(B, D, *x1.shape[-3:])
                    x1 = x1.permute(0, 2, 1, 3, 4)
                    x.append(x1)
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
