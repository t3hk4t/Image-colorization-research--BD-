import copy
import torch
from torch import nn
import torch.utils.data
import math
import torchvision
import modules.torch_utils as torch_utils


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.class_count = 1

        if args.model_type == 'deeplabv3_resnet50':
            self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
        elif args.model_type == 'deeplabv3_resnet101':
            self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        elif args.model_type == 'fcn_resnet50':
            self.model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
        elif args.model_type == 'fcn_resnet101':
            self.model = torchvision.models.segmentation.fcn_resnet101(pretrained=True)

        conv:torch.nn.Conv2d = self.model.classifier[-1]
        conv_new = torch.nn.Conv2d(256, self.class_count, kernel_size=1)
        conv_new.weight.data = conv.weight.mean(dim=0).unsqueeze(dim=0).data
        conv_new.bias.data = conv.bias.mean(dim=0).unsqueeze(dim=0).data
        self.model.classifier[-1] = conv_new

    def forward(self, input):

        if len(input.size()) == 3:
            input = input.unsqueeze(dim=1) #Batch, Channel, Width, Height
            input = input.repeat(1, 3, 1, 1)
        y_prim = self.model.forward(input)
        y_prim = torch.sigmoid(y_prim['out'])
        # aux classifier not used now
        return y_prim.squeeze(dim=1)