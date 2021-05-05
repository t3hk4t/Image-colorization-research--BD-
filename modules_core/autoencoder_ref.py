import random
import torch.nn.functional as F
import scipy
import torch
import torch.nn as nn
import numpy as np
import matplotlib
import modules.torch_utils as torch_utils
import torchvision
from tqdm import tqdm
import os

class TempConv( nn.Module ):
   def __init__(self, in_planes, out_planes, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1) ):
      super(TempConv, self).__init__()
      self.conv3d  = nn.Conv3d( in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding )
      self.bn      = nn.BatchNorm3d( out_planes )
   def forward(self, x):
      return F.elu( self.bn( self.conv3d( x ) ), inplace=False )

class Upsample( nn.Module ):
   def __init__(self, in_planes, out_planes, scale_factor=(1,2,2)):
      super(Upsample, self).__init__()
      self.scale_factor = scale_factor
      self.conv3d = nn.Conv3d( in_planes, out_planes, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1) )
      self.bn   = nn.BatchNorm3d( out_planes )
   def forward(self, x):
      return F.elu( self.bn( self.conv3d( F.interpolate(x, scale_factor=self.scale_factor, mode='trilinear', align_corners=False) ) ), inplace=False )


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            TempConv(1, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 0, 0)),
            TempConv(32, 96, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            TempConv(96, 96, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            TempConv(96, 192, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            TempConv(192, 192, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            TempConv(192, 192, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            TempConv(192, 192, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            TempConv(192, 192, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            Upsample(192, 96),
            TempConv(96, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            TempConv(32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            Upsample(32, 16),
            nn.Conv3d(16, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        )

        torch_utils.init_parameters(self)

    def forward(self, x):
        out = self.encoder.forward(x)
        out = x + out
        return torch.sigmoid(out)
