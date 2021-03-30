import torch
from torch import nn


a = torch.zeros((8,3,4,128,64)) # (B,C,S,W,H)
B, C,S,W,H = a.shape

layer = nn.Conv3d(in_channels=3, out_channels=5, kernel_size=3, padding=1)
pool = nn.MaxPool2d(2, 2)
up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


out = layer.forward(a)
print(out.shape)
out = out.permute(0,2,1,3,4)
out = out.reshape(B*S,*out.shape[-3:])
out = pool.forward(out)
print(out.shape)
out = up.forward(out)
print(out.shape)
out = out.view(B,S, *out.shape[-3:])
out = out.permute(0,2,1,3,4).contiguous()
print(out.shape)
print(out)