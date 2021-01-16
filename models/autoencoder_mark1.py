import torch

class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential( #input (1x320x480)
            torch.nn.Conv2d(1, 32, kernel_size=3,padding=1, bias=False), # (32x320x480)
            torch.nn.ReLU(True),
            torch.nn.GroupNorm(16, 32),
            #torch.nn.BatchNorm2d(num_features=32),
            torch.nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2,bias=False), # (32x160x240)
            torch.nn.ReLU(True),
            torch.nn.GroupNorm(32,64),
            #torch.nn.BatchNorm2d(num_features=64),
            torch.nn.Conv2d(64, 64, kernel_size=4, padding=1, stride=2,bias=False), # (64x80x120)
            torch.nn.ReLU(True),
            torch.nn.GroupNorm(32, 64),
            #torch.nn.BatchNorm2d(num_features=64),
            torch.nn.Conv2d(64, 128, kernel_size=8, padding=1, dilation=2, bias=False), # (128x68x108)
            torch.nn.ReLU(True),
            torch.nn.GroupNorm(64, 128),
            #torch.nn.BatchNorm2d(num_features=128),
            torch.nn.Conv2d(128, 128, kernel_size=5, padding=0, dilation=2, stride=1, bias=False), # (128x60x100)
            torch.nn.ReLU(True),
            torch.nn.GroupNorm(64, 128),
            #torch.nn.BatchNorm2d(num_features=128),
            torch.nn.Conv2d(128, 256, kernel_size=4, padding=0,dilation=3,stride=2, bias=False), # (256x26x46)
            torch.nn.ReLU(True),
            #torch.nn.BatchNorm2d(num_features=256),
            torch.nn.GroupNorm(128, 256)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, kernel_size=4, padding=0,dilation=3,stride=2, bias=False),
            torch.nn.ReLU(True),
            torch.nn.GroupNorm(64, 128),
            #torch.nn.BatchNorm2d(num_features=128),
            torch.nn.ConvTranspose2d(128, 128, kernel_size=5, padding=0, dilation=2, stride=1, bias=False),
            torch.nn.ReLU(True),
            torch.nn.GroupNorm(64, 128),
            #torch.nn.BatchNorm2d(num_features=128),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=8, padding=1, dilation=2, bias=False),
            torch.nn.ReLU(True),
            torch.nn.GroupNorm(32, 64),
            #torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ConvTranspose2d(64, 64, kernel_size=4, padding=1, stride=2, bias=False),
            torch.nn.ReLU(True),
            torch.nn.GroupNorm(32, 64),
            #torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=4, padding=1, stride=2, bias=False),
            torch.nn.ReLU(True),
            torch.nn.GroupNorm(16, 32),
            #torch.nn.BatchNorm2d(num_features=32),
            torch.nn.ConvTranspose2d(32, 1, kernel_size=3,padding=1, bias=False),
            torch.nn.Sigmoid()
        )


    def forward(self, x):
        out = self.encoder.forward(x)
        out = self.decoder.forward(out)
        return out
