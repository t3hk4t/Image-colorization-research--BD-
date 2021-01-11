import torch
import numpy as np
import matplotlib
from modules_core import dummy_loader
import argparse
from tqdm import tqdm
from modules import torch_utils
import time
from modules import tensorboard_utils
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,5)

import torch.utils.data
torch.cuda.empty_cache()
parser = argparse.ArgumentParser(description='Model trainer')
parser.add_argument('-run_name', default=f'run_{time.time()}', type=str)
parser.add_argument('-sequence_name', default=f'../../seq_default', type=str)
parser.add_argument('-is_cuda', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-learning_rate', default=2e-3, type=float)
parser.add_argument('-batch_size', default=8, type=int)
parser.add_argument('-epochs', default=1000, type=int)
# TODO add more params and make more beautitfull cuz this file is a mess
args = parser.parse_args()

summary_writer = tensorboard_utils.CustomSummaryWriter(
    log_dir=f'{args.sequence_name}/{args.run_name}'
)


USE_CUDA = torch.cuda.is_available()
MAX_LEN = 200 # limit max number of samples otherwise too slow training (on GPU use all samples / for final training)
if USE_CUDA:
    MAX_LEN = None

dataset = dummy_loader.SyntheticNoiseDataset(augmented_directory=r'C:\Users\37120\Documents\BachelorThesis\Bachelor thesis\datasets\flickr30k_augmented_test0',
                                             greyscale_directory=r'C:\Users\37120\Documents\BachelorThesis\Bachelor thesis\datasets\flickr30k_images_greyscale_test')
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
dataset_train, dataset_test = torch.utils.data.random_split(dataset, [train_size, test_size])

data_loader_train = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=args.batch_size,
    shuffle=True
)
data_loader_test = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=args.batch_size,
    shuffle=False
)

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


model = Autoencoder()

torch_utils.init_parameters(model)

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

if USE_CUDA:
    model = model.cuda()
    loss_func = loss_func.cuda()

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
    ]:
        metrics[f'{stage}_{metric}'] = []


for epoch in range(1, args.epochs):

    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for sample in tqdm(data_loader):
            x = sample['greyscale_image']
            y = sample['augmented_image']
            x = x.float()
            y = y.float()
            if USE_CUDA:
                x = x.cuda()
                y = y.cuda()

            y_prim = model.forward(y)
            loss = loss_func.forward(y_prim, x)
            metrics_epoch[f'{stage}_loss'].append(loss.item()) # Tensor(0.1) => 0.1f
            summary_writer.add_scalar(
                tag=f'{stage}_loss',
                scalar_value=loss.item(),
                global_step=epoch,
            )

            if data_loader == data_loader_test:
                summary_writer.add_hparams(
                    hparam_dict=args.__dict__,
                    metric_dict={
                        'test_loss':loss.item()
                    }
                )
            else:
                summary_writer.add_hparams(
                    hparam_dict=args.__dict__,
                    metric_dict={
                        'train_loss': loss.item()
                    }
                )
            summary_writer.flush()


            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            loss = loss.cpu()
            y_prim = y_prim.cpu()
            x = x.cpu()
            y = y.cpu()

            np_y_prim = y_prim.data.numpy()

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 2)}')

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

summary_writer.close()
input('quit?')

# TODO - Logging at the end of epoch