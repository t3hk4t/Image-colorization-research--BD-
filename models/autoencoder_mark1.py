import torch
import numpy as np
import matplotlib
from modules_core import dummy_loader
import torchvision
from tqdm import tqdm
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,5)

import torch.utils.data
import scipy.ndimage


USE_CUDA = torch.cuda.is_available()
MAX_LEN = 200 # limit max number of samples otherwise too slow training (on GPU use all samples / for final training)
#if USE_CUDA:
#    MAX_LEN = None

full_dataset = torch.utils.data.DataLoader(
    dataset=(dummy_loader.SyntheticNoiseDataset()),
    batch_size=16,
    shuffle=True,
)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
data_loader_train, data_loader_test = torch.utils.data.random_split(full_dataset, [train_size, test_size])


class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=5, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=4),
            torch.nn.Conv2d(4, 8, kernel_size=4, padding=1, stride=2, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.Conv2d(8, 8, kernel_size=7, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.Conv2d(8, 16, kernel_size=4, padding=1, stride=2, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.Conv2d(16, 16, kernel_size=4, padding=1, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.Conv2d(16, 32, kernel_size=4, padding=1, stride=2, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=32)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32, 16, kernel_size=4, padding=1, stride=2, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.ConvTranspose2d(16, 16, kernel_size=4, padding=1, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.ConvTranspose2d(16, 8, kernel_size=4, padding=1, stride=2, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.ConvTranspose2d(8, 8, kernel_size=7, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.ConvTranspose2d(8, 4, kernel_size=4, padding=1, stride=2, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=4),
            torch.nn.ConvTranspose2d(4, 1, kernel_size=5, bias=False),
            torch.nn.Sigmoid()
        )


    def forward(self, x):
        out = self.encoder.forward(x)
        out = self.decoder.forward(out)
        return out


model = Autoencoder()
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

if USE_CUDA:
    model = model.cuda()
    loss_func = loss_func.cuda()

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, 1000):

    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for sample in tqdm(data_loader):
            x = sample['greyscale_image']
            y = sample['augmented_image']

            if USE_CUDA:
                x = x.cuda()
                y = y.cuda()

            y_prim = model.forward(x)
            loss = loss_func.forward(y_prim, y)
            metrics_epoch[f'{stage}_loss'].append(loss.item()) # Tensor(0.1) => 0.1f

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

    plt.clf()
    plt.subplot(121) # row col idx
    plts = []
    c = 0
    for key, value in metrics.items():
        value = scipy.ndimage.gaussian_filter1d(value, sigma=2)
        plts += plt.plot(value, f'C{c}', label=key)
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])

    for i, j in enumerate([4, 5, 6, 16,17,18]):
        plt.subplot(4, 6, j)
        plt.imshow(x[i][0].T, cmap=plt.get_cmap('Greys'))
        plt.subplot(4, 6, j+6)
        plt.imshow(np_y_prim[i][0].T, cmap=plt.get_cmap('Greys'))

    plt.tight_layout(pad=0.5)
    plt.draw()
    plt.pause(0.001)

input('quit?')