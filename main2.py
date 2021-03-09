import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import numpy as np
from models import unetplusplus
import matplotlib
from modules_core import dummy_loader
import argparse
from skimage import color
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision
import time
from modules import tensorboard_utils
from modules import radam
import torchnet as tnt
matplotlib.use('TkAgg')
import torch.utils.data

def main():
    parser = argparse.ArgumentParser(description='Model trainer')
    parser.add_argument('-run_name', default=f'run_{time.time()}', type=str)
    parser.add_argument('-sequence_name', default=f'../new_folder', type=str)
    parser.add_argument('-is_cuda', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-learning_rate', default=1e-4, type=float)
    parser.add_argument('-batch_size', default=2, type=int)
    parser.add_argument('-data_workers', default=1, type=int)
    parser.add_argument('-epochs', default=1000, type=int)
    parser.add_argument('-is_deep_supervision', default=True, type=bool)
    parser.add_argument('-unet_depth', default=5, type=int)
    parser.add_argument('-first_conv_channel_count', default=8, type=int)
    parser.add_argument('-expansion_rate', default=3, type=int)
    # TODO add more params and make more beautitfull cuz this file is a mess
    args = parser.parse_args()

    writer = SummaryWriter(f'{args.sequence_name}/{args.run_name}')
    tensorboard_writer = tensorboard_utils.TensorBoardUtils(writer)
    USE_CUDA = torch.cuda.is_available()
    MAX_LEN = 200  # limit max number of samples otherwise too slow training (on GPU use all samples / for final training)
    if USE_CUDA:
        MAX_LEN = None
    data_loader_train, data_loader_test = dummy_loader.get_data_loaders(args)

    model = unetplusplus.Model(args)
    loss_func = torch.nn.MSELoss()
    optimizer = radam.RAdam(model.parameters(), lr=args.learning_rate)

    if USE_CUDA:
        model = model.cuda()
        loss_func = loss_func.cuda()

    metrics = {}

    image_metrics = {
        'Greyscale_image': [],
        'Augmented_image': []
    }

    for stage in ['train', 'test']:
        for metric in [
            'loss',
        ]:
            metrics[f'{stage}_{metric}'] = []

    epoch_test_loss = 0.0
    epoch_train_loss = 0.0
    running_test_loss = 0.0
    running_train_loss = 0

    # todo complete state and meters for your metrics
    state = {
        'train_loss': -1,
        'test_loss': -1
    }

    meters = dict(
        train_loss=tnt.meter.AverageValueMeter(),
        test_loss=tnt.meter.AverageValueMeter(),
    )
    i = 0
    for epoch in range(1, args.epochs):
        for key in meters.keys():
            meters[key].reset()

        for data_loader in [data_loader_train, data_loader_test]:
            metrics_epoch = {key: [] for key in metrics.keys()}
            n_total_steps = 0
            stage = 'train'
            if data_loader == data_loader_test:
                stage = 'test'

            for sample in tqdm(data_loader):
                y = sample['greyscale_image']
                x = sample['augmented_image']

                i += 1
                y = y.float()
                x = x.float()
                if USE_CUDA:
                    x = x.cuda()
                    y = y.cuda()
                y_prim = model.forward(x)
                loss = loss_func.forward(y_prim, y)
                metrics_epoch[f'{stage}_loss'].append(loss.item())  # Tensor(0.1) => 0.1f

                if data_loader == data_loader_train:
                    running_train_loss += loss.item()
                else:
                    running_test_loss += loss.item()

                meters[f'{stage}_loss'].add(np.average(loss.item()))
                if data_loader == data_loader_train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                loss = loss.cpu()
                y_prim = y_prim.cpu()
                x = x.cpu()
                y = y.cpu()

                if i % 100 == 0:
                    np_y_prim = torch.squeeze(y_prim, axis=0)
                    np_x = torch.squeeze(x, axis=0)
                    x_2 = torch.squeeze(y, axis=0)


                    #data = torch.cat([x_2, np_x, np_y_prim], 1)
                    #img_grid_validate = torchvision.utils.make_grid(data)
                    #writer.add_image(f'Validate{i}', data, dataformats='HW')

                    if data_loader == data_loader_train:
                        writer.add_scalar('Train loss', running_train_loss / 100,
                                          epoch * len(data_loader_train) + n_total_steps)

                    else:
                        writer.add_scalar('Test loss', running_test_loss / 100,
                                          epoch * len(data_loader_train) + n_total_steps)
                    running_train_loss = 0.0
                    running_test_loss = 0.0
                n_total_steps += 1

            state[f'{stage}_loss'] = meters[f'{stage}_loss'].value()[0]
            metrics_strs = []
            for key in metrics_epoch.keys():
                if stage in key:
                    value = np.mean(metrics_epoch[key])
                    if stage == 'train':
                        epoch_train_loss = value
                    else:
                        epoch_test_loss = value
                    metrics[key].append(value)
                    metrics_strs.append(f'{key}: {round(value, 2)}')

            print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    writer.close()
    input('quit?')


if __name__ == '__main__':
    main()
# TODO - Logging at the end of epoch