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
from modules import logging_utils
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
    parser.add_argument('-sequence_name', default=f'../new', type=str)
    parser.add_argument('-is_cuda', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-learning_rate', default=1e-4, type=float)
    parser.add_argument('-batch_size', default=3, type=int)
    parser.add_argument('-path_train', default=[r'C:\Users\37120\Documents\BachelorThesis\image_data\dataset_test_2\train'], nargs='*')
    parser.add_argument('-path_test', default=[r'C:\Users\37120\Documents\BachelorThesis\image_data\dataset_test_2\test'], nargs='*')
    parser.add_argument('-data_workers', default=1, type=int)
    parser.add_argument('-epochs', default=1000, type=int)
    parser.add_argument('-is_deep_supervision', default=True, type=bool)
    parser.add_argument('-unet_depth', default=5, type=int)
    parser.add_argument('-first_conv_channel_count', default=8, type=int)
    parser.add_argument('-expansion_rate', default=2, type=int)
    parser.add_argument('-continue_training', default=True, type=bool)
    parser.add_argument('-saved_model_path', default=r'C:\Users\37120\Documents\BachelorThesis\Bachelor thesis\save4', type=str)
    # TODO add more params and make more beautitfull cuz this file is a mess
    args = parser.parse_args()

    tensorboard_writer = tensorboard_utils.CustomSummaryWriter(log_dir=f'{args.sequence_name}/{args.run_name}')
    tensorboard_utilz = tensorboard_utils.TensorBoardUtils(tensorboard_writer)
    logging_utilz = logging_utils.LoggingUtils(filename=f'{args.sequence_name}/{args.run_name}.txt')
    last_epoch = 1

    USE_CUDA = torch.cuda.is_available()
    MAX_LEN = 200  # limit max number of samples otherwise too slow training (on GPU use all samples / for final training)
    if USE_CUDA:
        MAX_LEN = None
    data_loader_train, data_loader_test = dummy_loader.get_data_loaders(args)
    model = unetplusplus.Model(args)
    loss_func = torch.nn.MSELoss()
    optimizer = radam.RAdam(model.parameters(), lr=args.learning_rate)


    if(args.continue_training):
        checkpoint = torch.load(args.saved_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        model.train()

    if USE_CUDA:
        model = model.cuda()
        loss_func = loss_func.cuda()
    metrics = {}

    for stage in ['train', 'test']:
        for metric in [
            'loss',
        ]:
            metrics[f'{stage}_{metric}'] = []

    meters = dict(
        train_loss=tnt.meter.AverageValueMeter(),
        test_loss=tnt.meter.AverageValueMeter(),
    )
    i = 0

    hP = args.__dict__
    hP['path_train'] = ''
    hP['path_test'] = ''

    for epoch in range(last_epoch, args.epochs):
        for key in meters.keys():
            meters[key].reset()

        for data_loader in [data_loader_train, data_loader_test]:
            metrics_epoch = {key: [] for key in metrics.keys()}
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

                if data_loader == data_loader_train:
                    optimizer.zero_grad()
                    model.zero_grad()
                    y_prim = model.forward(x)
                    loss = loss_func.forward(y_prim, y)
                else:
                    with torch.no_grad():
                        y_prim = model.forward(x)
                        loss = loss_func.forward(y_prim, y)
                metrics_epoch[f'{stage}_loss'].append(loss.item())  # Tensor(0.1) => 0.1f
                if data_loader == data_loader_train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                loss = loss.cpu()
                y_prim = y_prim.cpu()
                x = x.cpu()
                y = y.cpu()

                meters[f'{stage}_loss'].add(loss.item())

                if i % 500 == 0:
                    np_y_prim = torch.squeeze(y_prim, axis=0)
                    np_x = torch.squeeze(x, axis=0)
                    x_2 = torch.squeeze(y, axis=0)

                    data = torch.cat([x_2[0,:,:], np_x[0,:,:], np_y_prim[0,:,:]], 1)
                    img_grid_validate = torchvision.utils.make_grid(data)
                    tensorboard_writer.add_image(f'Validate{i}', data, dataformats='HW')
                    break

        tensorboard_writer.add_hparams(
            hparam_dict=hP,
            metric_dict={
                'Train_loss': float(meters['train_loss'].value()[0]),
                'Test_loss': float(meters['test_loss'].value()[0])
            },
            global_step=epoch
        )


        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"./save{epoch}")
        tensorboard_writer.flush()
    tensorboard_writer.close()
    input('quit?')


if __name__ == '__main__':
    main()
# TODO - Logging at the end of epoch