import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import numpy as np
import matplotlib
import logging
import argparse
import copy
import math
from skimage import color
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision
import time
from datetime import datetime
import torchnet as tnt
matplotlib.use('TkAgg')
import torch.utils.data

from modules.file_utils import FileUtils
from modules import tensorboard_utils
from modules import radam
from modules import logging_utils
from modules.loss_functions import *
from modules_core import conv3d_dataloader
from models import unetplusplus
from models import temporal_unet_plus_pus
from models import DrunkUNET
from modules.csv_utils_2 import CsvUtils2


def main():
    parser = argparse.ArgumentParser(description='Model trainer')
    parser.add_argument('-run_name', default=f'run_{time.time()}', type=str)
    parser.add_argument('-sequence_name', default=f'temporal_unet_memmap3', type=str)
    parser.add_argument('-is_cuda', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-learning_rate', default=3e-4, type=float)
    parser.add_argument('-batch_size', default=4, type=int)
    parser.add_argument('-path_train', default=['C:\\Users\\vecin\\Documents\\PycharmProjects\\Research\\Image-colorization-research--BD-\\data\\train'], nargs='*')
    parser.add_argument('-path_test', default=['C:\\Users\\vecin\\Documents\\PycharmProjects\\Research\\Image-colorization-research--BD-\\data\\test'], nargs='*')
    parser.add_argument('-model', default='model_2_pretrained', type=str)
    parser.add_argument('-model_type', default='fcn_resnet50', type=str) # for pretrained
    parser.add_argument('-datasource', default='conv2D_dataloader', type=str)
    parser.add_argument('-data_workers', default=1, type=int)
    parser.add_argument('-epochs', default=50, type=int)
    parser.add_argument('-is_deep_supervision', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-is_debug', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-unet_depth', default=5, type=int)
    parser.add_argument('-first_conv_channel_count', default=2, type=int)
    parser.add_argument('-expansion_rate', default=2, type=int)
    parser.add_argument('-continue_training', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-conv3d_depth', default=5, type=int) #ammount of pictures in conv3d

    parser.add_argument('-early_stopping_patience', default=10, type=int)
    parser.add_argument('-early_stopping_param', default='test_loss', type=str)
    parser.add_argument('-early_stopping_delta_percent', default=0.005, type=float)

    # TODO add more params and make more beautitfull cuz this file is a mess
    args, _ = parser.parse_known_args()

    path_sequence = f'./results/{args.sequence_name}'
    args.run_name += ('-' + datetime.utcnow().strftime(f'%y-%m-%d--%H-%M-%S'))
    path_run = f'./results/{args.sequence_name}/{args.run_name}'
    FileUtils.createDir(path_run)
    FileUtils.writeJSON(f'{path_run}/args.json', vars(args))
    USE_CUDA = torch.cuda.is_available()
    CsvUtils2.create_global(path_sequence)
    CsvUtils2.create_local(path_sequence, args.run_name)

    rootLogger = logging.getLogger()
    logFormatter = logging.Formatter("%(asctime)s [%(process)d] [%(thread)d] [%(levelname)s]  %(message)s")
    rootLogger.level = logging.DEBUG  # level
    base_name = os.path.basename(path_sequence)
    fileHandler = logging.FileHandler(f'{path_run}/log-{base_name}.txt')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    tensorboard_writer = tensorboard_utils.CustomSummaryWriter(log_dir=path_run)

    get_data_loaders = getattr(__import__('modules_core.' + args.datasource, fromlist=['get_data_loaders']),
                               'get_data_loaders')
    data_loader_train, data_loader_test = get_data_loaders(args)

    Model = getattr(__import__('modules_core.' + args.model, fromlist=['Model']), 'Model')
    model = Model(args)

    loss_func = torch.nn.MSELoss()
    optimizer = radam.RAdam(model.parameters(), lr=args.learning_rate)

    if USE_CUDA:
        model = model.cuda()
        loss_func = loss_func.cuda()

    meters = dict(
        train_loss=tnt.meter.AverageValueMeter(),
        test_loss=tnt.meter.AverageValueMeter(),
        train_MS_SSIM_L1=tnt.meter.AverageValueMeter(),
        test_MS_SSIM_L1=tnt.meter.AverageValueMeter(),
        train_mse=tnt.meter.AverageValueMeter(),
        test_mse=tnt.meter.AverageValueMeter(),
        train_l1=tnt.meter.AverageValueMeter(),
        test_l1=tnt.meter.AverageValueMeter(),
    )

    state = {
        'epoch' : 0,
        'train_loss': -1.0,
        'test_loss' : -1.0,
        'best_loss': -1.0,
        'train_MS_SSIM_L1': -1.0,
        'test_MS_SSIM_L1': -1.0,
        'best_MS_SSIM_L1': -1.0,
        'train_mse': -1.0,
        'test_mse': -1.0,
        'best_mse': -1.0,
        'train_l1': -1.0,
        'test_l1': -1.0,
        'best_l1': -1.0,
        'avg_epoch_time': -1,
        'epoch_time': -1,
        'early_stopping_patience': 0,
        'early_percent_improvement': 0,
    }

    avg_time_epochs = []
    time_epoch = time.time()

    hP = args.__dict__
    hP['path_train'] = ''
    hP['path_test'] = ''

    for epoch in range(0, args.epochs):
        state_before = copy.deepcopy(state)
        tensorboard_image_idx = 0
        for key in meters.keys():
            meters[key].reset()

        for data_loader in [data_loader_train, data_loader_test]:
            stage = 'train'
            if data_loader == data_loader_test:
                stage = 'test'

            for batch in tqdm(data_loader):

                y = batch['greyscale_image']
                x = batch['augmented_image']

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

                if data_loader == data_loader_train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()


                MS_SSIM_L1 = MS_SSIM_L1_LOSS().forward(y_prim.detach(), y).cpu()
                mse = F.mse_loss(y_prim.detach(), y).cpu()
                l1 = F.l1_loss(y_prim.detach(), y).cpu()


                loss = loss.cpu()
                y_prim = y_prim.cpu()
                x = x.cpu()
                y = y.cpu()


                meters[f'{stage}_loss'].add(loss.item())
                meters[f'{stage}_MS_SSIM_L1'].add(MS_SSIM_L1.item())
                meters[f'{stage}_mse'].add(mse.item())
                meters[f'{stage}_l1'].add(l1.item())

                if tensorboard_image_idx < 100 and data_loader == data_loader_test:
                    for idx in range(x.shape[0]):
                        if tensorboard_image_idx < 100:
                            data = torch.cat([y[idx,:,:], x[idx,:,:], y_prim[idx,:,:]], 1)
                            tensorboard_writer.add_image(f'sample_{tensorboard_image_idx}', data, dataformats='HW', global_step=epoch)
                            tensorboard_image_idx += 1
                        else:
                            break

        epoch_time = (time.time() - time_epoch) / 60.0
        state['epoch_time'] = epoch_time

        avg_time_epochs.append(epoch_time)
        state['avg_epoch_time'] = np.average(avg_time_epochs)
        eta = ((args.epochs - epoch) * state['avg_epoch_time'])
        time_epoch = time.time()
        state['epoch'] = epoch

        state['train_loss'] = meters['train_loss'].value()[0]
        state['test_loss'] = meters['test_loss'].value()[0]
        state['train_MS_SSIM_L1'] = meters['train_MS_SSIM_L1'].value()[0]
        state['test_MS_SSIM_L1'] = meters['test_MS_SSIM_L1'].value()[0]
        state['train_mse'] = meters['train_mse'].value()[0]
        state['test_mse'] = meters['test_mse'].value()[0]
        state['train_l1'] = meters['train_l1'].value()[0]
        state['test_l1'] = meters['test_l1'].value()[0]

        state['epoch'] = epoch
        if epoch == 0:
            state['best_loss'] = state['test_loss']
        elif state['test_loss'] < state['best_loss']:
            state['best_loss'] = state['test_loss']
            torch.save(model.state_dict(), os.path.join(path_run, 'best_loss.pt'))

        if epoch == 0:
            state['best_MS_SSIM_L1'] = state['test_MS_SSIM_L1']
        elif state['test_MS_SSIM_L1'] < state['best_MS_SSIM_L1']:
            state['best_MS_SSIM_L1'] = state['test_MS_SSIM_L1']

        if epoch == 0:
            state['best_mse'] = state['test_mse']
        elif state['test_mse'] < state['best_mse']:
            state['best_mse'] = state['test_mse']

        if epoch == 0:
            state['best_l1'] = state['test_l1']
        elif state['test_l1'] < state['best_l1']:
            state['best_l1'] = state['test_l1']

        # early stopping
        percent_improvement = 0
        if epoch > 1:
            if state_before[args.early_stopping_param] != 0:
                percent_improvement = -(state[args.early_stopping_param] - state_before[args.early_stopping_param]) / \
                                      state_before[args.early_stopping_param]
            if state[args.early_stopping_param] >= 0:
                if args.early_stopping_delta_percent > percent_improvement:
                    state['early_stopping_patience'] += 1
                else:
                    state['early_stopping_patience'] = 0
            state['early_percent_improvement'] = percent_improvement


        tensorboard_writer.add_hparams(
            hparam_dict=hP,
            metric_dict=state,
            global_step=epoch,
            name=args.run_name
        )

        CsvUtils2.add_hparams(
            path_sequence=path_sequence,
            run_name=args.run_name,
            args_dict=hP,
            metrics_dict=state,
            global_step=epoch
        )

        percent = epoch / args.epochs
        logging.info(
            f'{round(percent * 100, 2)}%, eta: {round(eta, 2)}min, train loss: {round(state["train_loss"], 5)}, '
            f'test loss: {round(state["test_loss"], 5)}')

        torch.save(model.state_dict(), os.path.join(path_run, 'last.pt'))
        # save last model to continue
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(path_run, 'last_checkpoint.pt'))
        tensorboard_writer.flush()

        if state['early_stopping_patience'] >= args.early_stopping_patience or \
                math.isnan(percent_improvement):
            logging_utils.info('early stopping')
            break

    tensorboard_writer.close()
    input('quit?')


if __name__ == '__main__':
    main()
# TODO - Logging at the end of epoch