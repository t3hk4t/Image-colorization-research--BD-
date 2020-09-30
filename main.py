import os
import torch.utils.data
import tensorboardX
import numpy as np
import time
import argparse
from datetime import datetime
import math
import copy
import shutil
import json
import logging
from tqdm import tqdm

from modules.file_utils import FileUtils
from modules.radam import RAdam
from modules.tensorboard_utils import TensorBoardUtils
from modules.logging_utils import LoggingUtils
from modules.args_utils import ArgsUtils
from modules.csv_utils import CsvUtils

import torchnet as tnt  # pip install git+https://github.com/pytorch/tnt.git@master


if __name__ == '__main__': # avoid problems with spawned processes

    parser = argparse.ArgumentParser(description='Model trainer')

    parser.add_argument('-tf_cuda', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-tf_is_single_cuda_device', default=True, type=lambda x: (str(x).lower() == 'true'))

    parser.add_argument('-id', default=0, type=int)
    parser.add_argument('-repeat_id', default=0, type=int)
    parser.add_argument('-report', default='report', type=str)
    parser.add_argument('-params_report', nargs='*', required=False)  # extra params for report global for series of runs
    parser.add_argument('-params_report_local', nargs='*', required=False)  # extra params for local run report

    parser.add_argument('-name', help='Run name, by default date', default='', type=str)
    parser.add_argument('-tf_model', default='model', type=str)
    parser.add_argument('-tf_datasource', default='source', type=str)

    parser.add_argument('-tf_batch_size', default=64, type=int)  # now cannot be too big 64
    parser.add_argument('-tf_learning_rate', default=1e-3, type=float)
    parser.add_argument('-tf_epochs', default=10, type=int)

    parser.add_argument('-tf_data_workers', default=0, type=int)
    parser.add_argument('-tf_path_train', default='train', nargs='*')
    parser.add_argument('-tf_path_test', default='test', nargs='*')

    parser.add_argument('-tf_optimizer', default='radam', type=str)  # rmsprop adam sgd

    parser.add_argument('-early_stopping_patience', default=10, type=int)
    parser.add_argument('-early_stopping_param', default='test_loss', type=str)
    parser.add_argument('-early_stopping_delta_percent', default=0.005, type=float)
    parser.add_argument('-early_stopping_finish_value', default=1e-6, type=float)

    parser.add_argument('-tf_loss', default='mse', type=str)  # mse msa hinge

    parser.add_argument('-tensorboard_image_samples_count', default=400, type=int)
    parser.add_argument('-is_save_onnx_model', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-is_quick_test', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-is_windows_test', default=True, type=lambda x: (str(x).lower() == 'true'))

    args, args_other = parser.parse_known_args()

    tmp = ['id', 'name', 'repeat_id', 'epoch', 'train_loss', 'test_loss', 'best_loss', 'avg_epoch_time']
    if not args.params_report is None:
        for it in args.params_report:
            if not it in tmp:
                tmp.append(it)
    args.params_report = tmp

    tmp = ['epoch', 'train_loss', 'test_loss', 'epoch_time', 'early_percent_improvement']
    if not args.params_report_local is None:
        for it in args.params_report_local:
            if not it in tmp:
                tmp.append(it)
    args.params_report_local = tmp

    if len(args.name) == 0:
        args.name = datetime.now().strftime('%y-%m-%d_%H-%M-%S')

    FileUtils.createDir('./tasks/' + args.report)
    run_path = './tasks/' + args.report + '/runs/' + args.name
    if os.path.exists(run_path):
        shutil.rmtree(run_path, ignore_errors=True)
        time.sleep(3)
        while os.path.exists(run_path):
            pass

    tensorboard_writer = tensorboardX.SummaryWriter(logdir=run_path)
    tensorboard_utils = TensorBoardUtils(tensorboard_writer)
    logging_utils = LoggingUtils(filename=os.path.join(run_path, 'log.txt'))

    get_data_loaders = getattr(__import__('modules_core.' + args.tf_datasource, fromlist=['get_data_loaders']),
                               'get_data_loaders')
    data_loader_train, data_loader_test = get_data_loaders(args)

    ArgsUtils.log_args(args, 'main.py', logging_utils)

    with open(os.path.join(run_path + f'/{args.name}.json'), 'w') as outfile:
        json.dump(args.__dict__, outfile, indent=4)

    if not torch.cuda.is_available():
        args.device = 'cpu'
        logging.info('CUDA NOT AVAILABLE')
    else:
        args.device = 'cuda'
        logging.info('cuda devices: {}'.format(torch.cuda.device_count()))

    Model = getattr(__import__('modules_core.' + args.tf_model, fromlist=['Model']), 'Model')
    model = Model(args)

    is_data_parallel = False
    if not args.tf_is_single_cuda_device:
        if args.device == 'cuda' and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, dim=0)
            is_data_parallel = True
            logging.info(f'PARALLEL MODEL {torch.cuda.device_count()}')

    model = model.to(args.device)

    if args.is_save_onnx_model:
        dummy_input = torch.randn(data_loader_train.dataset.__getitem__(0)[0].shape, device=args.device)
        torch.onnx.export(model, dummy_input.unsqueeze(dim=0), os.path.join(run_path, f'{args.name}.onnx'))

    optimizer_func = None
    if args.tf_optimizer == 'adam':
        optimizer_func = torch.optim.Adam(
            model.parameters(),
            lr=args.tf_learning_rate
        )
    elif args.tf_optimizer == 'rmsprop':
        optimizer_func = torch.optim.RMSprop(
            model.parameters(),
            lr=args.tf_learning_rate
        )
    elif args.tf_optimizer == 'sgd':
        optimizer_func = torch.optim.SGD(
            model.parameters(),
            lr=args.tf_learning_rate,
            momentum=0.9
        )
    elif args.tf_optimizer == 'radam':
        optimizer_func = RAdam(
            model.parameters(),
            lr=args.tf_learning_rate
        )

    loss_func = torch.nn.MSELoss()
    if args.tf_loss == 'mse':
        loss_func = torch.nn.MSELoss()
    elif args.tf_loss == 'msa':
        loss_func = torch.nn.L1Loss()
    elif args.tf_loss == 'hinge':
        loss_func = torch.nn.SmoothL1Loss()  # hinge loss


    def forward(batch):
        x = batch[0].to(args.device)
        y = batch[1].to(args.device)

        model_module = model
        if is_data_parallel:
            model_module = model.module

        if hasattr(model_module, 'init_hidden'):
            hidden = model_module.init_hidden(batch_size=x.size(0))  # ! Important // cannot be used with data parallel
            output = model.forward(x, hidden)
        else:
            output = model.forward(x)

        loss = loss_func(output, y)

        if args.device != 'cpu':
            if type(output) == list:
                output = output[-1].to('cpu')
            else:
                output = output.to('cpu')
            x = x.to('cpu')
            y = y.to('cpu')
            loss = loss.to('cpu')


        result = dict(
            output=output,
            y=y,
            loss=loss
        )
        return result


    state = {
        'epoch': 0,
        'best_loss': -1,
        'avg_epoch_time': -1,
        'epoch_time': -1,
        'early_stopping_patience': 0,
        'early_percent_improvement': 0,
        'train_loss': -1,
        'test_loss': -1
    }
    avg_time_epochs = []
    time_epoch = time.time()

    if not args.is_windows_test:
        CsvUtils.create_local(args)

    meters = dict(
        train_loss=tnt.meter.AverageValueMeter(),
        test_loss=tnt.meter.AverageValueMeter(),
    )

    start_epoch = 1
    for epoch in range(start_epoch, args.tf_epochs + 1):
        state_before = copy.deepcopy(state)
        logging.info('epoch: {} / {}'.format(epoch, args.tf_epochs))

        for key in meters.keys():
            meters[key].reset()

        for data_loader in [data_loader_train, data_loader_test]:
            idx_quick_test = 0
            idx_tensorboard_image_samples = 0

            output_y = []

            meter_prefix = 'train'
            if data_loader == data_loader_train:
                model = model.train()
            else:
                meter_prefix = 'test'
                model = model.eval()

            for batch in tqdm(data_loader):
                if data_loader == data_loader_train:
                    optimizer_func.zero_grad()
                    model.zero_grad()

                    result = forward(batch)

                    result['loss'].backward()
                    optimizer_func.step()
                else:
                    with torch.no_grad():
                        result = forward(batch)

                        if idx_tensorboard_image_samples < args.tensorboard_image_samples_count:
                            for idx in range(batch[0].size(0)):
                                if idx_tensorboard_image_samples < args.tensorboard_image_samples_count:
                                    tensorboard_utils.addPlot2D(dataXY=batch[0][idx].to('cpu').data.numpy(),
                                                                tag=f'{idx_tensorboard_image_samples}_x', global_step=epoch)
                                    tensorboard_utils.addPlot2D(dataXY=batch[1][idx].to('cpu').data.numpy(),
                                                                tag=f'{idx_tensorboard_image_samples}_y', global_step=epoch)
                                    tensorboard_utils.addPlot2D(dataXY=result['output'][idx].to('cpu').data.numpy(),
                                                                tag=f'{idx_tensorboard_image_samples}_out',
                                                                global_step=epoch)
                                    idx_tensorboard_image_samples += 1
                                else:
                                    break

                meters[f'{meter_prefix}_loss'].add(np.average(result['loss'].to('cpu').data))
                torch.cuda.empty_cache()

                idx_quick_test += 1
                if args.is_quick_test and idx_quick_test >= 2:
                    break

            state[f'{meter_prefix}_loss'] = meters[f'{meter_prefix}_loss'].value()[0]
            tensorboard_writer.add_scalar(tag=f'{meter_prefix}_loss', scalar_value=state[f'{meter_prefix}_loss'],
                                          global_step=epoch)

        model_module = model
        if is_data_parallel:
            model_module = model.module

        if epoch == 1:
            state['best_loss'] = state['test_loss']
        elif state['best_loss'] > state['test_loss']:
            state['best_loss'] = state['test_loss']
            torch.save(model_module.state_dict(), os.path.join(run_path, 'best.pt'))

        epoch_time = (time.time() - time_epoch) / 60.0
        percent = epoch / args.tf_epochs
        state['epoch_time'] = epoch_time

        avg_time_epochs.append(epoch_time)
        state['avg_epoch_time'] = np.average(avg_time_epochs)
        eta = ((args.tf_epochs - epoch) * state['avg_epoch_time'])
        time_epoch = time.time()
        state['epoch'] = epoch

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

        tensorboard_writer.add_scalar(tag='improvement', scalar_value=state['early_percent_improvement'], global_step=epoch)
        # save checkpoint to continue training
        model_module = model
        if is_data_parallel:
            model_module = model.module
        # save last and best models
        torch.save(model_module.state_dict(), os.path.join(run_path, f'{args.name}.pt'))

        logging.info(
            f'{round(percent * 100, 2)}% each: {round(state["avg_epoch_time"], 2)}min eta: {round(eta, 2)} min loss: {round(state["train_loss"], 5)} improve: {round(percent_improvement, 3)}')

        if not args.is_windows_test:
            CsvUtils.add_results_local(args, state)
            CsvUtils.add_results(args, state)

        if state['early_stopping_patience'] >= args.early_stopping_patience or \
                math.isnan(percent_improvement):
            logging_utils.info('early stopping')
            break

    tensorboard_writer.close()
