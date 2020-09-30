import os
import unicodedata
import string
import glob
import io
import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
import random
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import tensorboardX
import argparse
from datetime import datetime
import sklearn
import sklearn.model_selection
from enum import Enum

class DataUtils(object):

    # (train, test)
    # args.training_func
    #   sin
    #   sin2d
    #   sin2d_ext - pattern
    # args.training_size
    #   how many samples to include
    # args
    #   main args to override for data specification
    @staticmethod
    def get_data_loaders(args):

        training_func_min = -100
        training_func_max = 100
        training_size_scaler = 1.0
        if args.training_func == 'sin':

            data_training_x = np.linspace(training_func_min * training_size_scaler,
                                          training_func_max * training_size_scaler,
                                          args.training_size * training_size_scaler)
            data_training_y = np.sin(data_training_x)
            data_training_y = np.array([np.array([it]) for it in data_training_y])

            args.tf_inputs = 1
            args.tf_outputs = 1
            args.training_func_valid_steps = 100

        elif args.training_func == 'sin2d':

            x = np.arange(args.training_size, step=0.1, dtype=np.float32)
            y = np.arange(20, step=0.1, dtype=np.float32)

            args.tf_inputs = args.tf_outputs = y.shape[0]
            args.training_func_valid_steps = 300

            data_training_y = np.zeros((x.shape[0], y.shape[0]))
            for i, ival in enumerate(x):
                for j, jval in enumerate(y):
                    # data[j][i] = np.sin(np.sqrt(ival**2 + jval**2))
                    data_training_y[i][j] = np.sin(ival) / 2.0 + np.sin(jval) / 2.0
        elif args.training_func == 'sin2d_ext':
            x = np.arange(args.training_size, step=0.1, dtype=np.float32)
            y = np.arange(12, step=0.1, dtype=np.float32)

            args.tf_inputs = args.tf_outputs = y.shape[0]
            args.training_func_valid_steps = 300

            data_training_y = np.zeros((x.shape[0], y.shape[0]))

            count_horz = 1
            polarity_last_horz = True

            pattern_idx = 0
            pattern = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 3], [6, 2]]

            for i, ival in enumerate(x):
                count_vert = 0
                polarity_last_vert = False
                for j, jval in enumerate(y):
                    sin_i = np.sin(ival)
                    sin_j = np.sin(jval)

                    if (sin_i < 0 and polarity_last_horz) or (sin_i > 0 and not polarity_last_horz):
                        polarity_last_horz = not polarity_last_horz
                        count_horz += 1
                        pattern_idx += 1
                        if pattern_idx >= len(pattern):
                            pattern_idx = 0
                            count_horz = 1

                    if (sin_j < 0 and polarity_last_vert) or (sin_j > 0 and not polarity_last_vert):
                        polarity_last_vert = not polarity_last_vert
                        count_vert += 1

                    pattern_el = pattern[pattern_idx]
                    if pattern_el[0] != count_horz or pattern_el[1] != count_vert:
                        sin_i = 0
                        sin_j = 0

                    data_training_y[i][j] = sin_i / 2.0 + sin_j / 2.0

            # MathPlotLibUtils.showPlot2D(data_training_y)

        data_syntetic_seed_x = []
        data_all = []
        for idx in range(0, data_training_y.shape[0] - args.tf_input_timesteps - args.tf_output_timesteps):

            x_timesteps = np.empty((0, args.tf_inputs,), float)
            for idx_timestep in range(idx, idx + args.tf_input_timesteps):
                x_timesteps = np.vstack([x_timesteps, np.array(data_training_y[idx_timestep])])

            y_timesteps = np.empty((0, args.tf_inputs,), float)
            for idx_timestep in range(idx + args.tf_input_timesteps, idx + args.tf_input_timesteps + args.tf_output_timesteps):
                y_timesteps = np.vstack([y_timesteps, np.array(data_training_y[idx_timestep])])

            #data_timesteps = np.reshape(data_timesteps, (args.tf_input_timesteps, args.tf_inputs))

            data_all.append({
                'x': x_timesteps,
                'y': y_timesteps
            })

            if idx == 0:
                data_syntetic_seed_x = np.copy(data_all[0]['x'])

        data_train, data_test = sklearn.model_selection.train_test_split(data_all, test_size=0.2, shuffle=args.tf_shuffle_batches)

        dataset_train = SimpleDataset(data_train)
        dataset_test = SimpleDataset(data_test)

        data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.tf_batch_size, shuffle=args.tf_shuffle_batches)
        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.tf_batch_size, shuffle=args.tf_shuffle_batches)

        return data_loader_train, data_loader_test, data_syntetic_seed_x

    @staticmethod
    def preprocess_batch(batch):
        input = torch.autograd.Variable(batch['x'].type(torch.FloatTensor))
        output_y = torch.autograd.Variable(batch['y'].type(torch.FloatTensor))
        return input, output_y