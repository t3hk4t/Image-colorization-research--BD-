import os
import torch
import torch.nn
import torch.utils.data
import torch.utils.data.sampler
import torchvision.transforms
from torchvision.transforms import RandomRotation, RandomPerspective, RandomAffine, RandomApply, Resize, ToTensor
import torchvision.transforms.functional as TF
import PIL.Image
import json
import logging
import numpy as np
import traceback, sys
import random
# import cv2
import matplotlib.pyplot as plt


# todo use as template for your loader
class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, paths_source, args, is_transforms=False):
        super().__init__()

        self.args = args
        self.is_transforms = is_transforms

        if isinstance(paths_source, str):
            paths_source = [paths_source]

        self.samples = []
        for idx_source, path_source in enumerate(paths_source):
            logging.info(f'loading path_source {path_source}')

            if not os.path.exists(path_source):
                logging.error(f'data source does not exist: {path_source}')
                exit()

            data_folders = [os.path.join(path_source, it) for it in os.listdir(path_source)]

            for data_folder in data_folders:
                if os.path.isdir(data_folder):
                    try:
                        with open(f'{data_folder}/data.json', 'r') as fp:
                            data_json = json.load(fp)

                        shape_mem = data_json['shape']
                        mem = np.memmap(f'{data_folder}/data.npy', mode='r', dtype=np.uint8,
                                        shape=tuple(shape_mem))

                        is_dent = 'mask' in data_json['features']
                        if self.args.is_filter_damage and is_dent:
                            if data_json['percent_damage'] < self.args.filter_damage_min:
                                continue
                            if data_json['percent_damage'] > self.args.filter_damage_max:
                                continue

                        if self.args.is_filter_panel_percent:
                            if data_json['percent_panel'] < self.args.panel_part_min:
                                continue
                            if data_json['percent_panel'] > self.args.panel_part_max:
                                continue

                        if is_dent: count_damaged += 1
                        else: count_undamaged += 1
                        self.samples.append((mem, is_dent, data_json))

                    except Exception as e:
                        logging.error(str(e))
                        exc_type, exc_value, exc_tb = sys.exc_info()
                        logging.error(traceback.format_exception(exc_type, exc_value, exc_tb))

        if len(self.samples) < self.args.batch_size:
            logging.error(f'not enough frames {len(self.samples)} / {self.args.batch_size}')
            exit()

        logging.info(f'dataset weights: {self.weights}')
        logging.info(f'size_samples: {len(self.samples)}')

    def normalize(self, input_data, min=0, max=1):
        min_val = 0
        max_val = 100
        if min_val != max_val:
            input_data[:] = (input_data - min_val) / (max_val - min_val)   # 0 to 1
            if min != 0 or max != 1:
                input_data[:] = input_data * (max - min) + min
        return input_data

    def __getitem__(self, index):
        data, data_json = self.samples[index]
        image = []
        for feature in self.args.data_features:
            if feature == 'rgb':
                image.append(np.array(data[:,:,:data_json['features'][feature]]))
            else:
                image.append(np.expand_dims(data[:,:,data_json['features'][feature]], axis=-1))

        if is_dent:
            mask = np.array(data[:,:,-1])
        else:
            mask_shape = image[0].shape[:2]
            mask = np.zeros(mask_shape, dtype=np.uint8)

        # do transforms
        if self.is_transforms:
            background_color = []
            for img in image:
                if len(img.shape) > 2 and img.shape[-1] == 3:
                    background_color.append(tuple(np.array(img[0,0], dtype=np.uint8)))
                else:
                    background_color.append(int(np.median(img)))
            # to tensor
            for idx in range(len(image)):
                image[idx] = TF.to_pil_image(image[idx])
            mask = TF.to_pil_image(mask)

            # X flip transformation
            if torch.rand(1) < 0.5:
                for idx in range(len(image)):
                    image[idx] = TF.hflip(image[idx])
                mask = TF.hflip(mask)

            # Affine transformation
            if torch.rand(1) < 0.3:
                real_size = image[0].size[0]
                scale = random.uniform(0.8, 1.2)
                scale_size = int(image[0].size[0] * scale)
                delta = int(abs(scale_size - real_size))
                half_delta = int(delta / 2)
                if half_delta * 2 != delta:
                    scale_size += 1
                    delta = int(abs(scale_size - real_size))
                    half_delta = int(delta / 2)

                for idx in range(len(image)):
                    image[idx] = TF.resize(image[idx], size=scale_size, interpolation=PIL.Image.BILINEAR)
                mask = TF.resize(mask, size=scale_size, interpolation=PIL.Image.NEAREST)
                if scale_size > real_size:
                    for idx in range(len(image)):
                        image[idx] = TF.center_crop(image[idx], output_size=real_size)
                    mask = TF.center_crop(mask, output_size=real_size)
                else:
                    for idx in range(len(image)):
                        image[idx] = TF.pad(image[idx], padding=half_delta, fill=background_color[idx])
                    mask = TF.pad(mask, padding=half_delta)

            # Rotation transformation
            if torch.rand(1) < 0.3:
                angle = random.randint(-15, 15)
                for idx in range(len(image)):
                    image[idx] = TF.rotate(image[idx], angle=angle, resample=PIL.Image.BILINEAR, fill=background_color[idx])
                mask = TF.rotate(mask, angle=angle, resample=PIL.Image.NEAREST)

            # perspective transformation
            if torch.rand(1) < 0.3:
                distortion_scale = random.uniform(0.2, 0.6)
                startpoints, endpoints = RandomPerspective.get_params(image[0].size[0], image[0].size[1], distortion_scale)
                for idx in range(len(image)):
                    image[idx] = TF.perspective(image[idx], startpoints, endpoints, interpolation=PIL.Image.BILINEAR, fill=background_color[idx])
                mask = TF.perspective(mask, startpoints, endpoints, interpolation=PIL.Image.NEAREST)

            # Brightness transform
            if torch.rand(1) < 0.3:
                scale = random.uniform(0.7, 1.3)
                for idx in range(len(image)):
                    image[idx] = TF.adjust_brightness(image[idx], brightness_factor=scale)

            # Contrast transform
            if torch.rand(1) < 0.3:
                scale = random.uniform(0.7, 1.3)
                for idx in range(len(image)):
                    image[idx] = TF.adjust_contrast(image[idx], contrast_factor=scale)

            for idx in range(len(image)):
                image[idx] = TF.to_tensor(image[idx])
            image = torch.vstack(image)
            mask = TF.to_tensor(mask).long().squeeze(0)
        else:
            for idx in range(len(image)):
                image[idx] = TF.to_tensor(image[idx])
            image = torch.vstack(image)
            mask = TF.to_tensor(mask).long().squeeze(0)

        return image, mask

    def __len__(self):
        return len(self.samples)


def get_data_loaders(args):
    logging.info('Using datasource simple')

    dataset_train = Dataset(args.path_train, args, is_transforms=True)
    dataset_test = Dataset(args.path_test, args)

    logging.info('train dataset')
    data_loader_train = torch.utils.data.DataLoader(dataset_train,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.data_workers,
                                                    pin_memory=True,
                                                    shuffle=True)
    logging.info('test dataset')
    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.data_workers,
                                                   pin_memory=True,
                                                   shuffle=False)

    return data_loader_train, data_loader_test