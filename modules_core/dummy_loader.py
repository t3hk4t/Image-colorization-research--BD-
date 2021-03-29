from __future__ import print_function, division
import os
import torch
import numpy as np
import json
import random
from modules import torch_utils
from torch.utils.data import Dataset


class SyntheticNoiseDataset(Dataset):
    """Dataset with original and augmented images."""

    def __init__(self, paths, conv3d_depth, is_transforms=True, is_debug=False):

        super().__init__()
        self.is_transforms = is_transforms
        self.dataset_samples = []
        self.conv3d_depth = conv3d_depth
        for path in paths:
            for it, img_dir in enumerate(os.scandir(path)):
                if is_debug and it > 9:
                    break
                with open(img_dir.path + f'{os.sep}data.json') as json_file:
                    train_json = json.load(json_file)
                filename = train_json["filename"]
                shape = train_json["shape"]
                self.grey_idx = train_json["features"]["grey"]
                self.augmented_idx = train_json["features"]["augmented"]

                memmap = np.memmap(img_dir.path + f'{os.sep}data.bin', dtype='float16',
                                             mode='r',
                                             shape=(shape[0], shape[1], shape[2]))
                self.dataset_samples.append(memmap)

        if len(self.dataset_samples) % self.conv3d_depth != 0:
            raise Exception("Invalid conv3d depth")

        print(f"{len(self.dataset_samples)} samples succesfully loaded")

    def __len__(self):
        return int(len(self.dataset_samples)/self.conv3d_depth)

    def __getitem__(self, idx):
        img_arr = []
        if idx == 0:
            for i in range(self.conv3d_depth):
                img_arr.append(np.array(self.dataset_samples[i][:], dtype='float32'))
        else:
            for i in range(self.conv3d_depth * idx, self.conv3d_depth * idx+self.conv3d_depth, 1):
                img_arr.append(np.array(self.dataset_samples[i][:], dtype='float32'))

        greyscale_image = img_arr[0][:, :, self.grey_idx - 1]
        augmented_image = img_arr[0][:, :, self.augmented_idx - 1]

        greyscale_image = np.expand_dims(greyscale_image, axis=0)
        augmented_image = np.expand_dims(augmented_image, axis=0)
        greyscale_image = np.expand_dims(greyscale_image, axis=0)
        augmented_image = np.expand_dims(augmented_image, axis=0)

        for i in range(1 , self.conv3d_depth, 1):
            img = img_arr[i][:,:,self.grey_idx-1]
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=0)
            greyscale_image = np.concatenate([greyscale_image, img], axis=0)
            img = img_arr[i][:, :, self.augmented_idx - 1]
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=0)
            augmented_image = np.concatenate([augmented_image, img], axis=0)

        # print(greyscale_image.shape)
        # print(augmented_image.shape)
        # import matplotlib.pyplot as plt
        # import matplotlib
        # matplotlib.use('TkAgg')
        # plt.imshow(greyscale_image[0,0,:,:], cmap='gray', vmin=0, vmax=100)
        # plt.show()
        # plt.imshow(greyscale_image[1,0,:, :], cmap='gray', vmin=0, vmax=100)
        # plt.show()
        # plt.imshow(augmented_image[0,0,:, :], cmap='gray', vmin=0, vmax=100)
        # plt.show()
        # plt.imshow(augmented_image[1,0,:, :], cmap='gray', vmin=0, vmax=100)
        # plt.show()

        greyscale_image = (greyscale_image - 0)/100
        augmented_image = (augmented_image - 0) / 100

        return {'greyscale_image': torch.from_numpy(greyscale_image),
                'augmented_image': torch.from_numpy(augmented_image)}


def get_data_loaders(args):

    dataset_train = SyntheticNoiseDataset(paths=args.path_train, is_transforms=True, is_debug=args.is_debug, conv3d_depth=args.conv3d_depth)
    dataset_test = SyntheticNoiseDataset(paths=args.path_test, is_transforms=True, is_debug=args.is_debug, conv3d_depth=args.conv3d_depth)

    data_loader_train = torch.utils.data.DataLoader(dataset_train,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.data_workers,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=True)
    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.data_workers,
                                                   pin_memory=True,
                                                   shuffle=False,
                                                   drop_last=True)

    return data_loader_train, data_loader_test