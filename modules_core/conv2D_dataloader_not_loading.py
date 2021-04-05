from __future__ import print_function, division
import os
import torch
import numpy as np
import json
import random
from modules import torch_utils
from torch.utils.data import Dataset


class SyntheticNoise2DDataset(Dataset):
    """Dataset with original and augmented images."""

    def __init__(self, paths, is_transforms=True, is_debug=False, train = True):

        super().__init__()
        self.is_transforms = is_transforms
        self.dataset_samples = []


        for path in paths:
            for it, video_dir in enumerate(os.scandir(path)):
                if is_debug and it > 2:
                    break
                for idx, img_dir in enumerate(os.scandir(video_dir)):
                    if is_debug and idx > 99:
                        break

                    if train and idx>225:
                        print(len(self.dataset_samples))
                        break
                    elif not train and idx > 73:
                        print(len(self.dataset_samples))
                        break

                    if train :
                        if len(self.dataset_samples) > 250000:
                            break
                    else:
                        if len(self.dataset_samples) > 20000:
                            break

                    if idx < 2:
                        with open(img_dir.path + f'{os.sep}data.json') as json_file:
                            train_json = json.load(json_file)
                            filename = train_json["filename"]
                            shape = train_json["shape"]
                            self.shape0 = shape[0]
                            self.shape1 = shape[1]
                            self.shape2 = shape[2]

                    self.grey_idx = train_json["features"]["grey"]
                    self.augmented_idx = train_json["features"]["augmented"]

                    self.dataset_samples.append(img_dir.path + f'{os.sep}data.bin')

        print(f"{len(self.dataset_samples)} samples succesfully loaded")

    def __len__(self):
        return len(self.dataset_samples)

    def __getitem__(self, idx):

        #print(self.dataset_samples[idx])

        memmap = np.memmap(self.dataset_samples[idx], dtype='float16',
                           mode='r',
                           shape=(self.shape0, self.shape1, self.shape2))

        image = np.array(memmap[:], dtype='float32')

        greyscale_image = image[:, :, self.grey_idx - 1]
        augmented_image = image[:, :, self.augmented_idx - 1]

        greyscale_image = (greyscale_image - 0) / 100
        augmented_image = (augmented_image - 0) / 100

        return {'greyscale_image': torch.from_numpy(greyscale_image),
                'augmented_image': torch.from_numpy(augmented_image)}


def get_data_loaders(args):

    dataset_train = SyntheticNoise2DDataset(paths=args.path_train, is_transforms=True, is_debug=args.is_debug, train=True)
    dataset_test = SyntheticNoise2DDataset(paths=args.path_test, is_transforms=True, is_debug=args.is_debug, train = False)

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