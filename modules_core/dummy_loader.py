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

    def __init__(self, paths, is_transforms=True):

        super().__init__()
        self.is_transforms = is_transforms
        self.dataset_samples = []

        for path in paths:
            for it, img_dir in enumerate(os.scandir(path)):
                with open(img_dir.path + r'\\train.json') as json_file:
                    train_json = json.load(json_file)
                filename = train_json["filename"]
                shape = train_json["shape"]
                self.grey_idx = train_json["features"]["grey"]
                self.augmented_idx = train_json["features"]["augmented"]

                memmap = np.memmap(img_dir.path + f'//{filename}', dtype='float16',
                                             mode='r',
                                             shape=(shape[0], shape[1], shape[2]))
                self.dataset_samples.append(memmap)
        print(f"{len(self.dataset_samples)} samples succesfully loaded")

    def __len__(self):
        return len(self.dataset_samples)

    def __getitem__(self, idx):
        image = np.array(self.dataset_samples[idx][:], dtype='float32')
        greyscale_image = image[:,:,self.grey_idx-1]
        augmented_image = image[:, :, self.augmented_idx-1]

        greyscale_image = (greyscale_image - 0)/100
        augmented_image = (augmented_image - 0) / 100

        return {'greyscale_image': torch.from_numpy(greyscale_image),
                'augmented_image': torch.from_numpy(augmented_image)}


def get_data_loaders(args):

    dataset_train = SyntheticNoiseDataset(paths=args.path_train, is_transforms=True)
    dataset_test = SyntheticNoiseDataset(paths=args.path_test, is_transforms=True)

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