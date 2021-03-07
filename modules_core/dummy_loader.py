from __future__ import print_function, division
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import numpy as np
import json
from modules import torch_utils
from torch.utils.data import Dataset


class SyntheticNoiseDataset(Dataset):
    """Dataset with original and augmented images."""

    def __init__(self, train, transform=True,
                 greyscale_directory_train=r'C:\Users\37120\Documents\BachelorThesis\image_data\flick30k_greyscale_train',
                 augmented_directory_train=r'C:\Users\37120\Documents\BachelorThesis\image_data\flick30k_10_augmented_train',
                 greyscale_directory_test=r'C:\Users\37120\Documents\BachelorThesis\image_data\flick30k_greyscale_test',
                 augmented_directory_test=r'C:\Users\37120\Documents\BachelorThesis\image_data\flick30k_10_augmented_test'):

        self.transform = transform
        self.train = train

        if train:

            self.greyscale_directory_train = greyscale_directory_train
            self.augmented_directory_train = augmented_directory_train

            with open(greyscale_directory_train + r'\\train.json') as json_file:
                self.greyscale_train_json = json.load(json_file)
            with open(greyscale_directory_train + r'\\train.json') as json_file:
                self.augmented_train_json = json.load(json_file)

            self.colorspace_greyscale = self.greyscale_train_json["Colorspace"]
            self.colorspace_augmented = self.augmented_train_json["Colorspace"]
            self.height_greyscale = self.greyscale_train_json["image_height"]
            self.height_augmented = self.augmented_train_json["image_height"]
            self.width_greyscale = self.greyscale_train_json["image_width"]
            self.width_augmented = self.augmented_train_json["image_width"]
            if self.colorspace_augmented != self.colorspace_greyscale:
                raise IOError("Greyscale and augmented colospace mismatch.")
            if self.height_augmented != self.height_greyscale:
                raise IOError("Greyscale and augmented height mismatch.")
            if self.width_augmented != self.width_greyscale:
                raise IOError("Greyscale and augmented width mismatch.")

        else:

            self.greyscale_directory_test = greyscale_directory_test
            self.augmented_directory_test = augmented_directory_test

            with open(greyscale_directory_test + r'\\test.json') as json_file:
                self.greyscale_test_json = json.load(json_file)
            with open(greyscale_directory_test + r'\\test.json') as json_file:
                self.augmented_test_json = json.load(json_file)

            self.colorspace_greyscale = self.greyscale_test_json["Colorspace"]
            self.colorspace_augmented = self.augmented_test_json["Colorspace"]
            self.height_greyscale = self.greyscale_test_json["image_height"]
            self.height_augmented = self.augmented_test_json["image_height"]
            self.width_greyscale = self.greyscale_test_json["image_width"]
            self.width_augmented = self.augmented_test_json["image_width"]
            if self.colorspace_augmented != self.colorspace_greyscale:
                raise IOError("Greyscale and augmented colospace mismatch.")
            if self.height_augmented != self.height_greyscale:
                raise IOError("Greyscale and augmented height mismatch.")
            if self.width_augmented != self.width_greyscale:
                raise IOError("Greyscale and augmented width mismatch.")

        self.colorspace = self.colorspace_greyscale
        self.height = self.height_greyscale
        self.width = self.width_greyscale
        self.dataset_samples = []
        self.loadAllImages()

    def __len__(self):
        if self.train:
            return len(self.augmented_train_json['image'])
        else:
            return len(self.augmented_test_json['image'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        augmented_image = np.array(self.dataset_samples[idx]['augmented_image'][:], dtype='float32')
        greyscale_image = np.array(self.dataset_samples[idx]['greyscale_image'][:], dtype='float32')

        normalized_greyscale = (greyscale_image - 0) / (
           100)

        normalized_augmented = (augmented_image - 0) / (
           100)

        sample = {'greyscale_image': normalized_greyscale, 'augmented_image': normalized_augmented}

        if self.transform:
            sample = torch_utils.toTensor(sample)

        return sample

    def loadAllImages(self):

        if self.train:
            total_idxes = len(self.greyscale_train_json['image'])

            for idx in range(total_idxes):
                img_name = self.greyscale_train_json['image'][idx]['filename']
                pre, ext = os.path.splitext(img_name)
                greyscale_memmap = np.memmap(self.greyscale_directory_train + r'\\' + pre + r'.dat', dtype='float16',
                                             mode='r',
                                             shape=(self.height, self.width, 3))
                augmented_memmap = np.memmap(self.augmented_directory_train + r'\\' + pre + r'.dat', dtype='float16',
                                             mode='r',
                                             shape=(self.height, self.width, 3))

                sample = {'greyscale_image': greyscale_memmap, 'augmented_image': augmented_memmap}


                self.dataset_samples.append(sample)

                if idx % 100 == 0:
                    print(f"{idx} out of {total_idxes} train dataset samples are loaded")

            print(f"All {total_idxes} train dataset samples are loaded successfully!")
        else:
            total_idxes = len(self.greyscale_test_json['image'])

            for idx in range(total_idxes):
                img_name = self.greyscale_test_json['image'][idx]['filename']
                pre, ext = os.path.splitext(img_name)
                greyscale_memmap = np.memmap(self.greyscale_directory_test + r'\\' + pre + r'.dat', dtype='float16',
                                             mode='r',
                                             shape=(self.height, self.width, 3))
                augmented_memmap = np.memmap(self.augmented_directory_test + r'\\' + pre + r'.dat', dtype='float16',
                                             mode='r',
                                             shape=(self.height, self.width, 3))

                sample = {'greyscale_image': greyscale_memmap, 'augmented_image': augmented_memmap}

                self.dataset_samples.append(sample)

                if idx % 100 == 0:
                    print(f"{idx} out of {total_idxes} train dataset samples are loaded")

            print(f"All {total_idxes} test dataset samples are loaded successfully!")
