from __future__ import print_function, division
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from skimage import color
import numpy as np
import json
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class SyntheticNoiseDataset(Dataset):
    """Dataset with original and augmented images."""

    def __init__(self, transform = True, greyscale_directory = r'C:\Users\37120\Documents\BachelorThesis\image_data\flickr30k_images_greyscale',
                 augmented_directory = r'C:\Users\37120\Documents\BachelorThesis\image_data\flickr30k_augmented_test0'):
        """
        Args:
            greyscale_directory: Directory with all the clean images.
            augmented_directory: Directory with all the noisy images.
        """
        # TODO - Nav smuks init. Daudz lieku darabību. Vajag vienkāršot tā, lai saglabājas redundance.
        with open(greyscale_directory + r'\\json_data.json') as json_file:
            self.greyscale_json = json.load(json_file)

        with open(augmented_directory + r'\\json_data.json') as json_file:
            self.augmented_json = json.load(json_file)
        self.transform = transform
        self.greyscale_directory = greyscale_directory
        self.augmented_directory = augmented_directory
        self.colorspace_greyscale = self.greyscale_json["Colorspace"]
        self.colorspace_augmented = self.augmented_json["Colorspace"]
        self.height_greyscale = self.greyscale_json["image_height"]
        self.height_augmented = self.augmented_json["image_height"]
        self.width_greyscale = self.greyscale_json["image_width"]
        self.width_augmented = self.augmented_json["image_width"]
        if self.colorspace_augmented != self.colorspace_greyscale:
            raise IOError("Greyscale and augmented colospace mismatch.")
        if self.height_augmented != self.height_greyscale:
            raise IOError("Greyscale and augmented height mismatch.")
        if self.width_augmented != self.width_greyscale:
            raise IOError("Greyscale and augmented width mismatch.")
        self.colorspace = self.colorspace_greyscale
        self.height = self.height_greyscale
        self.width = self.width_greyscale

    def __len__(self):
        return len(self.augmented_json['image']) #number of images

    def __getitem__(self, idx):
        """
        This dataset loader works with any image shape
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.augmented_json['image'][idx]['filename']
        pre, ext = os.path.splitext(img_name)
        greyscale_memmap = np.memmap(self.greyscale_directory + r'\\' + pre + r'.dat', dtype='float16', mode='r',
                                     shape=(self.height, self.width, 3))
        greyscale_image = np.array(greyscale_memmap[:], dtype='float32')
        augmented_memmap = np.memmap(self.augmented_directory + r'\\' + pre + r'.dat', dtype='float16', mode='r',
                                     shape=(self.height, self.width, 3))
        augmented_image = np.array(augmented_memmap[:], dtype='float32')

        sample = {'greyscale_image': greyscale_image, 'augmented_image': augmented_image}

        if self.transform:
            sample = self.toTensor(sample)

        return sample

    def show_images(self, greyscale_image, augmented_image, batch_size = 4):
        """Show image with landmarks"""

        # TODO - Learn list slicing as i am too stupid for python one liners atm
        if self.transform:
            image1 = np.zeros(shape=(320, 480, 3))
            image2 = np.zeros(shape=(320, 480, 3))
            for i in range(320):
                for j in range(480):
                    image1[i, j, 0] = greyscale_image[0, 0, i,j]
                    image2[i, j, 0] = augmented_image[0, 0, i,j]

        image1 = color.lab2rgb(image1)
        image2 = color.lab2rgb(image2)
        plot_image = np.concatenate((image1, image2), axis=1)
        plt.imshow(plot_image)
        plt.show()

    def toTensor(self, sample):
        greyscale_image, augmented_image = sample['greyscale_image'], sample['augmented_image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image1 = np.zeros(shape = (320,480,1))
        image2 = np.zeros(shape =(320, 480,1))
        for i in range(320):
            for j in range(480):
                image1[i,j,0] = greyscale_image[i,j,0]
                image2[i, j,0] = augmented_image[i, j, 0]
        image1= image1.transpose((2, 0, 1))
        image2 = image2.transpose((2, 0, 1))

        return {'greyscale_image': torch.from_numpy(image1),
                'augmented_image': torch.from_numpy(image2)}