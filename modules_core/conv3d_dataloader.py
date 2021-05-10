from __future__ import print_function, division
import os
import torch
import numpy as np
import json
import random
import time
from modules import torch_utils
from torch.utils.data import Dataset

def translate(img, shift=10, direction='right', roll=True):
    assert direction in ['right', 'left', 'down', 'up'], 'Directions should be top|up|left|right'
    if direction == 'right':
        right_slice = img[:, -shift:].copy()
        img[:, shift:] = img[:, :-shift]
        if roll:
            img[:,:shift] = np.fliplr(right_slice)
    if direction == 'left':
        left_slice = img[:, :shift].copy()
        img[:, :-shift] = img[:, shift:]
        if roll:
            img[:, -shift:] = left_slice
    if direction == 'down':
        down_slice = img[-shift:, :].copy()
        img[shift:, :] = img[:-shift,:]
        if roll:
            img[:shift, :] = down_slice
    if direction == 'up':
        upper_slice = img[:shift, :].copy()
        img[:-shift, :] = img[shift:, :]
        if roll:
            img[-shift:,:] = upper_slice
    return img

class SyntheticNoise3DDataset(Dataset):
    """Dataset with original and augmented images."""

    def __init__(self, paths, conv3d_depth, is_transforms=True, is_debug=False, train = True):

        super().__init__()
        self.is_transforms = is_transforms
        self.dataset_samples = []
        self.sample = []
        self.conv3d_depth = conv3d_depth
        self.videos_dirs = []
        self.train = train

        files = next(os.walk(paths[0]))[1]
        file_count = len(files)

        for it in range(file_count):
            if not train and is_debug:
                path = paths[0] + f"{os.sep}{it+15}"
                self.videos_dirs.append(r'C:\Users\37120\Documents\BachelorThesis\image_data\video_framed_memmap_dataset2\test\13')
                break
            else:
                path = paths[0] + f"{os.sep}{it}"
                self.videos_dirs.append(path)
                if is_debug and it > 6:
                    break

        for idx, item in enumerate(self.videos_dirs):
            imgs = next(os.walk(item))[1]
            file_count = len(imgs)
            imgs_list = []
            for it in range(file_count):
                path = item + f"{os.sep}{it}"
                imgs_list.append(path)
            imgs_list_len = len(imgs_list)-1
            # if train:
            #     for it in range(3):
            #         self.sample = []
            #         for idx in range(5):
            #             self.sample.append(imgs_list[idx])
            #         self.dataset_samples.append(self.sample.copy())
            #
            #     for i in range(1, 120, 1):
            #         self.sample = []
            #         for idx in range(5):
            #             self.sample.append(imgs_list[i+idx])
            #         self.dataset_samples.append(self.sample.copy())
            #
            #     for i in range(3):
            #         self.sample = []
            #         for idx in range(5):
            #             self.sample.append(imgs_list[123 - 5 + idx])
            #         self.dataset_samples.append(self.sample.copy())
            # else:
            #     for it in range(3):
            #         self.sample = []
            #         for idx in range(5):
            #             self.sample.append(imgs_list[idx])
            #         self.dataset_samples.append(self.sample.copy())
            #
            #     for i in range(1, 120, 1):
            #         self.sample = []
            #         for idx in range(5):
            #             self.sample.append(imgs_list[i + idx])
            #         self.dataset_samples.append(self.sample.copy())
            #
            #     for i in range(3):
            #         self.sample = []
            #         for idx in range(5):
            #             self.sample.append(imgs_list[123 - 5 + idx])
            #         self.dataset_samples.append(self.sample.copy())
            if train:
                for it in range(3):
                    self.sample = []
                    for idx in range(5):
                        self.sample.append(imgs_list[idx])
                    self.dataset_samples.append(self.sample.copy())

                for i in range(1, len(imgs_list) - 6, 1):
                    self.sample = []
                    for idx in range(5):
                        self.sample.append(imgs_list[i+idx])
                    self.dataset_samples.append(self.sample.copy())

                for i in range(3):
                    self.sample = []
                    for idx in range(5):
                        self.sample.append(imgs_list[len(imgs_list) - 6 + 3 - 5 + idx])
                    self.dataset_samples.append(self.sample.copy())
            else:
                for it in range(3):
                    self.sample = []
                    for idx in range(5):
                        self.sample.append(imgs_list[idx])
                    self.dataset_samples.append(self.sample.copy())
                for i in range(1, len(imgs_list) - 6, 1):
                    self.sample = []
                    for idx in range(5):
                        self.sample.append(imgs_list[i + idx])
                    self.dataset_samples.append(self.sample.copy())

                for i in range(3):
                    self.sample = []
                    for idx in range(5):
                        self.sample.append(imgs_list[len(imgs_list) - 6 + 3 - 5 + idx])
                    self.dataset_samples.append(self.sample.copy())
        print(len(self.dataset_samples))


    def __len__(self):
        return len(self.dataset_samples)

    def __getitem__(self, idx):
        time_start = time.time()
        img_arr = []

        for item in self.dataset_samples[idx]:
            memmap = np.memmap(item + os.sep + "data.bin", dtype='float16',
                               mode='r',
                               shape=(320, 480, 2))

            image = np.array(memmap[:], dtype='float32')
            img_arr.append(image.copy())

        flip_vertical = random.randint(0, 2)
        flip_horizontal = random.randint(0, 2)

        shift = random.randint(1, 4)

        greyscale_image = img_arr[0][:, :, 0]
        augmented_image = img_arr[0][:, :, 1]

        if flip_vertical == 1:
            greyscale_image = np.fliplr(greyscale_image)
            augmented_image = np.fliplr(augmented_image)
        if flip_horizontal == 1:
            augmented_image = np.flipud(augmented_image)
            greyscale_image = np.flipud(greyscale_image)
        roll = True
        augmented_image = np.expand_dims(augmented_image, axis=0)
        greyscale_image = np.expand_dims(greyscale_image, axis=0)
        greyscale_image = np.expand_dims(greyscale_image, axis=0)
        augmented_image = np.expand_dims(augmented_image, axis=0)
        shift_pixels = random.randint(0, 100)
        for i in range(1 , 5, 1):
            img1 = img_arr[i][:,:,0]
            if flip_vertical == 1:
                img1 = np.fliplr(img1)
            if flip_horizontal == 1:
                img1 = np.flipud(img1)
            img1 = np.expand_dims(img1, axis=0)
            if shift == 1:
                img1 = np.roll(img1, shift_pixels, 1)
            elif shift == 2:
                img1 = np.roll(img1, shift_pixels, 2)
            elif shift == 3:
                img1 = np.roll(img1, -shift_pixels, 1)
            elif shift == 4:
                img1 = np.roll(img1, -shift_pixels, 2)
            img1 = np.expand_dims(img1, axis=0)
            greyscale_image = np.concatenate([greyscale_image, img1.copy()], axis=0)
            img = img_arr[i][:, :, 1]
            if flip_vertical == 1:
                img = np.fliplr(img)
            if flip_horizontal == 1:
                img = np.flipud(img)
            img = np.expand_dims(img, axis=0)
            if shift == 1:
                img = np.roll(img, shift_pixels, 1)
            elif shift == 2:
                img = np.roll(img, shift_pixels, 2)
            elif shift == 3:
                img = np.roll(img, -shift_pixels, 1)
            elif shift == 4:
                img = np.roll(img, -shift_pixels, 2)
            img = np.expand_dims(img, axis=0)
            augmented_image = np.concatenate([augmented_image, img], axis=0)

        # print(greyscale_image.shape)
        # print(augmented_image.shape)
        # import matplotlib.pyplot as plt
        # import matplotlib
        # matplotlib.use('TkAgg')
        # plt.imshow(np.concatenate([greyscale_image[0,0,:,:], augmented_image[0,0,:,:]], axis=1), cmap='gray', vmin=0, vmax=100)
        # plt.show()
        # plt.imshow(np.concatenate([greyscale_image[1,0,:,:], augmented_image[1,0,:,:]], axis=1), cmap='gray', vmin=0, vmax=100)
        # plt.show()

        greyscale_image = (greyscale_image - 0)/100
        augmented_image = (augmented_image - 0) / 100
        greyscale_image = np.swapaxes(greyscale_image, 0, 1)
        augmented_image = np.swapaxes(augmented_image, 0, 1)



        return {'greyscale_image': torch.from_numpy(greyscale_image),
                'augmented_image': torch.from_numpy(augmented_image),
                'dataloader_speed' : (time.time() - time_start)/60, }


def get_data_loaders(args):

    dataset_train = SyntheticNoise3DDataset(paths=args.path_train, is_transforms=True, is_debug=args.is_debug, conv3d_depth=args.conv3d_depth)
    dataset_test = SyntheticNoise3DDataset(paths=args.path_test, is_transforms=True, is_debug=args.is_debug, conv3d_depth=args.conv3d_depth, train = False)

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
                                                   shuffle=True,
                                                   drop_last=True)

    return data_loader_train, data_loader_test