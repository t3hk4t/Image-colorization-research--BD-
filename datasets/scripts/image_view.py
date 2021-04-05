from PIL import Image
from skimage import color
import os
import matplotlib.pyplot as plt
import json
import numpy as np

image_dir = r'C:\Users\37120\Documents\BachelorThesis\image_data\video_framed_memmap\train\0'

path, dirs, files = next(os.walk(image_dir))
file_count = len(dirs)


for it in range(file_count):

    with open(image_dir + f'{os.sep}{it}{os.sep}data.json') as json_file:
        train_json = json.load(json_file)
    filename = train_json["filename"]
    shape = train_json["shape"]

    memmap = np.memmap(image_dir + f'{os.sep}{it}{os.sep}data.bin', dtype='float16',
                       mode='r',
                       shape=(shape[0], shape[1], shape[2]))

    img = np.array(memmap[:], dtype='float32')
    greyscale = img[:,:,0]
    augmented = img[:,:,1]
    img = np.concatenate([greyscale, augmented], axis=1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=100)
    plt.savefig(f'test\\{it}books_read.png')
    print(img.shape)

