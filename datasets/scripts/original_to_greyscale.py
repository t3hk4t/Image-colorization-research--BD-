import os
from PIL import Image
import numpy as np
from pathlib import Path
from skimage import color
import json

json_location_train = r'C:\Users\37120\Documents\BachelorThesis\image_data\flick30k_greyscale_train\train.json'
json_location_test = r'C:\Users\37120\Documents\BachelorThesis\image_data\flick30k_greyscale_test\test.json'

directory = r'C:\Users\37120\Documents\BachelorThesis\image_data\flickr30k_images\flickr30k_images'

out_dir_train = r'C:\Users\37120\Documents\BachelorThesis\image_data\flick30k_greyscale_train'
out_dir_test = r'C:\Users\37120\Documents\BachelorThesis\image_data\flick30k_greyscale_test'

json_data_train = {
    'Colorspace': 'CieLab',
    'image_height': 320,
    'image_width': 480,
    'original_images': directory,
    'directory': out_dir_train,
    'image': []}

json_data_test = {
    'Colorspace': 'CieLab',
    'image_height': 320,
    'image_width': 480,
    'original_images': directory,
    'directory': out_dir_test,
    'image': []}


def file_lengthy(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def save_json_train():
    with open(json_location_train, 'w') as out:
        json.dump(json_data_train, out)

def save_json_test():
    with open(json_location_test, 'w') as out:
        json.dump(json_data_test, out)


def generate_memmap(shape, path, image: np.ndarray, train = True):
    if train:
        filename = Path(path).stem
        greyscale_filename = filename + '.dat'
        filename2 = out_dir_train + r'\\' + greyscale_filename
        json_data_train['image'].append({
            'filename': os.path.basename(greyscale_filename),
            'location': filename2,
            'original': directory + r"\\" + os.path.basename(filename) + '.jpg'
        })
        fp = np.memmap(filename2, dtype='float16', mode='w+', shape=shape)
        fp[:] = image[:]
        del fp
    else:
        filename = Path(path).stem
        greyscale_filename = filename + '.dat'
        filename2 = out_dir_test + r'\\' + greyscale_filename
        json_data_test['image'].append({
            'filename': os.path.basename(greyscale_filename),
            'location': filename2,
            'original': directory + r"\\" + os.path.basename(filename) + '.jpg'
        })
        fp = np.memmap(filename2, dtype='float16', mode='w+', shape=shape)
        fp[:] = image[:]
        del fp


def process_file(path, train):
    image = Image.open(path).convert("RGB")
    image = image.resize((480, 320), Image.ANTIALIAS)
    image_lab = color.rgb2lab(image)
    image_lab[..., 1] = image_lab[..., 2] = 0
    data = np.asarray(image_lab)
    generate_memmap(data.shape, path, data, train = train)


if __name__ == '__main__':
    save_train = False
    for it, img in enumerate(os.scandir(directory)):
        if img.path.endswith(".jpg") and img.is_file():
            if it < 31783 * 0.85:
                process_file(img.path, True)
            else:
                if save_train is False:
                    save_json_train()
                    save_train = True
                process_file(img.path, False)

        if it%100 == 0:
            print(f"{it} out of 30k images finished. {(it/31783) * 100} % done")
    save_json_test()
