import os
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json

json_location = r'C:\Users\37120\Documents\BachelorThesis\Bachelor thesis\BD izstrade\Datasets\flickr30k_images_memmap\json_data.json'
directory = r'C:\Users\37120\Documents\BachelorThesis\Bachelor thesis\BD izstrade\Datasets\flickr30k_images\flickr30k_images'
out_dir = r'C:\Users\37120\Documents\BachelorThesis\Bachelor thesis\BD izstrade\Datasets\flickr30k_images_memmap'
json_data = {'image': []}


def file_lengthy(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def save_json():
    with open(json_location, 'w') as out:
        json.dump(json_data, out)


def generate_memmap(shape, path, image: np.ndarray):
    filename = Path(path).stem
    filename = filename + '.dat'
    filename = out_dir+r'\\'+filename
    json_data['image'].append({
        'filename': os.path.basename(filename),
        'location': filename,
        'shape': shape
    })
    fp = np.memmap(filename, dtype='float16', mode='w+', shape=shape)
    fp[:] = image[:]
    del fp


def process_file(path):
    image = Image.open(path).convert("RGBA")
    image = image.resize((480, 320), Image.ANTIALIAS)
    data = np.asarray(image)
    data = np.mean(data, axis=2)
    generate_memmap(data.shape, path, data)


if __name__ == '__main__':
    for it, img in enumerate(os.scandir(directory)):
        if img.path.endswith(".jpg") and img.is_file():
            process_file(img.path)
        if it%20 == 0:
            print(f"{it} out of 30k images finished. {(it/30000) * 100} % done")
    save_json()
