import os
from PIL import Image
import numpy as np
from pathlib import Path
from skimage import color
import matplotlib.pyplot as plt
import json

json_location = r'C:\Users\37120\Documents\BachelorThesis\image_data\flickr30k_images_greyscale\json_data.json'
directory = r'C:\Users\37120\Documents\BachelorThesis\image_data\flickr30k_images\flickr30k_images'
out_dir = r'C:\Users\37120\Documents\BachelorThesis\image_data\flickr30k_images_greyscale'
json_data = {
    'Colorspace': 'CieLab',
    'image_height': 480,
    'image_width': 320,
    'original_images': directory,
    'directory': out_dir,
    'image': []}


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
    greyscale_filename = filename + '.dat'
    filename2 = out_dir + r'\\' + greyscale_filename
    json_data['image'].append({
        'filename': os.path.basename(greyscale_filename),
        'location': filename2,
        'original': directory + r"\\" + os.path.basename(filename) + '.jpg'
    })
    fp = np.memmap(filename2, dtype='float16', mode='w+', shape=shape)
    fp[:] = image[:]
    del fp


def process_file(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((480, 320), Image.ANTIALIAS)
    image_lab = color.rgb2lab(image)
    image_lab[..., 1] = image_lab[..., 2] = 0
    data = np.asarray(image_lab)
    generate_memmap(data.shape, path, data)
    # plt.figure(figsize=(20, 10))
    # plt.subplot(121), plt.imshow(image), plt.axis('off'), plt.title('Original image', size=20)
    # plt.subplot(122), plt.imshow(image_lab), plt.axis('off'), plt.title('Gray scale image', size=20)
    # plt.show()


if __name__ == '__main__':
    for it, img in enumerate(os.scandir(directory)):
        if img.path.endswith(".jpg") and img.is_file():
            process_file(img.path)
        if it%20 == 0:
            print(f"{it} out of 30k images finished. {(it/30000) * 100} % done")
    save_json()
