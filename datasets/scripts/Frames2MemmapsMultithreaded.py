import os
import numpy as np
from pathlib import Path
from skimage import color
from PIL import Image, ImageEnhance, ImageFilter
import json
import sys
import time
from multiprocessing import Process
import multiprocessing
import random
from blend_modes import multiply
from blend_modes import darken_only
import concurrent.futures
it = 0
location_test = r'/mnt/beegfs2/home/leo01/image_data/video_framed_memmap_dataset/test/'
location_train = r'/mnt/beegfs2/home/leo01/image_data/video_framed_memmap_dataset/train/'
original_img_dir_test = r'/mnt/beegfs2/home/leo01/image_data/video_framed_dataset/test/'
original_img_dir_train = r'/mnt/beegfs2/home/leo01/image_data/video_framed_dataset/train/'
noise_dir = r'/mnt/beegfs2/home/leo01/noise_data/'


def invert(image):
    return image.point(lambda p: 255 - p)

def file_lengthy(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def getRandomFile(path):
    files = os.listdir(path)
    index = random.randrange(0, len(files))
    return files[index]

def save_json(location, json_data):
    with open(location, 'w') as out:
        json.dump(json_data, out)

def sp_noise(image,prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                if random.randint(0, 100) <= 50:
                    output[i][j] = 255
                else:
                    output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def generate_memmap(greyscale_image: np.ndarray, augmented_image: np.ndarray, path,folder,image_name, train = True):
    global it
    if train:
        out_data = np.concatenate((greyscale_image, augmented_image), axis=2)

        image_folder = path+f'{os.sep}{folder}' #if not exists image folder, then create
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        memmap_folder = image_folder + f'{os.sep}{image_name}' #if not exists image folder, then create
        if not os.path.exists(memmap_folder):
            os.makedirs(memmap_folder)

        filename = image_name + ".jpg"
        memmap_location = memmap_folder + f'{os.sep}' + r'data.bin'
        json_data = {
            'Colorspace': 'CieLab',
            'filename': filename,
            'shape': [320, 480, 2],
            'original_images': original_img_dir_train,
            'features': {'grey': 1, 'augmented': 2}}

        fp = np.memmap(memmap_location, dtype='float16', mode='w+', shape=out_data.shape)
        fp[:] = out_data[:]
        del fp
        save_json(memmap_folder + f'{os.sep}'+"data.json",json_data)
    else:
        out_data = np.concatenate((greyscale_image, augmented_image), axis=2)

        image_folder = path + f'{os.sep}{folder}'  # if not exists image folder, then create
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        memmap_folder = image_folder + f'{os.sep}{image_name}'  # if not exists image folder, then create
        if not os.path.exists(memmap_folder):
            os.makedirs(memmap_folder)

        filename = image_name + ".jpg"

        memmap_location = memmap_folder + f'{os.sep}' + r'data.bin'

        json_data = {
            'Colorspace': 'CieLab',
            'filename': filename,
            'shape': [320, 480, 2],
            'original_images': original_img_dir_test,
            'features': {'grey': 1, 'augmented': 2}}

        fp = np.memmap(memmap_location, dtype='float16', mode='w+', shape=out_data.shape)
        fp[:] = out_data[:]
        del fp
        save_json(memmap_folder + f'{os.sep}' + "data.json", json_data)


def validate(it):
    pass


def basic_deterioration(deterioration_overlay_img):
    if random.randint(0, 100) <= 50:
        deterioration_overlay_img = deterioration_overlay_img.transpose(Image.FLIP_LEFT_RIGHT) #Flip horizontal
    if random.randint(0, 100) <= 50:
        deterioration_overlay_img = deterioration_overlay_img.transpose(Image.FLIP_TOP_BOTTOM) #Flip vertical
    deg = random.uniform(-5.0, 5.0)

    enhancer = ImageEnhance.Sharpness(deterioration_overlay_img)
    deterioration_overlay_img = enhancer.enhance(random.uniform(0.5, 1.5))
    enhancer = ImageEnhance.Color(deterioration_overlay_img)
    deterioration_overlay_img = enhancer.enhance(random.uniform(0.8, 1.2))
    enhancer = ImageEnhance.Contrast(deterioration_overlay_img)
    deterioration_overlay_img = enhancer.enhance(random.uniform(0.95, 1.05))
    enhancer = ImageEnhance.Brightness(deterioration_overlay_img)
    deterioration_overlay_img = enhancer.enhance(random.uniform(0.95, 1.05))

    return deterioration_overlay_img.rotate(deg)

def scale_img(deterioration_overlay_img):
    scale = random.uniform(1.2, 1.5) * 2
    w, h = deterioration_overlay_img.size
    deterioration_overlay_img = deterioration_overlay_img.crop((w/2 - w / scale, h/2 - h / scale,
                    w/2 + w / scale, h/2 + h / scale))
    return deterioration_overlay_img.resize((w, h), Image.LANCZOS)

def blend_imgs(deterioration_overlay_img, original_image, opacity):
    background_img = np.array(original_image)  # Inputs to blend_modes need to be numpy arrays.
    background_img_float = background_img.astype(float)  # Inputs to blend_modes need to be floats.

    foreground_img = np.array(deterioration_overlay_img)  # Inputs to blend_modes need to be numpy arrays.
    foreground_img_float = foreground_img.astype(float)  # Inputs to blend_modes need to be floats.

    image_add_type = random.randint(1, 2)
    if image_add_type == 1:
        new_img = darken_only(background_img_float, foreground_img_float, opacity)
    elif image_add_type == 2:
        new_img = multiply(background_img_float, foreground_img_float, opacity)

        # Convert blended image back into PIL image
    blended_img = np.uint8(new_img)  # Image needs to be converted back to uint8 type for PIL handling.
    return Image.fromarray(blended_img)


def heavy_damage(original_image):
    dir_new = r'/mnt/beegfs2/home/leo01/noise_data/'
    oval_noise = random.randint(0, 2)
    if random.uniform(0.0, 1.0) < 0.2:  # 001 noise data dir
        for i in range(oval_noise):
            deterioration_overlay_img = Image.open(dir_new + f"001{os.sep}" + getRandomFile(dir_new + f"001{os.sep}")).convert("RGBA")
            deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
            deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
            deterioration_overlay_img = scale_img(deterioration_overlay_img)
            original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.85, 1))

    scratchy_noise = random.randint(1, 5)
    if random.uniform(0.0, 1.0) < 0.8:  # 002 noise data dir
        for i in range(scratchy_noise):
            deterioration_overlay_img = Image.open(dir_new + f"002{os.sep}" + getRandomFile(dir_new + f"002{os.sep}")).convert("RGBA")
            deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
            deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
            deterioration_overlay_img = scale_img(deterioration_overlay_img)
            original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.65, 1))

    minor_noise_1 = random.randint(1, 2)
    if random.uniform(0.0, 1.0) < 0.3:  # 003 noise data dir
        for i in range(minor_noise_1):
            deterioration_overlay_img = Image.open(dir_new + f"003{os.sep}" + getRandomFile(dir_new + f"003{os.sep}")).convert("RGBA")
            deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
            deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
            deterioration_overlay_img = scale_img(deterioration_overlay_img)
            original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.75, 1))

    minor_noise_2 = random.randint(1, 3)
    if random.uniform(0.0, 1.0) < 0.3:  # 004 noise data dir
        for i in range(minor_noise_2):
            deterioration_overlay_img = Image.open(dir_new + f"004{os.sep}" + getRandomFile(dir_new + f"004{os.sep}")).convert("RGBA")
            deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
            deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
            deterioration_overlay_img = scale_img(deterioration_overlay_img)
            original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.75, 1))

    minor_noise_3 = random.randint(1, 2)
    if random.uniform(0.0, 1.0) < 0.2:  # 005 noise data dir
        for i in range(minor_noise_3):
            deterioration_overlay_img = Image.open(dir_new + f"005{os.sep}" + getRandomFile(dir_new + f"005{os.sep}")).convert("RGBA")
            deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
            deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
            deterioration_overlay_img = scale_img(deterioration_overlay_img)
            original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.75, 1))

    major_noise_1 = random.randint(1, 2)
    if random.uniform(0.0, 1.0) < 0.1:  # 005 noise data dir
        for i in range(major_noise_1):
            deterioration_overlay_img = Image.open(dir_new + f"006{os.sep}" + getRandomFile(dir_new + f"006{os.sep}")).convert("RGBA")
            deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
            deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
            deterioration_overlay_img = scale_img(deterioration_overlay_img)
            original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.85, 1))

    scratchy_noise_2 = random.randint(1, 5)
    if random.uniform(0.0, 1.0) < 0.8:  # 006 noise data dir
        for i in range(scratchy_noise_2):
            deterioration_overlay_img = Image.open(dir_new + f"007{os.sep}" + getRandomFile(dir_new + f"007{os.sep}")).convert("RGBA")
            deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
            deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
            deterioration_overlay_img = scale_img(deterioration_overlay_img)
            original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.85, 1))

    if random.uniform(0.0, 1.0) < 0.5:  # 007 grain effect
        deterioration_overlay_img = Image.open(dir_new + f"007{os.sep}" + getRandomFile(dir_new + f"007{os.sep}")).convert("RGBA")
        deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
        deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
        deterioration_overlay_img = scale_img(deterioration_overlay_img)
        original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.85, 1))

    scratchy_noise_3 = random.randint(1, 5)
    if random.uniform(0.0, 1.0) < 0.8:  # 008 noise data dir
        for i in range(scratchy_noise_3):
            deterioration_overlay_img = Image.open(dir_new + f"009{os.sep}" + getRandomFile(dir_new + f"009{os.sep}")).convert("RGBA")
            deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
            deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
            deterioration_overlay_img = scale_img(deterioration_overlay_img)
            original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.85, 1))

    major_noise_2 = random.randint(1, 2)
    if random.uniform(0.0, 1.0) < 0.05:  # 009 noise data dir
        for i in range(major_noise_2):
            deterioration_overlay_img = Image.open(dir_new + f"010{os.sep}" + getRandomFile(dir_new + f"010{os.sep}")).convert("RGBA")
            deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
            deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
            deterioration_overlay_img = scale_img(deterioration_overlay_img)
            original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.85, 1))

    major_noise_3 = random.randint(1, 2)
    if random.uniform(0.0, 1.0) < 0.05:  # 010 noise data dir
        for i in range(major_noise_3):
            deterioration_overlay_img = Image.open(dir_new + f"011{os.sep}" + getRandomFile(dir_new + f"011{os.sep}")).convert("RGBA")
            deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
            deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
            deterioration_overlay_img = scale_img(deterioration_overlay_img)
            original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.85, 1))

    scratchy_noise_4 = random.randint(1, 5)
    if random.uniform(0.0, 1.0) < 0.8:  # 011 noise data dir
        for i in range(scratchy_noise_4):
            deterioration_overlay_img = Image.open(dir_new + f"012{os.sep}" + getRandomFile(dir_new + f"012{os.sep}")).convert("RGBA")
            deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
            deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
            deterioration_overlay_img = scale_img(deterioration_overlay_img)
            original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.85, 1))

    return original_image

def medium_damage(original_image):
    dir_new = r'/mnt/beegfs2/home/leo01/noise_data/'
    oval_noise = random.randint(0, 2)
    if random.uniform(0.0, 1.0) < 0.1:  # 001 noise data dir
        for i in range(oval_noise):
            deterioration_overlay_img = Image.open(dir_new + f"001{os.sep}" + getRandomFile(dir_new + f"001{os.sep}")).convert("RGBA")
            deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
            deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
            deterioration_overlay_img = scale_img(deterioration_overlay_img)
            original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.65, 0.85))

    scratchy_noise = random.randint(1, 5)
    if random.uniform(0.0, 1.0) < 0.6:  # 002 noise data dir
        for i in range(scratchy_noise):
            deterioration_overlay_img = Image.open(dir_new + f"002{os.sep}" + getRandomFile(dir_new + f"002{os.sep}")).convert("RGBA")
            deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
            deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
            deterioration_overlay_img = scale_img(deterioration_overlay_img)
            original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.45, 0.8))

    minor_noise_1 = random.randint(1, 2)
    if random.uniform(0.0, 1.0) < 0.15:  # 003 noise data dir
        for i in range(minor_noise_1):
            deterioration_overlay_img = Image.open(dir_new + f"003{os.sep}" + getRandomFile(dir_new + f"003{os.sep}")).convert("RGBA")
            deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
            deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
            deterioration_overlay_img = scale_img(deterioration_overlay_img)
            original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.65, 0.85))

    minor_noise_2 = random.randint(1, 3)
    if random.uniform(0.0, 1.0) < 0.15:  # 004 noise data dir
        for i in range(minor_noise_2):
            deterioration_overlay_img = Image.open(dir_new + f"004{os.sep}" + getRandomFile(dir_new + f"004{os.sep}")).convert("RGBA")
            deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
            deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
            deterioration_overlay_img = scale_img(deterioration_overlay_img)
            original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.65, 0.85))

    minor_noise_3 = random.randint(1, 2)
    if random.uniform(0.0, 1.0) < 0.1:  # 005 noise data dir
        for i in range(minor_noise_3):
            deterioration_overlay_img = Image.open(dir_new + f"005{os.sep}" + getRandomFile(dir_new + f"005{os.sep}")).convert("RGBA")
            deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
            deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
            deterioration_overlay_img = scale_img(deterioration_overlay_img)
            original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.65, 0.85))

    major_noise_1 = random.randint(1, 2)
    if random.uniform(0.0, 1.0) < 0.02:  # 005 noise data dir
        for i in range(major_noise_1):
            deterioration_overlay_img = Image.open(dir_new + f"006{os.sep}" + getRandomFile(dir_new + f"006{os.sep}")).convert("RGBA")
            deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
            deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
            deterioration_overlay_img = scale_img(deterioration_overlay_img)
            original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.50, 0.7))

    scratchy_noise_2 = random.randint(1, 5)
    if random.uniform(0.0, 1.0) < 0.6:  # 006 noise data dir
        for i in range(scratchy_noise_2):
            deterioration_overlay_img = Image.open(dir_new + f"007{os.sep}" + getRandomFile(dir_new + f"007{os.sep}")).convert("RGBA")
            deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
            deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
            deterioration_overlay_img = scale_img(deterioration_overlay_img)
            original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.45, 0.8))

    if random.uniform(0.0, 1.0) < 0.25:  # 007 grain effect
        deterioration_overlay_img = Image.open(dir_new + f"007{os.sep}" + getRandomFile(dir_new + f"007{os.sep}")).convert("RGBA")
        deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
        deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
        deterioration_overlay_img = scale_img(deterioration_overlay_img)
        original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.7, 0.9))

    scratchy_noise_3 = random.randint(1, 5)
    if random.uniform(0.0, 1.0) < 0.6:  # 008 noise data dir
        for i in range(scratchy_noise_3):
            deterioration_overlay_img = Image.open(dir_new + f"009{os.sep}" + getRandomFile(dir_new + f"009{os.sep}")).convert("RGBA")
            deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
            deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
            deterioration_overlay_img = scale_img(deterioration_overlay_img)
            original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.45, 0.8))

    major_noise_2 = random.randint(1, 2)
    if random.uniform(0.0, 1.0) < 0.02:  # 009 noise data dir
        for i in range(major_noise_2):
            deterioration_overlay_img = Image.open(dir_new + f"010{os.sep}" + getRandomFile(dir_new + f"010{os.sep}")).convert("RGBA")
            deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
            deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
            deterioration_overlay_img = scale_img(deterioration_overlay_img)
            original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.3, 0.7))

    major_noise_3 = random.randint(1, 2)
    if random.uniform(0.0, 1.0) < 0.02:  # 010 noise data dir
        for i in range(major_noise_3):
            deterioration_overlay_img = Image.open(dir_new + f"011{os.sep}" + getRandomFile(dir_new + f"011{os.sep}")).convert("RGBA")
            deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
            deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
            deterioration_overlay_img = scale_img(deterioration_overlay_img)
            original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.3, 0.7))

    scratchy_noise_4 = random.randint(1, 5)
    if random.uniform(0.0, 1.0) < 0.6:  # 011 noise data dir
        for i in range(scratchy_noise_4):
            deterioration_overlay_img = Image.open(dir_new + f"012{os.sep}" + getRandomFile(dir_new + f"012{os.sep}")).convert("RGBA")
            deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
            deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
            deterioration_overlay_img = scale_img(deterioration_overlay_img)
            original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.45, 0.8))

    return original_image

def low_damage(original_image):
    dir_new = r'/mnt/beegfs2/home/leo01/noise_data/'
    oval_noise = random.randint(0, 2)
    if random.uniform(0.0, 1.0) < 0.01:  # 001 noise data dir
        for i in range(oval_noise):
            deterioration_overlay_img = Image.open(dir_new + f"001{os.sep}" + getRandomFile(dir_new + f"001{os.sep}")).convert("RGBA")
            deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
            deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
            deterioration_overlay_img = scale_img(deterioration_overlay_img)
            original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.45, 0.55))

    scratchy_noise = random.randint(1, 5)
    if random.uniform(0.0, 1.0) < 0.5:  # 002 noise data dir
        for i in range(scratchy_noise):
            deterioration_overlay_img = Image.open(dir_new + f"002{os.sep}" + getRandomFile(dir_new + f"002{os.sep}")).convert("RGBA")
            deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
            deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
            deterioration_overlay_img = scale_img(deterioration_overlay_img)
            original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.15, 0.35))

    if random.uniform(0.0, 1.0) < 0.05:  # 003 noise data dir
        deterioration_overlay_img = Image.open(dir_new + f"003{os.sep}" + getRandomFile(dir_new + f"003{os.sep}")).convert("RGBA")
        deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
        deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
        deterioration_overlay_img = scale_img(deterioration_overlay_img)
        original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.15, 0.25))

    if random.uniform(0.0, 1.0) < 0.01:  # 004 noise data dir
        deterioration_overlay_img = Image.open(dir_new + f"004{os.sep}" + getRandomFile(dir_new + f"004{os.sep}")).convert("RGBA")
        deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
        deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
        deterioration_overlay_img = scale_img(deterioration_overlay_img)
        original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.15, 0.20))

    if random.uniform(0.0, 1.0) < 0.01:  # 005 noise data dir
        deterioration_overlay_img = Image.open(dir_new + f"005{os.sep}" + getRandomFile(dir_new + f"005{os.sep}")).convert("RGBA")
        deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
        deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
        deterioration_overlay_img = scale_img(deterioration_overlay_img)
        original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.15, 0.20))

    scratchy_noise_2 = random.randint(1, 5)
    if random.uniform(0.0, 1.0) < 0.5:  # 006 noise data dir
        for i in range(scratchy_noise_2):
            deterioration_overlay_img = Image.open(dir_new + f"007{os.sep}" + getRandomFile(dir_new + f"007{os.sep}")).convert("RGBA")
            deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
            deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
            deterioration_overlay_img = scale_img(deterioration_overlay_img)
            original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.15, 0.35))

    if random.uniform(0.0, 1.0) < 0.45:  # 007 grain effect
        deterioration_overlay_img = Image.open(dir_new + f"007{os.sep}" + getRandomFile(dir_new + f"007{os.sep}")).convert("RGBA")
        deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
        deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
        deterioration_overlay_img = scale_img(deterioration_overlay_img)
        original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.2, 0.7))

    scratchy_noise_3 = random.randint(1, 5)
    if random.uniform(0.0, 1.0) < 0.5:  # 008 noise data dir
        for i in range(scratchy_noise_3):
            deterioration_overlay_img = Image.open(dir_new + f"009{os.sep}" + getRandomFile(dir_new + f"009{os.sep}")).convert("RGBA")
            deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
            deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
            deterioration_overlay_img = scale_img(deterioration_overlay_img)
            original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.15, 0.35))

    scratchy_noise_4 = random.randint(1, 5)
    if random.uniform(0.0, 1.0) < 0.6:  # 011 noise data dir
        for i in range(scratchy_noise_4):
            deterioration_overlay_img = Image.open(dir_new + f"012{os.sep}" + getRandomFile(dir_new + f"012{os.sep}")).convert("RGBA")
            deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
            deterioration_overlay_img = basic_deterioration(deterioration_overlay_img)
            deterioration_overlay_img = scale_img(deterioration_overlay_img)
            original_image = blend_imgs(deterioration_overlay_img, original_image, random.uniform(0.15, 0.3))

    return original_image


def process_file(paths, train, augment_type, folder, name_list):
    greyscale_image_list = []
    original_image_list = []

    for path in paths:
        greyscale_image = Image.open(path).convert("RGB")
        greyscale_image = greyscale_image.resize((480, 320), Image.ANTIALIAS)
        image_lab = color.rgb2lab(greyscale_image)
        image_lab[..., 1] = image_lab[..., 2] = 0
        greyscale_image = np.asarray(image_lab)
        greyscale_image = np.delete(greyscale_image, 1,2)
        greyscale_image = np.delete(greyscale_image, 1,2)
        greyscale_image_list.append(np.copy(greyscale_image))
        # FINAL GREYSCALE IMAGE THAT IS (320, 480)


        original_image = (Image.open(path)).convert("RGBA")
        original_image = original_image.resize((480, 320), Image.ANTIALIAS)
        if augment_type == 'high':
            original_image = heavy_damage(original_image)
        elif augment_type == 'medium':
            original_image = medium_damage(original_image)
        else:
            original_image = low_damage(original_image)

        original_image_list.append(original_image.copy())

    if random.randint(0, 100) <= 10:  # Gaussian noise
        alpha = random.uniform(0, 0.02)
        for item in original_image_list:
            ata = np.asarray(item)
            salt_pepper = sp_noise(ata, alpha)
            im2 = Image.fromarray(salt_pepper)
            item.paste(im2, (0, 0), im2)
    if random.randint(0, 100) <= 30:  # Blur
        for item in original_image_list:
            item = item.filter(ImageFilter.BoxBlur(1))

    for it, img in enumerate(original_image_list):
        rgb_image = img.convert('RGB')
        image_lab = color.rgb2lab(rgb_image)
        image_lab[..., 1] = image_lab[..., 2] = 0
        augmented_image = np.asarray(image_lab)
        augmented_image = np.delete(augmented_image, 1, 2)
        augmented_image = np.delete(augmented_image, 1, 2)
        greyscale_image = greyscale_image_list[it]
        file_name = f"{name_list[it]}"

        if train:
            generate_memmap(greyscale_image, augmented_image, location_train,folder,file_name, train=train)
        else:
            generate_memmap(greyscale_image, augmented_image, location_test,folder, file_name, train=train)


def process_folder(path, train, augment_type, folder):
    files = next(os.walk(path+f"{os.sep}"))[2]
    file_count = len(files)
    image_list = []
    name_list = []


    for img_path in range(file_count):
        image_list.append(f"{path}{os.sep}{img_path}.jpg")
        name_list.append(f"{img_path}")

    sublist = [image_list[i:i + 5] for i in range(0, len(image_list), 5)]
    sublist_names = [name_list[i:i + 5] for i in range(0, len(name_list), 5)]

    if len(sublist) != len(sublist_names):
        raise Exception("Batch list path size is mot equals batch list name size.")

    for it, batch in enumerate(sublist):
        #print(sublist_names[it])
        process_file(batch, train, augment_type, folder, sublist_names[it])
        #print("Something happened")


if __name__ == '__main__':
    files = next(os.walk(original_img_dir_train))[1]
    file_count_train = len(files)
    files = next(os.walk(original_img_dir_test))[1]
    file_count_test = len(files)

    total = file_count_train + file_count_test

    train_folder_list = []
    test_folder_list = []

    for it in range(file_count_train):
        path = original_img_dir_train + f"{os.sep}{it}"
        train_folder_list.append(path)

    for it in range(file_count_test):
        path = original_img_dir_test + f"{os.sep}{it}"
        test_folder_list.append(path)

    #Ready to start multiprocessing
    processes = []
    multiprocessing.set_start_method('spawn', force=True)
    it = 0
    while True:
        time.sleep(0.01)

        if it == 80:
            print("Last of the train data is being processed")
            break

        if len(processes) < 60:
            worker_data = train_folder_list[it]
            if it < 954:
                new_worker = Process(target=process_folder, args=(worker_data, True, 'low', it))
                new_worker.start()
                processes.append(new_worker)
                it+=1
            elif it > 954 and it < (984 + 98):
                new_worker = Process(target=process_folder, args=(worker_data, True, 'medium', it))
                new_worker.start()
                processes.append(new_worker)
                it += 1
            else:
                new_worker = Process(target=process_folder, args=(worker_data, True, 'high', it))
                new_worker.start()
                processes.append(new_worker)
                it += 1

        new_worker = []
        for t in processes:
            if t.is_alive():
                new_worker.append(t)
        processes = new_worker.copy()

    for process in processes:
        process.join()



    processes = []
    multiprocessing.set_start_method('spawn', force=True)
    it = 0
    while True:
        time.sleep(0.01)

        if it == len(test_folder_list):
            print("Last of the train data is being processed")
            break

        if len(processes) < 60:
            worker_data = test_folder_list[it]
            if it < 236:
                new_worker = Process(target=process_folder, args=(worker_data, False, 'low', it))
                new_worker.start()
                processes.append(new_worker)
                it += 1
            elif it > 236 and it < (246 + 25):
                new_worker = Process(target=process_folder, args=(worker_data, False, 'medium', it))
                new_worker.start()
                processes.append(new_worker)
                it += 1
            else:
                new_worker = Process(target=process_folder, args=(worker_data, False, 'high', it))
                new_worker.start()
                processes.append(new_worker)
                it += 1

        new_worker = []
        for t in processes:
            if t.is_alive():
                new_worker.append(t)
        processes = new_worker.copy()

    for process in processes:
        process.join()

