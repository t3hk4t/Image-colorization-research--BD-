import os
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from pathlib import Path
from skimage import color
import json
import time
import random

directory = r'C:\Users\37120\Documents\BachelorThesis\image_data\flickr30k_images\flickr30k_images'
noise_dir = r'C:\Users\37120\Documents\BachelorThesis\noise_data'

json_location_train = r'C:\Users\37120\Documents\BachelorThesis\image_data\flick30k_10_augmented_train'
out_dir_train = r'C:\Users\37120\Documents\BachelorThesis\image_data\flick30k_10_augmented_train'

json_location_test = r'C:\Users\37120\Documents\BachelorThesis\image_data\flick30k_10_augmented_test'
out_dir_test = r'C:\Users\37120\Documents\BachelorThesis\image_data\flick30k_10_augmented_test'


json_data_train = {
    'Colorspace': 'CieLab',
    'image_height': 320,
    'image_width': 480,
    'original_images': directory,
    'image': []}

json_data_test = {
    'Colorspace': 'CieLab',
    'image_height': 320,
    'image_width': 480,
    'original_images': directory,
    'image': []}


def save_json_train(it):
    with open(json_location_train+r'\\train.json', 'a+') as out:
        json.dump(json_data_train, out)

def save_json_test(it):
    with open(json_location_test+r'\\test.json', 'a+') as out:
        json.dump(json_data_test, out)



def generate_memmap(shape, path, image: np.ndarray, it, train = True):
    if train:
        path2 = out_dir_train + str(it.real)
        if not os.path.exists(path2):
            os.makedirs(path2)

        filename = Path(path).stem
        filename = filename + '.dat'
        filename = path2 + r'\\' + filename
        json_data_train['image'].append({
            'filename': os.path.basename(filename),
            'location': filename,
            'shape': shape
        })
        fp = np.memmap(filename, dtype='float16', mode='w+', shape=shape)
        fp[:] = image[:]
        del fp
    else:
        path2 = out_dir_test + str(it.real)
        if not os.path.exists(path2):
            os.makedirs(path2)

        filename = Path(path).stem
        filename = filename + '.dat'
        filename = path2 + r'\\' + filename
        json_data_test['image'].append({
            'filename': os.path.basename(filename),
            'location': filename,
            'shape': shape
        })
        fp = np.memmap(filename, dtype='float16', mode='w+', shape=shape)
        fp[:] = image[:]
        del fp


def getRandomFile(path):
    files = os.listdir(path)
    index = random.randrange(0, len(files))
    return files[index]


def sp_noise(image,prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def process_file(path, it, train = True):
    global dir_new
    image = (Image.open(path)).convert("RGBA")
    image = image.resize((480, 320), Image.ANTIALIAS)
    image.putalpha(128)
    data2 = np.asarray(image)
    data2 = np.mean(data2, axis=2)
    dir_new = r'C:\Users\37120\Documents\BachelorThesis\noise_data\\'
    alpha_noise_add = random.randint(3, 8)
    for i in range(alpha_noise_add):
        dir_new = r'C:\Users\37120\Documents\BachelorThesis\noise_data\\'
        noise = getRandomFile(noise_dir)
        dir_new = dir_new + noise + '\\'
        image2 = Image.open(dir_new + getRandomFile(dir_new))
        image2 = image2.resize((480, 320), Image.ANTIALIAS)
        image2 = image2.convert("RGBA")

        datas = image2.getdata()
        newData = []
        for item in datas:
            if item[0] > 170 and item[1] > 170 and item[3] > 170:
                newData.append((255, 255, 255, 0))
            else:
                newData.append((item[0], item[1], item[2], random.randint(50, 150)))
        image2.putdata(newData)
        image_new = Image.new("RGBA", image.size)
        image_new = Image.alpha_composite(image_new, image)
        image_new = Image.alpha_composite(image_new, image2)
        image = image_new.copy()

    brightness_control = ImageEnhance.Brightness(image)
    contrast_control = ImageEnhance.Contrast(image)
    saturation_control = ImageEnhance.Color(image)

    if random.randint(0,100) <= 20:  # brightness
        alpha = random.uniform(0.8, 1.2)
        brightness_control.enhance(alpha)

    if random.randint(0,100) <= 20:  # Contrast
        alpha = random.uniform(0.9, 1.0)
        contrast_control.enhance(alpha)

    if random.randint(0,100) <= 10:  # Gaussian noise
        alpha = random.uniform(0, 0.02)
        ata = np.asarray(image)
        salt_pepper = sp_noise(ata, alpha)
        im2 = Image.fromarray(salt_pepper)
        image.paste(im2, (0, 0), im2)

    if random.randint(0, 100) <= 10:  # Saturation
        alpha = random.uniform(0.3, 1)
        saturation_control.enhance(alpha)

    if random.randint(0, 100) <= 30:  # Blur
        alpha = random.uniform(0.3, 1)
        image = image.filter(ImageFilter.BoxBlur(1))
    rgb_image = image.convert('RGB')
    image_lab = color.rgb2lab(rgb_image)
    image_lab[..., 1] = image_lab[..., 2] = 0
    image_rgb = color.lab2rgb(image_lab)
    import matplotlib.pyplot as plt
    plt.imshow(image_rgb)
    plt.show()
    # data = np.asarray(image_lab)
    # if train:
    #     generate_memmap(data.shape, path, data, it, True)
    # else:
    #     generate_memmap(data.shape, path, data, it, False)



if __name__ == '__main__':

    for i in range(1):
        save_train = False
        for it, img in enumerate(os.scandir(directory)):
            if img.path.endswith(".jpg") and img.is_file():
                if it < 31783 * 0.85:
                    process_file(img.path, True)
                else:
                    if save_train is False:
                        save_json_train(i)
                        save_train = True
                    process_file(img.path, False)
            if it % 100 == 0:
                print(f"{it} out of 30k images finished. {(it / 31783) * 100} % done")
        save_json_test(i)
        json_data_train.clear()
        json_data_test.clear()

        json_data_train = {
            'Colorspace': 'CieLab',
            'image_height': 320,
            'image_width': 480,
            'original_images': directory,
            'image': []}

        json_data_test = {
            'Colorspace': 'CieLab',
            'image_height': 320,
            'image_width': 480,
            'original_images': directory,
            'image': []}

