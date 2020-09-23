import os
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from pathlib import Path
import json
import random

json_location = r'C:\Users\37120\Documents\BachelorThesis\Bachelor thesis\BD izstrade\Datasets\flickr30k_images_memmap\json_data.json'
directory = r'C:\Users\37120\Documents\BachelorThesis\Bachelor thesis\BD izstrade\Datasets\flickr30k_images\flickr30k_images'
out_dir = r'C:\Users\37120\Documents\BachelorThesis\Bachelor thesis\BD izstrade\Datasets\flickr30k_augmented'
noise_dir = r'C:\Users\37120\Documents\BachelorThesis\Bachelor thesis\BD izstrade\Datasets\noise_data'
json_data = {'image': []}


def file_lengthy(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def save_json():
    with open(json_location, 'w') as out:
        json.dump(json_data, out)


def generate_memmap(shape, path, image: np.ndarray, it):
    path2 = out_dir+str(it.real)
    if not os.path.exists(path2):
        os.makedirs(path2)

    filename = Path(path).stem
    filename = filename + '.dat'
    filename = path2 + r'\\' + filename
    json_data['image'].append({
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


def process_file(path, it):
    global dir_new
    image = (Image.open(path)).convert("RGBA")
    image = image.resize((480, 320), Image.ANTIALIAS)
    dir_new = r'C:\Users\37120\Documents\BachelorThesis\Bachelor thesis\BD izstrade\Datasets\noise_data\\'
    noise = getRandomFile(noise_dir)
    dir_new = dir_new + noise + '\\'
    image2 = Image.open(dir_new + getRandomFile(dir_new)).convert("RGBA")
    image2 = image2.resize((480, 320), Image.ANTIALIAS)

    datas = image2.getdata()
    newData = []
    for item in datas:
        if item[0] > 150 and item[1] > 150 and item[3] > 150:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    image2.putdata(newData)
    image.paste(image2, (0, 0), image2)
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
        alpha = random.uniform(0, 0.04)
        ata = np.asarray(image)
        salt_pepper = sp_noise(ata, alpha)
        im2 = Image.fromarray(salt_pepper)
        image.paste(im2, (0, 0), im2)

    if random.randint(0, 100) <= 10:  # Saturation
        alpha = random.uniform(0.3, 1)
        saturation_control.enhance(alpha)

    data = np.asarray(image)
    data = np.mean(data, axis=2)

    img = Image.fromarray(data)
    img.show()

    generate_memmap(data.shape, path, data, it)


if __name__ == '__main__':
    for i in range(3):
        for it, img in enumerate(os.scandir(directory)):
            if img.path.endswith(".jpg") and img.is_file():
                process_file(img.path, i)
            if it % 20 == 0:
                print(f"{it} out of 30k images finished. {(it / 30000) * 100} % done")
        save_json()
        json_data.clear()

