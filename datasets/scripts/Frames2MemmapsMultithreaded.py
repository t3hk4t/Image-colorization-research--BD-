import os
import numpy as np
from pathlib import Path
from skimage import color
from PIL import Image, ImageEnhance, ImageFilter
import json
import sys
import argparse
import random
import concurrent.futures
save_dir = r'C:\Users\37120\Documents\BachelorThesis\image_data\video_framed_dataset'
dataset_dir = r'C:\Users\37120\Documents\BachelorThesis\image_data\video_dataset'
train_list = []
test_list = []
validate_list = []

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

def generate_memmap(greyscale_image: np.ndarray, augmented_image: np.ndarray, path, subtype:str):
    if train:
        out_data = np.concatenate((greyscale_image, augmented_image), axis=2)

        filename = Path(path).stem
        memmap_folder = location_train+f'//{filename}'
        if not os.path.exists(memmap_folder):
            os.makedirs(memmap_folder)
        filename = filename + '.dat'

        memmap_location = memmap_folder + r'//' + r'data.bin'

        json_data = {
            'Colorspace': 'CieLab',
            'filename': filename,
            'shape': [320, 480, 2],
            'original_images': original_img_dir,
            'features': {'grey': 1, 'augmented': 2}}


        fp = np.memmap(memmap_location, dtype='float16', mode='w+', shape=out_data.shape)
        fp[:] = out_data[:]
        del fp
        save_json(memmap_folder + r'//'+"data.json",json_data)
    else:
        out_data = np.concatenate((greyscale_image, augmented_image), axis=2)

        filename = Path(path).stem
        memmap_folder = location_test + f'//{filename}'
        if not os.path.exists(memmap_folder):
            os.makedirs(memmap_folder)
        filename = filename + '.dat'

        memmap_location = memmap_folder + r'//' + r'data.bin'

        json_data = {
            'Colorspace': 'CieLab',
            'filename': filename,
            'shape': [320, 480, 2],
            'original_images': original_img_dir,
            'features': {'grey': 1, 'augmented': 2}}

        fp = np.memmap(memmap_location, dtype='float16', mode='w+', shape=out_data.shape)
        fp[:] = out_data[:]
        del fp
        save_json(memmap_folder + r'//' + "data.json", json_data)


def validate(it):
    pass

def worker(path, args, subtype : str, it):

    img_list = []
    for video in os.scandir(path):
        if video.path.endswith(".mp4") and video.is_file():
            img_list.append(video.path)

    for idx, img in enumerate(img_list):
        if not os.path.exists(args.save_dir +f'//{subtype}//{it}//{idx}'):
            os.makedirs(args.save_dir +f'//{subtype}//{it}//{idx}')

    greyscale_image = Image.open(path).convert("RGB")
    greyscale_image = greyscale_image.resize((480, 320), Image.ANTIALIAS)
    image_lab = color.rgb2lab(greyscale_image)
    image_lab[..., 1] = image_lab[..., 2] = 0
    greyscale_image = np.asarray(image_lab)
    greyscale_image = np.delete(greyscale_image, 1,2)
    greyscale_image = np.delete(greyscale_image, 1,2)
    # FINAL GREYSCALE IMAGE THAT IS (320, 480)


    original_image = (Image.open(path)).convert("RGBA")
    original_image = original_image.resize((480, 320), Image.ANTIALIAS)
    original_image.putalpha(128)
    alpha_noise_add = random.randint(3, 8)
    for i in range(alpha_noise_add):
        dir_new = r'C:\Users\37120\Documents\BachelorThesis\noise_data\\'
        noise = getRandomFile(noise_dir)
        dir_new = dir_new + noise + '\\'
        deterioration_overlay_img = Image.open(dir_new + getRandomFile(dir_new))
        deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)
        deterioration_overlay_img = deterioration_overlay_img.convert("RGBA") #Now I want to add some augmentations like rotate etc


        if random.randint(0, 100) <= 50:
            deterioration_overlay_img = deterioration_overlay_img.transpose(Image.FLIP_LEFT_RIGHT) #Flip horizontal
        if random.randint(0, 100) <= 50:
            deterioration_overlay_img = deterioration_overlay_img.transpose(Image.FLIP_TOP_BOTTOM) #Flip vertical
        deg = random.uniform(-5.0, 5.0)
        deterioration_overlay_img = deterioration_overlay_img.rotate(deg)

        datas = deterioration_overlay_img.getdata()
        newData = []
        for item in datas:
            if item[0] > 170 and item[1] > 170 and item[3] > 170:
                newData.append((255, 255, 255, 0))
            else:
                newData.append((item[0], item[1], item[2], random.randint(50, 150)))

        deterioration_overlay_img.putdata(newData)

        if random.randint(0, 100) <= 50: #Invert colors
            r, g, b, a = deterioration_overlay_img.split()
            r, g, b = map(invert, (r, g, b))
            deterioration_overlay_img = Image.merge(deterioration_overlay_img.mode, (r, g, b, a))


        image_new = Image.new("RGBA", original_image.size)
        image_new = Image.alpha_composite(image_new, original_image)
        image_new = Image.alpha_composite(image_new, deterioration_overlay_img)
        image = image_new.copy()

    if random.randint(0, 100) <= 10:  # Gaussian noise
        alpha = random.uniform(0, 0.02)
        ata = np.asarray(image)
        salt_pepper = sp_noise(ata, alpha)
        im2 = Image.fromarray(salt_pepper)
        image.paste(im2, (0, 0), im2)
    if random.randint(0, 100) <= 30:  # Blur
        image = image.filter(ImageFilter.BoxBlur(1))


    rgb_image = image.convert('RGB')
    image_lab = color.rgb2lab(rgb_image)
    image_lab[..., 1] = image_lab[..., 2] = 0
    augmented_image = np.asarray(image_lab)
    augmented_image = np.delete(augmented_image, 1, 2)
    augmented_image = np.delete(augmented_image, 1, 2)
     # FINAL GREYSCALE IMAGE THAT IS (320, 480)

    generate_memmap(greyscale_image,augmented_image, path, train = train)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Dataprocessor')
    parser.add_argument('-dataset_dir', default=r'C:\Users\37120\Documents\BachelorThesis\image_data\video_dataset', type=str)
    parser.add_argument('-save_dir', default=r'C:\Users\37120\Documents\BachelorThesis\image_data\video_framed_dataset', type=str)
    parser.add_argument('-noise_dir', default=r'C:\Users\37120\Documents\BachelorThesis\noise_data', type=str)
    parser.add_argument('-workers', default=30, type=int)
    args, _ = parser.parse_known_args()

    for subtype_dir in os.scandir(args.dataset_dir):
        if subtype_dir.path == f'{args.dataset_dir}\\train':
            for video_dirs in os.scandir(subtype_dir):
                train_list.append(video_dirs.path)
        elif subtype_dir.path == f'{args.dataset_dir}\\test':
            for video_dirs in os.scandir(subtype_dir):
                test_list.append(video_dirs.path)
        elif subtype_dir.path == f'{args.dataset_dir}\\validate':
            for video_dirs in os.scandir(subtype_dir):
                validate_list.append(video_dirs.path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        # Start the load operations and mark each future with its URL
        futures_train = []
        futures_test = []
        futures_validate = []
        for it, path in enumerate(train_list):
            futures_train.append(executor.submit(worker,args, path, "train", it))
        for it, path in enumerate(test_list):
            futures_train.append(executor.submit(worker, args, path, "test", it))
        for it, path in enumerate(validate_list):
            futures_train.append(executor.submit(worker, args, path, "validate", it))

        for future in concurrent.futures.as_completed(futures_train):
            print("lol")

        for future in concurrent.futures.as_completed(futures_test):
            print("lol")

        for future in concurrent.futures.as_completed(futures_validate):
            print("lol")
