import os
import numpy as np
from pathlib import Path
from skimage import color
from PIL import Image, ImageEnhance, ImageFilter
import json
import sys
import random
from blend_modes import multiply
from blend_modes import darken_only
it = 0
location_test = r'C:\Users\37120\Documents\BachelorThesis\image_data\dataset_test_2\test'
location_train = r'C:\Users\37120\Documents\BachelorThesis\image_data\dataset_test_2\train'
original_img_dir = r'C:\Users\37120\Documents\BachelorThesis\image_data\flickr30k_images\flickr30k_images'
noise_dir = r'C:\Users\37120\Documents\BachelorThesis\noise_data'


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

def generate_memmap(greyscale_image: np.ndarray, augmented_image: np.ndarray, path, train = True):
    global it
    if train:
        out_data = np.concatenate((greyscale_image, augmented_image), axis=2)
        print(out_data.shape)
        ref_img = out_data[:,:,0]
        print(ref_img.shape)
        ref_img2 = out_data[:,:, 1]
        print(ref_img2.shape)
        import matplotlib.pyplot as plt
        plt.imshow(np.concatenate([ref_img, ref_img2], axis=1), 'gray')
        #plt.show()
        plt.savefig(f'test\\{it}books_read.png')
        it = it + 1

        # filename = Path(path).stem
        # memmap_folder = location_train+f'//{filename}'
        # if not os.path.exists(memmap_folder):
        #     os.makedirs(memmap_folder)
        # filename = filename + '.dat'
        #
        # memmap_location = memmap_folder + r'//' + r'data.bin'
        #
        # json_data = {
        #     'Colorspace': 'CieLab',
        #     'filename': filename,
        #     'shape': [320, 480, 2],
        #     'original_images': original_img_dir,
        #     'features': {'grey': 1, 'augmented': 2}}
        #
        #
        # fp = np.memmap(memmap_location, dtype='float16', mode='w+', shape=out_data.shape)
        # fp[:] = out_data[:]
        # del fp
        # save_json(memmap_folder + r'//'+"data.json",json_data)
    else:
        out_data = np.concatenate((greyscale_image, augmented_image), axis=2)
        # print(out_data.shape)
        # ref_img = out_data[:,:,0]
        # print(ref_img.shape)
        # ref_img2 = out_data[:,:, 1]
        # print(ref_img2.shape)
        # import matplotlib.pyplot as plt
        # plt.imshow(np.concatenate([ref_img, ref_img2]), 1)
        # plt.show()

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

def process_file(path, train):

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
    alpha_noise_add = random.randint(1, 4)
    for i in range(alpha_noise_add):
        dir_new = r'C:\Users\37120\Documents\BachelorThesis\noise_data\\'
        noise = getRandomFile(noise_dir)
        dir_new = dir_new + noise + '\\'
        deterioration_overlay_img = Image.open(dir_new + getRandomFile(dir_new)).convert("RGBA")
        deterioration_overlay_img = deterioration_overlay_img.resize((480, 320), Image.ANTIALIAS)

        if random.randint(0, 100) <= 50:
            deterioration_overlay_img = deterioration_overlay_img.transpose(Image.FLIP_LEFT_RIGHT) #Flip horizontal
        if random.randint(0, 100) <= 50:
            deterioration_overlay_img = deterioration_overlay_img.transpose(Image.FLIP_TOP_BOTTOM) #Flip vertical
        deg = random.uniform(-5.0, 5.0)
        deterioration_overlay_img = deterioration_overlay_img.rotate(deg)

        scale = random.uniform(1.2, 1.5) * 2
        w, h = deterioration_overlay_img.size
        deterioration_overlay_img = deterioration_overlay_img.crop((w/2 - w / scale, h/2 - h / scale,
                        w/2 + w / scale, h/2 + h / scale))
        deterioration_overlay_img = deterioration_overlay_img.resize((w, h), Image.LANCZOS)


        background_img = np.array(original_image)  # Inputs to blend_modes need to be numpy arrays.
        background_img_float = background_img.astype(float)  # Inputs to blend_modes need to be floats.

        foreground_img = np.array(deterioration_overlay_img)  # Inputs to blend_modes need to be numpy arrays.
        foreground_img_float = foreground_img.astype(float)  # Inputs to blend_modes need to be floats.

        image_add_type = random.randint(1, 2)
        if image_add_type == 1:
            new_img = darken_only(background_img_float, foreground_img_float, 1)
        elif image_add_type == 2:
            new_img = multiply(background_img_float, foreground_img_float, 1)

        # Convert blended image back into PIL image
        blended_img = np.uint8(new_img)  # Image needs to be converted back to uint8 type for PIL handling.
        original_image = Image.fromarray(blended_img)


    if random.randint(0, 100) <= 10:  # Gaussian noise
        alpha = random.uniform(0, 0.02)
        ata = np.asarray(original_image)
        salt_pepper = sp_noise(ata, alpha)
        im2 = Image.fromarray(salt_pepper)
        original_image.paste(im2, (0, 0), im2)
    if random.randint(0, 100) <= 30:  # Blur
        original_image = original_image.filter(ImageFilter.BoxBlur(1))



    rgb_image = original_image.convert('RGB')
    image_lab = color.rgb2lab(rgb_image)
    image_lab[..., 1] = image_lab[..., 2] = 0
    augmented_image = np.asarray(image_lab)
    augmented_image = np.delete(augmented_image, 1, 2)
    augmented_image = np.delete(augmented_image, 1, 2)
     # FINAL GREYSCALE IMAGE THAT IS (320, 480)
    generate_memmap(greyscale_image,augmented_image, path, train = train)


if __name__ == '__main__':
    for it, img in enumerate(os.scandir(original_img_dir)):
        if img.path.endswith(".jpg") and img.is_file():
            if it < 31783 * 0.80:
                process_file(img.path, True)
            else:
                process_file(img.path, False)
        if it%100 == 0:
            print(f"{it} out of 30k images finished. {(it/31783) * 100} % done")
