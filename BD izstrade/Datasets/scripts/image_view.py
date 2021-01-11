import matplotlib.pyplot as plt
from PIL import Image
from skimage import color
import os
import matplotlib.gridspec as gridspace
import json
import numpy as np

gs = gridspace.GridSpec(1, 3)

augmented_directory = r'C:\Users\37120\Documents\BachelorThesis\image_data\flickr30k_augmented0'
original_directory = r'C:\Users\37120\Documents\BachelorThesis\image_data\flickr30k_images\flickr30k_images'
greyscale_directory = r'C:\Users\37120\Documents\BachelorThesis\image_data\flickr30k_images_greyscale'


# Every memmap file is in cielab colorspace.
# To compare images we need to convert them to rgb
def compare():
    with open(greyscale_directory + r'\\json_data.json') as json_file:
        greyscale_json = json.load(json_file)

    with open(augmented_directory + r'\\json_data.json') as json_file:
        augmented_json = json.load(json_file)

    # Print data properties
    colorspace = greyscale_json["Colorspace"]
    height = greyscale_json["image_height"]
    width = augmented_json["image_width"]
    print(fr'Colorspace:{colorspace} Height:{height} Witdh:{width}')
    # Comparing images
    for i in range(30):
        filename = greyscale_json['image'][i]['filename']
        pre, ext = os.path.splitext(filename)
        filename = pre + r'.jpg'
        image = Image.open(original_directory + r'\\' + filename).convert("RGB")
        image = image.resize((width, height), Image.ANTIALIAS)
        image_lab = color.rgb2lab(image)
        image_lab = color.lab2rgb(image_lab)
        fig = plt.figure(figsize=(48, 32), dpi=50)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title("Original", fontsize = 80)
        ax1.imshow(image_lab)

        greyscale_memmap = np.memmap(greyscale_directory+r'\\'+pre+r'.dat', dtype='float16', mode='r' , shape=(height, width, 3))
        greyscale_image = np.array(greyscale_memmap[:], dtype='float32')
        greyscale_image = color.lab2rgb(greyscale_image)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title("Greyscale", fontsize = 80)
        ax2.imshow(greyscale_image)
        fig.suptitle("CIELAB colorspace images", fontsize=120)
        augmented_memmap = np.memmap(augmented_directory + r'\\' + pre + r'.dat', dtype='float16', mode='r',
                                     shape=(height, width, 3))
        augmented_image = np.array(augmented_memmap[:], dtype='float32')
        augmented_image = color.lab2rgb(augmented_image)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_title("Augmented", fontsize = 80)
        ax3.imshow(augmented_image)
        plt.show()


if __name__ == '__main__':
    compare()
