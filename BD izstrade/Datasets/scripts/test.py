import numpy as np
from PIL import Image

filename = r'C:\Users\37120\Documents\BachelorThesis\Bachelor thesis\BD izstrade\Datasets\flickr30k_images_memmap\36979.dat'
fp = np.memmap(filename, dtype='float16', mode='r', shape=(320,480))

image = np.array(fp[:], dtype='float32')

img = Image.fromarray(image)
img.show()