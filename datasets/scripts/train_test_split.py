import os
import argparse

parser = argparse.ArgumentParser(description='Model trainer')
parser.add_argument('-splitGreyscale', default=True, type=bool)
parser.add_argument('-splitAugmented', default=True, type=bool)
parser.add_argument('-splitOriginal', default=False, type=bool)
parser.add_argument('-augmented_dir', default=r'C:\Users\37120\Documents\BachelorThesis\image_data\flickr30k_augmented0', type=str)
parser.add_argument('-greyscale_dir', default=r'C:\Users\37120\Documents\BachelorThesis\image_data\flickr30k_images_greyscale', type=str)
parser.add_argument('-original_dir', default=r'C:\Users\37120\Documents\BachelorThesis\image_data\flickr30k_images\flickr30k_images', type=str)
parser.add_argument('-split_augmented_dir', default=r'C:\Users\37120\Documents\BachelorThesis\image_data\flickr30k_images\flickr30k_images', type=str)
parser.add_argument('-split_greyscale_dir', default=r'C:\Users\37120\Documents\BachelorThesis\image_data\flickr30k_images\flickr30k_images', type=str)
parser.add_argument('-split_original_dir', default=r'C:\Users\37120\Documents\BachelorThesis\image_data\flickr30k_images\flickr30k_images', type=str)
args = parser.parse_args()

def split():
    if not os.path.exists(args.split_greyscale_dir) and args.splitGreyscale: # First we check if dir exists. If not, create them
        os.makedirs(args.split_greyscale_dir)

    if not os.path.exists(args.split_augmented_dir) and args.splitAugmented: # First we check if dir exists. If not, create them
        os.makedirs(args.split_augmented_dir)

    if not os.path.exists(args.split_original_dir) and args.splitOriginal: # First we check if dir exists. If not, create them
        os.makedirs(args.split_original_dir)



if __name__ == "__main__":
    split()