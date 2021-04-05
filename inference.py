import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import numpy as np
import multiprocessing
from multiprocessing import Process
from PIL import Image
import tensorflow as tf
import argparse
import cv2
from skimage import color

from modules import radam
from models import unetplusplus
it = 0
def convert_video_to_frames(video_path : str, save_output_frames : bool, video_name : str, output_dir:str, force=False):
    global it
    vidcap = cv2.VideoCapture(video_path)

    save_dir = output_dir + os.sep + f'{video_name}_frames'

    if os.path.exists(save_dir) and force == False:
        raise Exception("Video path with frames already exists. Use -force True if you want to overwrite data")
    elif not os.path.exists(save_dir):
        os.makedirs(save_dir)

    frame = 0
    def getFrame(sec, frame):
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        hasFrames, image = vidcap.read()
        if hasFrames:
            image = cv2.resize(image, (480, 320))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if not cv2.imwrite(f"{save_dir}\\{frame}.jpg", image):
                raise Exception("Could not write image")  # save frame as JPG file
        return hasFrames

    sec = 0.0
    frameRate = 0.04166666666  # //it will capture image in each 0.5 second
    count = 1
    success = getFrame(sec, frame)
    frame += 1
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec, frame)
        frame += 1
        if frame > 300:
            break
    print(f"{video_path} has been split into frames")


def generate_list_of_imgs(video_name : str, output_dir:str):
    save_dir = output_dir + os.sep + f'{video_name}_frames'
    files = next(os.walk(save_dir + f"{os.sep}"))[2]
    file_count = len(files)
    if file_count == 0:
        raise Exception("Folder has no images to denoise")
    img_list = []
    for it in range(file_count):
        img_list.append(save_dir + os.sep + f'{it}.jpg')
    return img_list



def forward2D(image_path, output_folder : str, model_path : str):
    global it
    parser = argparse.ArgumentParser(description='Model trainer')
    parser.add_argument('-learning_rate', default=0.005, type=float)
    parser.add_argument('-is_deep_supervision', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-unet_depth', default=5, type=int)
    parser.add_argument('-first_conv_channel_count', default=8, type=int)
    parser.add_argument('-expansion_rate', default=2, type=int)

    # TODO add more params and make more beautitfull cuz this file is a mess
    args2, _ = parser.parse_known_args()
    model = unetplusplus.Model(args2)
    loss_func = torch.nn.MSELoss()
    optimizer = radam.RAdam(model.parameters(), lr=args2.learning_rate)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image = image.resize((480, 320), Image.ANTIALIAS)
    image_lab = color.rgb2lab(image)
    image_lab[..., 1] = image_lab[..., 2] = 0
    greyscale_image = np.asarray(image_lab, dtype='float32')
    greyscale_image = np.delete(greyscale_image, 1, 2)
    greyscale_image = np.delete(greyscale_image, 1, 2)
    before = greyscale_image.copy()
    greyscale_image = (greyscale_image - 0) / 100
    greyscale_image = np.swapaxes(greyscale_image, 0, 1)
    greyscale_image = np.swapaxes(greyscale_image, 0, 2)
    output = model(torch.from_numpy(greyscale_image))
    output = output.cpu().detach().numpy()
    output = output*100
    output = np.swapaxes(output, 0, 2)
    output = np.swapaxes(output, 0, 1)
    import matplotlib.pyplot as plt

    img1 = np.sum(before)
    print(img1)
    img2 = np.sum(output)
    print(img2)

    diff = img1 - img2

    print(f"images are {diff} pixel values different")

    img = np.concatenate([before, output], axis=1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=100)
    plt.savefig(f'{output_folder}{os.sep}{it}.jpg')
    it+=1


def forward3D():
    pass


def main(args):
    try:
        convert_video_to_frames(args.video_path, True, args.video_name, args.output_dir)
    except Exception as exc:
        warning = exc.args
        print(warning[0])
        print("Continuing data parsing with an existing video folder contents.")

    images = generate_list_of_imgs(args.video_name, args.output_dir)

    for item in images:
        forward2D(item, args.output_dir, args.conv2d_model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model inference script')
    parser.add_argument('-conv2d_model_path', default=r'C:\Users\37120\Documents\results\models\test1\last_checkpoint.pt', type=str)
    parser.add_argument('-conv3d_model_path', default='', type=str)
    parser.add_argument('-video_path', default=r'C:\Users\37120\Documents\results\test2.mp4', type=str)
    parser.add_argument('-video_name', default='test2_dummy', type=str)
    parser.add_argument('-force', default=False, type=bool)
    parser.add_argument('-augmented_video_dir', default=f'None', type=str)
    parser.add_argument('-is_vintage', default=True, type=bool)
    parser.add_argument('-use_temporal_network', default=False, type=bool)
    parser.add_argument('-output_dir', default=r'C:\Users\37120\Documents\results', type=str)
    parser.add_argument('-save_output_frames', default=False, type=bool)
    parser.add_argument('-save_input_frames', default=False, type=bool)
    parser.add_argument('-num_workers', default=multiprocessing.cpu_count(), type=int)
    args, _ = parser.parse_known_args()

    if args.is_vintage == False and args.augmented_video_dir == 'None':
        raise Exception("If video is not vintage, user must provide augmented video path. USAGE: -augmented_video_path path ")

    main(args)