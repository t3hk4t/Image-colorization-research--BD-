import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import numpy as np
import multiprocessing
from PIL import Image
import argparse
import json
from skimage import color
from modules import video_utils
from modules import dict_to_obj
from models import unetplusplus
from models import temporal_unet_plus_pus
from models import DrunkUNET
from models import TUNETREF
from modules import loss_functions
from modules_core import autoencoder_ref
from datasets.scripts import Frames2MemmapsMultithreaded

parser = argparse.ArgumentParser(description='Model inference script')
parser.add_argument('-model_path', default=r'C:\Users\37120\Documents\results\models\last_checkpoint.pt', type=str)
parser.add_argument('-args_path', default=r'C:\Users\37120\Documents\results\models\args.json', type=str)
parser.add_argument('-video_path', default=r'C:\Users\37120\Documents\BachelorThesis\image_data\video_dataset\train\videoplayback.mp4', type=str)
parser.add_argument('-jobName', default='inference_non_vintage_1', type=str)
parser.add_argument('-force', default=False, type=bool)
parser.add_argument('-augmented_video_dir', default=f'None', type=str)
parser.add_argument('-loss_type', default='MSSIM_L1', type=str)
parser.add_argument('-is_vintage', default=False, type=bool)
parser.add_argument('-augment_level', default="medium", type=str)
parser.add_argument('-model', default="autoencoder", type=str)
parser.add_argument('-output_dir', default=r'C:\Users\37120\Documents\results', type=str)
parser.add_argument('-save_output_frames', default=False, type=bool)
parser.add_argument('-save_input_frames', default=False, type=bool)
parser.add_argument('-save_video', default=False, type=bool)
parser.add_argument('-num_workers', default=multiprocessing.cpu_count(), type=int)
args, _ = parser.parse_known_args()

idx = 0

def loadModel(args):
    with open(args.args_path) as f:
        data = json.load(f)
    data = dict_to_obj.DictToObj(**data)

    if args.model == "unet":
        model = unetplusplus.Model(data)
    elif args.model == "tunet":
        model = temporal_unet_plus_pus.Model(data)
    elif args.model == "autoencoder":
        model = autoencoder_ref.Model()
    elif args.model == "drunkunet":
        model = DrunkUNET.Model(data)
    elif args.model == "tunetref":
        model = TUNETREF.Model(data)
    else:
        raise Exception("Unknown model has been passed as an argument")
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.eval()


def process3D(images, model, losses_list):
    global idx

    if args.loss_func == "MSSIM_L1":
        loss_func = loss_functions.MS_SSIM_L1_LOSS()
    elif args.loss_func == "MSE":
        loss_func = torch.nn.MSELoss()
    elif args.loss_func == "L1":
        loss_func = torch.nn.L1Loss()
    else:
        raise Exception("Unknown loss function supplied as an argument")


    for image_path in images:
        sample = []
        before = []
        for it,  item in enumerate(image_path):
            image = Image.open(item).convert("RGB")
            image = image.resize((480, 320), Image.ANTIALIAS)
            image_lab = color.rgb2lab(image)
            image_lab[..., 1] = image_lab[..., 2] = 0
            greyscale_image = np.asarray(image_lab, dtype='float32')
            greyscale_image = np.delete(greyscale_image, 1, 2)
            greyscale_image = np.delete(greyscale_image, 1, 2)
            if it == 2:
                before.append(greyscale_image.copy())
            greyscale_image = (greyscale_image - 0) / 100
            greyscale_image = np.expand_dims(greyscale_image, axis=0)

            if it == 0:
                sample.append(greyscale_image.copy())
            else:
                sample[0] = np.concatenate([sample[0], greyscale_image], axis=0)

        sample[0] = np.swapaxes(sample[0], 1, 3)
        sample[0] = np.swapaxes(sample[0], 0, 1)
        sample[0] = np.swapaxes(sample[0], 2, 3)
        sample[0] = np.expand_dims(sample[0], axis=0)
        output = model((torch.from_numpy(sample[0])))
        output = output.cpu().detach().numpy()
        output = output * 100
        import matplotlib.pyplot as plt


        img = np.concatenate([before[0][:,:,0], output[0,0,2,:,:]], axis=1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=100)
        plt.savefig(f'{args.output_dir}{os.sep}teste_3d_2{os.sep}{idx}.jpg', dpi=150)
        idx += 1

id = 0
def process3D_synthetic(images_augmented, images_clean, model):
    global id

    loss_func_mssim = loss_functions.MS_SSIM_L1_LOSS()
    loss_func_MSE = torch.nn.MSELoss()
    loss_func_L1 = torch.nn.L1Loss()

    loss1 = []
    loss2 = []
    loss3 = []

    for i in range(len(images_augmented)):
        sample = []
        before = []
        for it, item in enumerate(images_augmented[i]):
            image_augmented = Image.open(item).convert("RGB")
            image_clean = Image.open(images_clean[i][it]).convert("RGB")
            image_augmented = image_augmented.resize((480, 320), Image.ANTIALIAS)
            image_clean = image_clean.resize((480, 320), Image.ANTIALIAS)
            image_lab_augmented = color.rgb2lab(image_augmented)
            image_lab_clean = color.rgb2lab(image_clean)
            image_lab_augmented[..., 1] = image_lab_augmented[..., 2] = 0
            image_lab_clean[..., 1] = image_lab_clean[..., 2] = 0
            greyscale_image = np.asarray(image_lab_clean, dtype='float32')
            augmented_image = np.asarray(image_lab_augmented, dtype='float32')
            greyscale_image = np.delete(greyscale_image, 1, 2)
            greyscale_image = np.delete(greyscale_image, 1, 2)
            augmented_image = np.delete(augmented_image, 1, 2)
            augmented_image = np.delete(augmented_image, 1, 2)
            if it == 2:
                before.append(greyscale_image.copy())
            greyscale_image = (greyscale_image - 0) / 100
            greyscale_image = np.expand_dims(greyscale_image, axis=0)
            augmented_image = (augmented_image - 0) / 100
            augmented_image = np.expand_dims(augmented_image, axis=0)

            if it == 0:
                sample.append(augmented_image.copy())
            else:
                sample[0] = np.concatenate([sample[0], augmented_image], axis=0)

        sample[0] = np.swapaxes(sample[0], 1, 3)
        sample[0] = np.swapaxes(sample[0], 0, 1)
        sample[0] = np.swapaxes(sample[0], 2, 3)
        sample[0] = np.expand_dims(sample[0], axis=0)
        output = model((torch.from_numpy(sample[0])))
        output = output.cpu().detach().numpy()
        output = output * 100
        import matplotlib.pyplot as plt
        plt.imshow(output[0, 0, 2, :, :], cmap='gray', vmin=0, vmax=100)
        plt.savefig(f'{args.output_dir}{os.sep}teste_3d_2{os.sep}{id}.jpg', dpi=150)
        id = id+1
        print(loss1)
        print(loss2)



def main(args):
    try:
        video_utils.convert_video_to_frames(args, 240)
    except Exception as exc:
        warning = exc.args
        print(warning[0])
        print("Continuing data parsing with an existing video folder contents.")

    images_augmented = video_utils.generate_3d_list_of_imgs_synthetic(args, r"C:\Users\37120\Documents\results\split_frames_augmented")
    images_clean = video_utils.generate_3d_list_of_imgs_synthetic(args, r"C:\Users\37120\Documents\results\split_frames")

    model = loadModel(args)
    process3D_synthetic(images_augmented,images_clean, model)

if __name__ == '__main__':
    main(args)