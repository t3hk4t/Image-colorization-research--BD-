import os
import cv2

def convert_video_to_frames(args, img_count : int,force=False):
    vidcap = cv2.VideoCapture(args.video_path)

    save_dir = args.output_dir + os.sep + f'split_frames'

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
    frameRate = 0.04166666666  # //it will capture image in each 1/24 second
    count = 1
    success = getFrame(sec, frame)
    frame += 1
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec, frame)
        frame += 1
        print(f"{frame/10000} % done")
        if frame > img_count:
            break
    print(f"{args.video_path} has been split into frames")


def generate_list_of_imgs(output_dir:str):
    save_dir = output_dir + os.sep + f'split_frames'
    files = next(os.walk(save_dir + f"{os.sep}"))[2]
    file_count = len(files)
    if file_count == 0:
        raise Exception("Folder has no images to denoise")
    img_list = []
    for it in range(file_count):
        img_list.append(save_dir + os.sep + f'{it}.jpg')
    return img_list


def generate_3d_list_of_imgs(args):
    save_dir = args.output_dir + os.sep + f'split_frames'
    files = next(os.walk(save_dir + f"{os.sep}"))[2]
    file_count = len(files)
    if file_count == 0:
        raise Exception("Folder has no images to denoise")

    imgs_list = []
    imgs = next(os.walk(save_dir))[2]
    file_count = len(imgs)
    for it in range(file_count):
        if it > 2100 :
            path = save_dir + f"{os.sep}{it}.jpg"
            imgs_list.append(path)


    dataset_samples = []

    for it in range(3):
        sample = []
        for idx in range(5):
            sample.append(imgs_list[idx])
        dataset_samples.append(sample.copy())

    for i in range(1, 294, 1):
        sample = []
        for idx in range(5):
            sample.append(imgs_list[i + idx])
        dataset_samples.append(sample.copy())

    for i in range(3):
        sample = []
        for idx in range(5):
            sample.append(imgs_list[299 - 5 + idx])
        dataset_samples.append(sample.copy())

    return dataset_samples


def generate_3d_list_of_imgs_synthetic(args, path_sy):
    save_dir = path_sy
    files = next(os.walk(save_dir + f"{os.sep}"))[2]
    file_count = len(files)
    if file_count == 0:
        raise Exception("Folder has no images to denoise")

    imgs_list = []
    imgs = next(os.walk(save_dir))[2]
    file_count = len(imgs)
    for it in range(file_count):
        path = save_dir + f"{os.sep}{it}.jpg"
        imgs_list.append(path)

    dataset_samples = []

    for it in range(3):
        sample = []
        for idx in range(5):
            sample.append(imgs_list[idx])
        dataset_samples.append(sample.copy())

    for i in range(1, 160, 1):
        sample = []
        for idx in range(5):
            sample.append(imgs_list[i + idx])
        dataset_samples.append(sample.copy())

    for i in range(3):
        sample = []
        for idx in range(5):
            sample.append(imgs_list[164 - 5 + idx])
        dataset_samples.append(sample.copy())

    return dataset_samples