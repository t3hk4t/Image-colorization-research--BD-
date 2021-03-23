import cv2
import os

save_dir = r'C:\Users\37120\Documents\BachelorThesis\image_data\video_framed_dataset'
dataset_dir = r'C:\Users\37120\Documents\BachelorThesis\image_data\video_dataset'

def process_file(path, subtype, iter):
    vidcap = cv2.VideoCapture(path) #video capturre file

    #Now we need to create a folder to save file at
    current_save_dir = f"{save_dir}\\{subtype}\\{iter}"

    if not os.path.exists(f"{save_dir}\\{subtype}\\{iter}"):
        os.makedirs(f"{save_dir}\\{subtype}\\{iter}")


    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        hasFrames, image = vidcap.read()
        if hasFrames:
            cv2.imwrite(f"{save_dir}\\{subtype}\\{os.path.splitext(path)[0]}.jpg", image)  # save frame as JPG file
        return hasFrames

    sec = 0
    frameRate = 0.04166666666  # //it will capture image in each 0.5 second
    count = 1
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)


if __name__ == '__main__':
    for subtype_dir in os.scandir(dataset_dir):
        iter = 0
        for video in os.scandir(subtype_dir):
            if video.path.endswith(".mp4") and video.is_file():
                process_file(video.path)