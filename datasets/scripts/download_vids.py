from pytube import YouTube
import os

directory = r'C:\Users\37120\Documents\BachelorThesis\Bachelor thesis\datasets\video_urls'
test = r'C:\Users\37120\Documents\BachelorThesis\image_data\video_dataset\test'
train = r'C:\Users\37120\Documents\BachelorThesis\image_data\video_dataset\train'
validate = r'C:\Users\37120\Documents\BachelorThesis\image_data\video_dataset\validate'

def file_lengthy(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def process_file(path):
    lines = file_lengthy(path)
    files_done = 0
    percent_finished = 0.0
    with open(path, "r") as file:
        print("Downloading dataset: " + path)
        print(f"{files_done} out of {lines} downloads finished. 0% finished")
        for video_url in file:
            url = video_url.strip()
            download_vid(url, path)
            files_done += 1
            percent_finished = (files_done/lines) * 100
            print(f"{files_done} out of {lines} downloads finished. {percent_finished}% finished")


def download_vid(path, root_file_name):
    try:
        yt_obj = YouTube(path)
        if os.path.basename(root_file_name) == 'video_urls_test.txt':
            yt_obj.streams.get_lowest_resolution().download(output_path=test)
        elif os.path.basename(root_file_name) == 'video_urls_train.txt':
            yt_obj.streams.get_lowest_resolution().download(output_path=train)
        else:
            yt_obj.streams.get_lowest_resolution().download(output_path=validate)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    for video in os.scandir(directory):
        if video.path.endswith(".txt") and video.is_file():
            process_file(video.path)

