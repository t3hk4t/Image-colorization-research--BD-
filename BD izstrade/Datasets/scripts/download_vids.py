from pytube import YouTube
import os

directory = r'C:\Users\37120\Documents\BachelorThesis\Bachelor thesis\BD izstrade\Datasets\video_urls'


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
            download_vid(url)
            files_done += 1
            percent_finished = (files_done/lines) * 100
            print(f"{files_done} out of {lines} downloads finished. {percent_finished}% finished")


def download_vid(path):
    try:
        yt_obj = YouTube(path)
        yt_obj.streams.get_lowest_resolution().download(output_path=
                                                        r'C:\Users\37120\Documents\BachelorThesis\Bachelor thesis\BD izstrade\Datasets\youtube_videos\test')
    except Exception as e:
        pass  # simple ignore for non existing videos


if __name__ == '__main__':
    for video in os.scandir(directory):
        if video.path.endswith(".txt") and video.is_file():
            print(video.path)
            process_file(video.path)
            break
