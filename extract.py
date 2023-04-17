import sys
import os
import os.path as ops

import argparse
from tqdm import tqdm


def arg_parse():
    parser = argparse.ArgumentParser(description='Extract frames from videos')

    parser.add_argument('-d', '--data_dir', required=True, type=str, help='directory to extract frames')
    parser.add_argument('-o', '--output_dir', required=True, type=str, help='directory to save extracted results')
    parser.add_argument('-n', '--frame_rate', default=None, type=int, help='extract framerate of videos')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()
    data_path = args.data_dir
    output_path = args.output_dir
    frame_rate = args.frame_rate

    videos = list(filter(lambda x: not x.startswith('.') and (x.endswith('mp4') or x.endswith('mov') or x.endswith('m4v')), os.listdir(data_path)))
    videos.sort()

    for video_file in tqdm(videos):
        video_path = ops.join(data_path, video_file)
        img_folder = ops.join(output_path, '.'.join(video_file.split('.')[:-1]))

        # make new director
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
        else:
            print('Overlapped videos: {}'.format(video_file))
            sys.exit(0)

        # deal with YUV videos
        # cmdline = f'ffmpeg -s:v 1920x1080 -r 1 -i {video_path} -r 1 {img_folder}/%06d.png -loglevel quiet'
        if frame_rate is not None:
            cmdline = f'ffmpeg -i {video_path} -r {frame_rate} {img_folder}/%06d.png -loglevel quiet'
        else:
            cmdline = f'ffmpeg -i {video_path} {img_folder}/%06d.png -loglevel quiet'
        
        os.system(cmdline)
