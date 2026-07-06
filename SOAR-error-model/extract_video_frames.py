import os
import cv2
import numpy as np
import argparse
import json
from tqdm import tqdm

def convert_to_flat_set(nested_list):
    flat_list = []
    for item in nested_list:
        flat_list.extend(list(range(item[0], item[1]+1)))
    
    flat_list = set(flat_list)
    return flat_list


def extract_clip_frames_cv2(args):

    video_path = args.video_path
    video_id = video_path.split('/')[-1].replace('.mp4', '')

    clip_save_path = os.path.join(args.frames_dir, video_id)

    print(f'Saving frames for {clip_save_path}...')

    os.makedirs(clip_save_path, exist_ok=True)

    # Read this specific video
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    # Load oob_json
    start_frame = 0
    end_frame = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_blackout = []
    if args.oob_json:
        with open(args.oob_json, 'r') as f:
            formatted_oob_results = json.load(f)
        
        start_frame = formatted_oob_results['start']
        end_frame = formatted_oob_results['end'] + 1
        frames_to_blackout = convert_to_flat_set(formatted_oob_results['frames_to_black_out'])

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    num_frames = end_frame - start_frame

    progress_bar = tqdm(total=num_frames, desc="Extracting features")

    # Write frames of video in correct format to the right folder
    frame_num = 0
    progress_bar = tqdm(total=num_frames, desc="Saving Frames")
    desired_shorter_side = 384
    num_saved_frames = 0

    sampling_frame_interval = int(fps // args.sampling_frame_rate)
    while True:
        success, frame = vidcap.read()
        if frame_num >= end_frame or not success:
            break
        if frame_num % sampling_frame_interval == 0:
            original_height, original_width, _ = frame.shape
            if original_height < original_width:
                # Height is the shorter side
                new_height = desired_shorter_side
                new_width = np.round(
                    original_width*(desired_shorter_side/original_height)
                ).astype(np.int32)
            elif original_height > original_width:
                # Width is the shorter side
                new_width = desired_shorter_side
                new_height = np.round(
                    original_height*(desired_shorter_side/original_width)
                ).astype(np.int32)
            else:
                # Both are the same
                new_height = desired_shorter_side
                new_width = desired_shorter_side
            assert np.isclose(
                new_width/new_height,
                original_width/original_height,
                0.01
            )
            
            if frame_num not in frames_to_blackout: # skip over all frames to black out
                frame = cv2.resize(
                    frame,
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA
                )
                cv2.imwrite(
                    os.path.join(
                        clip_save_path,
                        f'{int(frame_num / sampling_frame_interval)}.jpeg'
                    ),
                    frame
                    # # NOTE: Frames are saved in BGR format
                    # cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
                )

            num_saved_frames += 1
            if frame_num % 1000 == 0:
                print(f'Wrote frame number {frame_num}')
        frame_num += 1
        progress_bar.update(1)

    vidcap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_dir', type=str)
    parser.add_argument('--sampling_frame_rate', type=int, default=10)
    parser.add_argument('--video_path', type=str, default=None)
    parser.add_argument('--oob_json', type=str, default=None)

    args = parser.parse_args()

    extract_clip_frames_cv2(args)