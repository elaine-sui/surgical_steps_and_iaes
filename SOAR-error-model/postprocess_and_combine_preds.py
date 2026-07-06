import json
import argparse
from lib2to3.pytree import convert
import pandas as pd
import os
import cv2


def convert_to_timestamp(frame_num, fps):
    num_seconds = int(frame_num / fps)
    num_hours = int(num_seconds / (60**2))
    num_seconds -= num_hours * (60**2)
    num_mins = int(num_seconds / 60)
    num_seconds -= num_mins * 60
    
    return f'{num_hours:02d}:{num_mins:02d}:{num_seconds:02d}'


def get_video_info(video_path, sampled_fps):
    vidcap = cv2.VideoCapture(video_path)
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    extracted_frame_interval = int(fps // sampled_fps)
    num_total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    vidcap.release()

    return {'video_duration': num_total_frames, 'fps': fps, 'extracted_frame_interval': extracted_frame_interval}


def format_interval_preds(args, video_info, json_file, threshold, buffer, error_name):
    with open(json_file, 'r') as f:
        preds = json.load(f)['predictions']

    formatted_preds = []
    for interval, score in preds.items():
        interval = interval.split('_')
        interval = [int(num) for num in interval]
        interval = [i * video_info['extracted_frame_interval'] for i in interval]

        if score > threshold:
            interval[0] = max(0, int(interval[0] - buffer))
            interval[1] = min(int(interval[1] + buffer), video_info['video_duration'])
            formatted_preds.append(interval + [score])
            assert len(formatted_preds[-1]) == 3
    
    # Merge preds
    if len(formatted_preds) > 1:
        formatted_preds = merge_preds(formatted_preds)
    df = convert_preds_to_df(args, video_info, formatted_preds, error_name)

    return df


def format_frame_preds(args, video_info, json_file, threshold, buffer, error_name):
    with open(json_file, 'r') as f:
        preds = json.load(f)['predictions']
    
    formatted_preds = []
    for pred in preds:
        frame = pred['pnr'] * video_info['extracted_frame_interval']
        score = pred['pred']

        if score > threshold:
            interval = [max(0, int(frame - buffer)), min(int(frame + buffer), video_info['video_duration'])]
            formatted_preds.append(interval + [score])
            assert len(formatted_preds[-1]) == 3

    # Merge preds
    if len(formatted_preds) > 1:
        formatted_preds = merge_preds(formatted_preds)
    df = convert_preds_to_df(args, video_info, formatted_preds, error_name)

    return df


def merge_preds(intervals):
    # Step 1: Sort the intervals based on the start frame
    intervals.sort(key=lambda x: x[0])

    merged_intervals = []
    current_start, current_end, current_scores = intervals[0][0], intervals[0][1], [intervals[0][2]]

    for i in range(1, len(intervals)):
        start, end, score = intervals[i]

        # Check if the current interval overlaps with the previous one
        if start <= current_end:  # Overlapping intervals
            current_end = max(current_end, end)  # Extend the interval
            current_scores.append(score)  # Add score to calculate mean later
        else:
            # No overlap, finalize the previous interval
            mean_score = sum(current_scores) / len(current_scores)
            merged_intervals.append([current_start, current_end, mean_score])

            # Start a new interval
            current_start, current_end, current_scores = start, end, [score]

    # Finalize the last interval
    mean_score = sum(current_scores) / len(current_scores)
    merged_intervals.append([current_start, current_end, mean_score])

    return merged_intervals


def convert_preds_to_df(args, video_info, formatted_preds, error_name):
    df = []
    for i, pred in enumerate(formatted_preds):
        clip_id = f'{error_name}_model_pred_{i}'
        start_frame, end_frame, score = pred
        start_time = convert_to_timestamp(start_frame, video_info['fps'])
        end_time = convert_to_timestamp(end_frame, video_info['fps'])

        item = {'clip_id': clip_id, 'start_frame': start_frame, 'end_frame': end_frame, 'duration': f'{start_frame}-{end_frame}', 'start_time': start_time, 'end_time': end_time, 'score': score}

        df.append(item)

    columns = ['clip_id', 'annotation_type', 'label', 'duration', 'fps', 'start_frame', 'end_frame', 'start_time', 'end_time', 'score', 'video_path']
    
    if len(df) > 0:
        df = pd.DataFrame(df)
        df['annotation_type'] = 'errors'
        df['label'] = error_name
        df['fps'] = args.fps
        df['video_path'] = args.video_path

        df = df[columns]
    else:
        df = pd.DataFrame(columns=columns)

    return df


def main(args):
    video_info = get_video_info(args.video_path, args.fps)

    # Load bleeding predictions
    df_bleeding = format_interval_preds(args, video_info, args.bleeding_preds_json, args.bleeding_threshold, args.interval_buffer, 'bleeding')
    
    # Load spillage predictions
    df_spillage = format_interval_preds(args, video_info, args.spillage_preds_json, args.spillage_threshold, args.interval_buffer, 'spillage')

    # Load thermal_injury predictions
    df_thermal_injury = format_frame_preds(args, video_info, args.thermal_injury_preds_json, args.thermal_injury_threshold, args.frame_buffer, 'thermal_injury')

    # Concatenate the predictions
    df = pd.concat([df_bleeding, df_spillage, df_thermal_injury])
    df = df.sort_values(by='start_frame')

    # Adjust the frame numbers and times by the oob json output
    if args.oob_json is not None:
        with open(args.oob_json, 'r') as f:
            formatted_oob_results = json.load(f)
        start_frame = formatted_oob_results['start'] # this is at the full fps
        full_fps = formatted_oob_results['fps']

        if start_frame > 0:
            df['start_frame'] = df.apply(lambda x: x['start_frame'] + start_frame, axis=1)
            df['end_frame'] = df.apply(lambda x: x['end_frame'] + start_frame, axis=1)
            df['start_time'] = df.apply(lambda x: convert_to_timestamp(x['start_frame'], full_fps), axis=1)
            df['end_time'] = df.apply(lambda x: convert_to_timestamp(x['end_frame'], full_fps), axis=1)
            df['duration'] = df.apply(lambda x: f"{x['start_frame']}-{x['end_frame']}", axis=1)
        
    parent = os.path.split(args.postprocessed_csv)[0]
    os.makedirs(parent, exist_ok=True)
    df.to_csv(args.postprocessed_csv, index=False)
    print(f'Dumped to {args.postprocessed_csv}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--postprocessed_csv', type=str)
    parser.add_argument('--bleeding_preds_json', type=str)
    parser.add_argument('--spillage_preds_json', type=str)
    parser.add_argument('--thermal_injury_preds_json', type=str)

    parser.add_argument('--bleeding_threshold', type=float, default=0.4)
    parser.add_argument('--spillage_threshold', type=float, default=0.4)
    parser.add_argument('--thermal_injury_threshold', type=float, default=0.4)

    parser.add_argument('--interval_buffer', type=int, default=50)
    parser.add_argument('--frame_buffer', type=int, default=50)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--video_path', type=str)
    parser.add_argument('--oob_json', type=str, default=None)

    args = parser.parse_args()
    main(args)