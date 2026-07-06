import argparse
import os
import json

from tqdm import tqdm
import pandas as pd
import itertools

def interval_extract(detection_preds):
    # Only get consecutive frame intervals (because OOB might cut it)
    frame_nums = [int(k) for k in detection_preds]

    # code from https://www.geeksforgeeks.org/python-make-a-list-of-intervals-with-sequential-numbers/
    iterable = sorted(set(frame_nums))
    for key, group in itertools.groupby(enumerate(iterable),
    lambda t: t[1] - t[0]):
        group = list(group)
        yield [group[0][1], group[-1][1]]


def get_clips_in_coco_format(args, detection_preds):
    frame_nums = [int(k) for k in detection_preds]
    total_frames = max(frame_nums)
    fps = args.original_fps

    sample_rate = int(args.original_fps // args.sampled_fps)
    clip_len_frames_total = int(args.clip_len_frames * sample_rate)
    clip_len_half = clip_len_frames_total // 2

    coco_anns = []
    start_clip_frame = None
    pnr = None
    pnr_frame2video_name = {}
    detected_segments = []
    detected_seg = []

    # Sort detection preds
    intervals = list(interval_extract(detection_preds))

    for interval in tqdm(intervals, total=len(intervals)):
        detection_preds_interval = {}
        for frame_num in range(interval[0], interval[1] + 1):
            detection_preds_interval[str(frame_num)] = detection_preds[str(frame_num)]

        num_frames = len(detection_preds_interval)
        for i, (frame_num, frame_preds) in tqdm(enumerate(detection_preds_interval.items()), total=num_frames):
            frame_num = int(frame_num)
            if frame_preds is not None and isinstance(frame_preds['scores'], float):
                if frame_preds['scores'] < args.threshold:
                    frame_preds['scores'] = None
            elif frame_preds is not None: # list of scores
                idx_high_conf = [idx for idx, score in enumerate(frame_preds['scores']) if score > args.threshold]
                if len(idx_high_conf) == 0:
                    frame_preds['scores'] = None

            # Get first frame that has bbox detections
            if start_clip_frame is None:
                if frame_preds['scores'] is not None: # detection after null
                    start_clip_frame = frame_num
                    pnr = frame_num
                    detected_seg.append(pnr)
                else: # still null
                    continue
            else:
                if frame_preds['scores'] is not None: # same detection
                    continue
                else: # becomes null
                    start_clip_frame = None
                    detected_seg.append(frame_num)
                    detected_segments.append(detected_seg)
                    detected_seg = []
                    continue

            if pnr < clip_len_half:
                start = 0
                end = start + clip_len_frames_total
            elif (total_frames - pnr) < clip_len_half:
                end = total_frames
                start = end - clip_len_frames_total
            else:
                start = pnr - clip_len_half
                end = start + clip_len_frames_total           

            ann = {
                'parent_start_sec': start / fps,
                'parent_end_sec': end / fps,
                'parent_start_frame': start,
                'parent_end_frame': end,
                'video_uid': args.video_id,
                'unique_id': args.video_id + f"_pred{i}",
                'state_change': None,
                'fps': fps,
                'pnr_frame': pnr
            }

            clip_id = ann['unique_id']
            video_id = pnr

            assert start >= 0 and end <= total_frames + 1

            clip_frame_nums = list(range(start, end, sample_rate))

            if pnr >= max(clip_frame_nums) or pnr <= start:
                continue # no actual state change!

            assert pnr > start and pnr < max(clip_frame_nums), f'{start}, {pnr}, {max(clip_frame_nums)}, {total_frames}'
            assert len(clip_frame_nums) == args.clip_len_frames

            pnr_frame2video_name[pnr] = video_id

            # Convert to COCO format
            for clip_frame_num in clip_frame_nums:
                path = os.path.join(args.frames_dir, args.video_id, f'{clip_frame_num}.jpeg')
                coco_item = {'id': f'{clip_id}_{clip_frame_num}', 'file_name': path, 'video_id': video_id, "ds": None, 'video_name': clip_id}
                coco_anns.append(coco_item)

    return coco_anns, pnr_frame2video_name, detected_segments


def get_sliding_window_throughout_detected_segments(args, detected_segments, total_frames, detected_pnr_frame2video_name):
    fps = args.original_fps

    sample_rate = int(args.original_fps // args.sampled_fps)
    clip_len_frames_total = int(args.clip_len_frames * sample_rate)
    clip_len_half = clip_len_frames_total // 2

    pnr_frame2video_name = {}
    coco_anns = []

    for seg in detected_segments:
        if seg[1] - seg[0] < clip_len_half: # skip the segments that are too small
            continue

        # print(seg)

        sliding_window_starts = list(range(seg[0]+clip_len_half, seg[1], clip_len_half))
        for window_start in sliding_window_starts:
            pnr = window_start

            if pnr in detected_pnr_frame2video_name.keys():
                continue

            if pnr < clip_len_half:
                start = 0
                end = start + clip_len_frames_total
            elif (total_frames - pnr) < clip_len_half:
                end = total_frames
                start = end - clip_len_frames_total
            else:
                start = pnr - clip_len_half
                end = start + clip_len_frames_total           

            ann = {
                'parent_start_sec': start / fps,
                'parent_end_sec': end / fps,
                'parent_start_frame': start,
                'parent_end_frame': end,
                'video_uid': args.video_id,
                'unique_id': args.video_id + f"_pred{pnr}",
                'state_change': None,
                'fps': fps,
                'pnr_frame': pnr
            }

            clip_id = ann['unique_id']
            video_id = pnr

            assert start >= 0 and end <= total_frames + 1

            clip_frame_nums = list(range(start, end, sample_rate))

            if pnr >= max(clip_frame_nums) or pnr <= start:
                continue # no actual state change!

            assert pnr > start and pnr < max(clip_frame_nums), f'{start}, {pnr}, {max(clip_frame_nums)}, {total_frames}'
            assert len(clip_frame_nums) == args.clip_len_frames

            pnr_frame2video_name[pnr] = video_id

            # Convert to COCO format
            for clip_frame_num in clip_frame_nums:
                path = os.path.join(args.frames_dir, args.video_id, f'{clip_frame_num}.jpeg')
                coco_item = {'id': f'{clip_id}_{clip_frame_num}', 'file_name': path, 'video_id': video_id, "ds": None, 'video_name': clip_id}
                coco_anns.append(coco_item)

    return coco_anns, pnr_frame2video_name


def merge_segments(detected_segments):
    new_segments = []
    prev_seg = None
    for seg in detected_segments:
        if prev_seg is None:
            prev_seg = seg
        else:
            if seg[0] - prev_seg[1] < 10:
                prev_seg[1] = seg[1]
            else:
                new_segments.append(prev_seg)
                prev_seg = seg
    return new_segments


def main(args):

    with open(args.detection_json, 'r') as f:
        detection_preds = json.load(f)
    if 'annotations' in detection_preds:
        detection_preds = detection_preds['annotations']

    clips, pnr_frame2video_name, detected_segments = get_clips_in_coco_format(args, detection_preds)
    # print(detected_segments)

    args.output_dir = os.path.join(args.output_dir, args.video_id)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.save_detected_segments:
        detected_segments = merge_segments(detected_segments)
        df_detected_segments = pd.DataFrame({'start': [seg[0] for seg in detected_segments], 'end': [seg[1] for seg in detected_segments]})
        out_path = os.path.join(args.output_dir, 'detected_segments.csv')
        df_detected_segments.to_csv(out_path, index=False)
        print(f"Saved detected segs to {out_path}")

    if args.add_sliding_window_throughout_detected_segments:
        frame_nums = [int(k) for k in detection_preds]
        total_frames = max(frame_nums)
        clips2, pnr_frame2video_name2 = get_sliding_window_throughout_detected_segments(args, detected_segments, total_frames, pnr_frame2video_name)
        # print(f"Added frames: {len(clips2)}")
        clips = clips + clips2
        pnr_frame2video_name.update(pnr_frame2video_name2)

    output_path = os.path.join(args.output_dir, f'sampled_{args.sampled_fps}fps.json')
    with open(output_path, 'w') as f:
        json.dump({'images': clips, 'pnr2video': pnr_frame2video_name}, f)
    
    print(f"Total clips: {len(pnr_frame2video_name)}")
    
    print(f'COCO formatted annotations at {output_path}')
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_json', type=str, default=None)
    parser.add_argument('--original_fps', type=int, default=10)
    parser.add_argument('--sampled_fps', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--clip_len_frames', type=int, default=16)
    parser.add_argument('--frames_dir', type=str)
    parser.add_argument('--video_id', type=str, default=None)
    parser.add_argument('--threshold', type=float, default=0.5)

    parser.add_argument('--add_sliding_window_throughout_detected_segments', action='store_true')
    parser.add_argument('--save_detected_segments', action='store_true')

    args = parser.parse_args()

    main(args)