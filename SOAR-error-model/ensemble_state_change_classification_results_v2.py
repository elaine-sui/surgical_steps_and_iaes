import argparse
import os
import json
from glob import glob
import numpy as np
import pandas as pd

def format_annotations(args):
    ann_files = glob(f"{args.annotations_dir}/*.json")
    formatted_annotations = {}

    for ann_file in ann_files:
        with open(ann_file, 'r') as f:
            anns = json.load(f)['images']

            vid_id2frame_num = {}
            for ann in anns:
                vid_id = ann['video_id']
                frame_num = int(ann['file_name'].split('/')[-1].replace('.jpeg', ''))
                if vid_id not in vid_id2frame_num:
                    vid_id2frame_num[vid_id] = []
                vid_id2frame_num[vid_id].append(frame_num)
            
            for vid_id, frame_lst in vid_id2frame_num.items():
                frame_lst = sorted(frame_lst)
                min_frame, max_frame = frame_lst[0], frame_lst[-1]
                # mid_frame = int((min_frame + max_frame)/2)
                formatted_annotations[(ann_file, vid_id)] = (min_frame, max_frame)

    return formatted_annotations


def format_predictions(args, formatted_annotations):
    pred_files = glob(f'{args.results_dir}/*.json')

    if args.sampled_fps is not None:
        num_frames = int(16 * (10 / args.sampled_fps))
        pred_files = [f for f in pred_files if f'_{num_frames}_' in f]

    formatted_preds = {}

    for pred_file in pred_files:
        if 'ensembled' in pred_file:
            continue
        
        with open(pred_file, 'r') as f:
            preds = json.load(f)
        
        predictions = preds['predictions']
        ann_file = preds['annotation_file']

        if args.prefix:
            ann_file = f'{args.prefix}/{ann_file}'

        for pred in predictions:
            video_name = pred['id']
            clip_start_end = formatted_annotations[(ann_file, video_name)]

            if clip_start_end not in formatted_preds:
                formatted_preds[clip_start_end] = []
            formatted_preds[clip_start_end].append(pred['pred'])

    return formatted_preds


def intersect_predictions_v2(formatted_preds):
    candidates = [(k[0], k[1], v) for k,v in formatted_preds.items()]
    # Sort intervals by their start times
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    # List to store the resulting intersected intervals and their mean scores
    result = []
    
    # Iterate through each candidate interval to find intervals that it intersects with
    for i in range(len(candidates)):
        current_start, current_end, current_scores = candidates[i]

        intervals_in_intersection = []
        for j in range(i + 1, len(candidates)):
            next_start, next_end, next_score = candidates[j]

            if next_start <= current_end:
                intervals_in_intersection.append(candidates[j])
            else:
                break
        
        # intersection_start = max([current_start] + [interval[0] for interval in intervals_in_intersection])
        # intersection_end = min([current_end] + [interval[1] for interval in intervals_in_intersection])
        intersection_scores = []
        for interval in intervals_in_intersection:
            intersection_scores.extend(interval[2])
        result.append((current_start, current_end, np.mean(current_scores + intersection_scores)))
    
    formatted_preds = {}
    for item in result:
        formatted_preds[(item[0], item[1])] = [item[2].item()]
    return formatted_preds
    


def intersect_predictions(formatted_preds):   
    candidates = [(k[0], k[1], v) for k,v in formatted_preds.items()]
    # Sort intervals by their start times
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    # List to store the resulting intersected intervals and their mean scores
    result = []
    
    # Iterate through each candidate interval
    for i in range(len(candidates)):
        current_start, current_end, current_scores = candidates[i]
        
        # Start with the current interval as the base intersection
        intersection_start = current_start
        intersection_end = current_end
        scores_in_intersection = current_scores
        
        # Check for intersections with the following intervals
        for j in range(i + 1, len(candidates)):
            next_start, next_end, next_score = candidates[j]
            
            # If the intervals overlap, compute the intersection
            if next_start <= intersection_end:
                # The intersection of two intervals [a,b] and [c,d] is [max(a,c), min(b,d)]
                intersection_start = max(intersection_start, next_start)
                intersection_end = min(intersection_end, next_end)
                scores_in_intersection.extend(next_score)
            else:
                break
        
        # If there's a valid intersection (the end time is after the start time)
        if intersection_start < intersection_end:
            # Compute the mean score for the intersected interval
            mean_score = np.mean(scores_in_intersection)
            result.append((intersection_start, intersection_end, mean_score))
    
    formatted_preds = {}
    for item in result:
        formatted_preds[(item[0], item[1])] = item[2].item()
    return formatted_preds


def union_predictions(formatted_preds):   
    candidates = [(k[0], k[1], v) for k,v in formatted_preds.items()]

    # Sort intervals by their start times
    candidates.sort(key=lambda x: x[0])
    
    # List to store the resulting union intervals and their mean scores
    result = []
    
    # Start with the first interval
    current_start, current_end, current_scores = candidates[0]
    scores_in_union = current_scores
    
    for i in range(1, len(candidates)):
        next_start, next_end, next_score = candidates[i]
        
        # If the intervals overlap or are adjacent, merge them
        if next_start <= current_end:
            # The union of two intervals [a, b] and [c, d] is [min(a, c), max(b, d)]
            current_end = max(current_end, next_end)
            scores_in_union.extend(next_score)
        else:
            # Otherwise, the current interval is complete; add it to the result
            mean_score = np.mean(scores_in_union)
            result.append((current_start, current_end, mean_score))
            
            # Move to the next interval
            current_start, current_end, current_scores = next_start, next_end, next_score
            scores_in_union = current_scores
    
    # Add the last merged interval
    mean_score = np.mean(scores_in_union)
    result.append((current_start, current_end, mean_score))

    formatted_preds = {}
    for item in result:
        formatted_preds[(item[0], item[1])] = item[2].item()
    
    return formatted_preds
    

def ensemble_predictions(formatted_preds, pool):
    ensembled_preds = []

    # Compute the per-frame probability
    frame_num2prob = {}
    for clip_start_end, prob_lst in formatted_preds.items():
        for frame_num in range(clip_start_end[0], clip_start_end[1]+1):
            if frame_num not in frame_num2prob:
                frame_num2prob[frame_num] = []
            frame_num2prob[frame_num].extend(prob_lst)
    
    for frame_num, prob_lst in frame_num2prob.items():
        ensembled_prob = np.array(prob_lst)
        if pool == 'mean':
            ensembled_prob = ensembled_prob.mean().item()
        elif pool == 'max':
            ensembled_prob = ensembled_prob.max().item()
        elif pool == 'max_in_at_least_2':
            idx = np.where(ensembled_prob > args.threshold)[0]
            ensembled_prob = ensembled_prob[idx].mean().item() if len(idx) >= 2 else 0.
        else:
            raise ValueError
        ensembled_preds.append({'pnr': int(frame_num), 'pred': ensembled_prob})
    
    return ensembled_preds


def get_pnr_pos_frames(ensembled_preds):
    pnr_pos, scores = [], []
    for item in ensembled_preds:
        pnr = item['pnr']
        pred = item['pred']

        if pred > args.threshold:
            pnr_pos.append(pnr)
            scores.append(pred)
    
    return pnr_pos, scores


def smooth_predictions(args, frame_numbers, scores, smoothing_fn=np.mean):
    # Sort frame numbers to ensure they are in ascending order
    idx = np.argsort(frame_numbers)
    frame_numbers = [frame_numbers[i] for i in idx]
    scores = [scores[i] for i in idx]
    # frame_numbers.sort()
    result = []
    result_scores = []
    temp_group = [frame_numbers[0]]
    temp_scores = [scores[0]]
    
    # Iterate through the frame numbers to group close frames
    for i in range(1, len(frame_numbers)):
        if frame_numbers[i] - temp_group[-1] <= args.smoothing_window:
            # If within the threshold, add to the temporary group
            temp_group.append(frame_numbers[i])
            temp_scores.append(scores[i])
        else:
            # Calculate the mean of the current group and reset
            result.append(int(np.mean(temp_group)))
            result_scores.append(smoothing_fn(temp_scores))
            temp_group = [frame_numbers[i]]
            temp_scores = [scores[i]]
    
    # Add the last group to the result
    if temp_group:
        result.append(int(np.mean(temp_group)))
        result_scores.append(smoothing_fn(temp_scores))
    
    return result, result_scores


def intervals_are_disjoint(formatted_preds):
    intervals = list(formatted_preds.keys())
    intervals.sort(key=lambda x: x[0])

    # Check for overlap in consecutive intervals
    for i in range(1, len(intervals)):
        # If the start of the current interval is less than or equal to the end of the previous one, they overlap
        if intervals[i][0] <= intervals[i - 1][1]:
            return False

    return True


def main(args):
    if args.sampled_fps is not None:
        ensembled_pred_file = os.path.join(args.results_dir, f'ensembled_preds_{args.sampled_fps}fps.json')
    elif args.do_intersect:
        ensembled_pred_file = os.path.join(args.results_dir, f'ensembled_preds_intersection.json')
    elif args.do_union:
        ensembled_pred_file = os.path.join(args.results_dir, f'ensembled_preds_union.json')
    else:
        ensembled_pred_file = os.path.join(args.results_dir, 'ensembled_preds.json')

    if os.path.exists(ensembled_pred_file):
        os.remove(ensembled_pred_file)

    # get annotations in {(ann_file, video_name): pnr}
    formatted_annotations = format_annotations(args)

    # get predictions in {pnr: [probs]}
    formatted_preds = format_predictions(args, formatted_annotations)

    if args.do_intersect or args.do_union:

        if args.do_intersect:
            ensembled_preds = intersect_predictions(formatted_preds)
        else:
            ensembled_preds = union_predictions(formatted_preds)
        pred_intervals, scores = [], []

        for interval, score in ensembled_preds.items():
            if score > args.threshold:
                pred_intervals.append([interval, score])
        
        print("Predicted positive state change intervals:")
        print(sorted(pred_intervals))
    else:
        # {pnr: ensembled_pred}
        ensembled_preds = ensemble_predictions(formatted_preds, args.pool)
        pnr_pos_frames, scores = get_pnr_pos_frames(ensembled_preds)

        if args.take_top_pred_only and len(scores) > 0:
            idx = np.argmax(scores)
            pnr_pos_frames = [pnr_pos_frames[idx]]
        
        if args.smooth_predictions:
            pnr_pos_frames, scores = smooth_predictions(args, pnr_pos_frames, scores)

            smoothed_pnr_pos_frames, smoothed_scores = [], []
            for frame, score in zip(pnr_pos_frames, scores):
                if score > args.threshold:
                    smoothed_pnr_pos_frames.append(frame)
                    smoothed_scores.append(score)
            
            pnr_pos_frames, scores = smoothed_pnr_pos_frames, smoothed_scores
        
        print("Predicted positive PNR frames:")
        print(sorted(pnr_pos_frames))

    ensembled_preds = {f'{k[0]}_{k[1]}':v for k,v in ensembled_preds.items()}
    with open(ensembled_pred_file, 'w') as f:
        json.dump({'predictions': ensembled_preds}, f)
    
    print(f"Dumped ensembled predictions: {ensembled_pred_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--annotations_dir', type=str, default=None)
    parser.add_argument('--video_id', type=str, default=None)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--pool', type=str, default='mean')
    parser.add_argument('--take_top_pred_only', action='store_true')
    parser.add_argument('--prefix', type=str, default=None, help='this is for post-hoc ensembling only')
    parser.add_argument('--sampled_fps', type=int, default=None)
    parser.add_argument('--smooth_predictions', action='store_true')
    parser.add_argument('--smoothing_window', type=int, default=10)
    parser.add_argument('--do_intersect', action='store_true')
    parser.add_argument('--do_union', action='store_true')
    args = parser.parse_args()

    main(args)