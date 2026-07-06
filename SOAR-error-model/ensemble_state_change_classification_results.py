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
            anns = json.load(f)
            pnr2video = anns['pnr2video']
            
            for pnr, video_name in pnr2video.items():
                formatted_annotations[(ann_file, video_name)] = pnr

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
            pnr = formatted_annotations[(ann_file, video_name)]

            if pnr not in formatted_preds:
                formatted_preds[pnr] = []
            formatted_preds[pnr].append(pred['pred'])

    return formatted_preds
    

def ensemble_predictions(formatted_preds, pool, topk):
    ensembled_preds = []
    for pnr, prob_lst in formatted_preds.items():

        ensembled_prob = np.array(prob_lst)
        if topk is not None:
            ensembled_prob = np.sort(ensembled_prob)[::-1][:topk]
        if pool == 'mean':
            ensembled_prob = ensembled_prob.mean().item()
        elif pool == 'max':
            ensembled_prob = ensembled_prob.max().item()
        elif pool == 'max_in_at_least_2':
            idx = np.where(ensembled_prob > args.threshold)[0]
            ensembled_prob = ensembled_prob[idx].mean().item() if len(idx) >= 2 else 0.
        else:
            raise ValueError
        ensembled_preds.append({'pnr': int(pnr), 'pred': ensembled_prob})
    
    return ensembled_preds


def get_pnr_pos_frames(ensembled_preds, with_threshold=True):
    pnr_pos, scores = [], []
    for item in ensembled_preds:
        pnr = item['pnr']
        pred = item['pred']

        if with_threshold:
            if pred > args.threshold:
                pnr_pos.append(pnr)
                scores.append(pred)
        else:
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



def main(args):
    if args.ensembled_dir is None:
        args.ensembled_dir = args.results_dir
    os.makedirs(args.ensembled_dir, exist_ok=True)

    if args.sampled_fps is not None:
        ensembled_pred_file = os.path.join(args.ensembled_dir, f'ensembled_preds_{args.sampled_fps}fps.json')
    else:
        ensembled_pred_file = os.path.join(args.ensembled_dir, 'ensembled_preds.json')

    if os.path.exists(ensembled_pred_file):
        os.remove(ensembled_pred_file)

    # get annotations in {(ann_file, video_name): pnr}
    formatted_annotations = format_annotations(args)

    # get predictions in {pnr: [probs]}
    formatted_preds = format_predictions(args, formatted_annotations)

    # {pnr: ensembled_pred}
    ensembled_preds = ensemble_predictions(formatted_preds, args.pool, args.topk)

    with open(ensembled_pred_file, 'w') as f:
        json.dump({'predictions': ensembled_preds}, f)
    
    print(f"Dumped ensembled predictions: {ensembled_pred_file}")

    pnr_pos_frames, scores = get_pnr_pos_frames(ensembled_preds, with_threshold=not args.smooth_predictions)

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
    print(sorted(zip(pnr_pos_frames, scores)))


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
    parser.add_argument('--ensembled_dir', type=str, default=None)
    parser.add_argument('--topk', type=int, default=None)
    args = parser.parse_args()

    main(args)