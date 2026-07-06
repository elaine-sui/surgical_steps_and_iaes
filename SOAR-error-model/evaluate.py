import argparse
import pandas as pd
import json
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt

"""
Precision/recall:
Hit: GT within predicted interval
Miss: otherwise
"""

def plot_pr_curve(avg_metrics, plot_path):
    precision_values = [avg_metrics[t]['precision_interval_avg'] for t in avg_metrics][::-1]
    recall_values = [avg_metrics[t]['recall_interval_avg'] for t in avg_metrics][::-1]

    plt.figure()
    plt.plot(recall_values, precision_values)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Plot dumped to {plot_path}")
    plt.close()

def keep_preds_between_steps2and5(df_preds, clip_csv_step, duration_frames):
    df_step = pd.read_csv(clip_csv_step)
    min_frame = 0

    if '2ExposeGallbladder' in df_step['label'].values:
        df_step2 = df_step[df_step['label'] == '2ExposeGallbladder']
        start = df_step2.apply(lambda x: int(x['duration'].split('-')[0]), axis=1).values
        min_frame = start.min().item()
    
    max_frame = duration_frames
    if '5GallbladderDissection' in df_step['label'].values:
        df_step5 = df_step[df_step['label'] == '5GallbladderDissection']
        end = df_step5.apply(lambda x: int(x['duration'].split('-')[1]), axis=1).values
        max_frame = end.max().item()

    # Restrict
    restricted_ensembled_preds = []
    for i, row in df_preds.iterrows():
        if row['start_frame'] >= min_frame and row['end_frame'] <= max_frame:
            restricted_ensembled_preds.append(row)
    
    return pd.DataFrame(restricted_ensembled_preds)


def merge_predictions(pnr_pos, scores):
    # Merge predictions if they are less than 1 second apart? (10 frames)
    idx = np.argsort(pnr_pos)
    pnr_pos = [pnr_pos[i] for i in idx]
    scores = [scores[i] for i in idx]

    merged_pnr_pos, merged_scores = [], []
    _merged_pnr_pos = [pnr_pos[0]]
    _merged_scores = [scores[0]]

    for frame_num, score in zip(pnr_pos[1:], scores[1:]):
        if frame_num - _merged_pnr_pos[0] < 5:
            _merged_pnr_pos.append(frame_num)
            _merged_scores.append(score)
        else:
            merged_pnr_pos.append(int(np.mean(_merged_pnr_pos)))
            merged_scores.append(int(np.mean(_merged_scores)))
            _merged_pnr_pos = [frame_num]
            _merged_scores = [score]

    merged_pnr_pos.append(int(np.mean(_merged_pnr_pos)))
    merged_scores.append(int(np.mean(_merged_scores)))
    
    return merged_pnr_pos, merged_scores


def get_gt_intervals(args, video_id, df_master):
    clip_csv_error = df_master.loc[video_id, args.gt_col]

    df_error = pd.read_csv(clip_csv_error)

    if len(df_error) > 0:
        df_error = df_error[df_error['label'].isin(args.label_lst)]

        if len(df_error) > 0:
            intervals = df_error['duration'].values.tolist()
            intervals = [interval.split('-') for interval in intervals]
            intervals = [[int(i) for i in interval] for interval in intervals]
            return intervals
    return []


def is_loose_hit(pred_interval, gt_intervals):
    """
    Hit: if there is any GT that is within pred_interval
    """
    for i, interval in enumerate(gt_intervals):
        start, end = interval
        if start >= pred_interval[0] and start <= pred_interval[1]:
            return True, i
    
    return False, None


def compute_precision(pred_intervals, gt_intervals):
    if len(pred_intervals) == 0:
        return None, 0, 0
    hits = [is_loose_hit(pred_interval, gt_intervals)[0] for pred_interval in pred_intervals]
    precision = sum(hits) / len(pred_intervals)

    assert precision >= 0 and precision <= 1, f'{hits} / {gt_intervals}'

    return precision, sum(hits), len(pred_intervals)
    

def compute_recall(pred_intervals, gt_intervals):
    if len(gt_intervals) == 0:
        return None, 0, 0
    gt_hits = [is_loose_hit(pred_interval, gt_intervals)[1] for pred_interval in pred_intervals]
    unique_gt_hits = [hit_idx for hit_idx in gt_hits if hit_idx is not None]
    unique_gt_hits = set(unique_gt_hits)
    recall = len(unique_gt_hits) / len(gt_intervals)

    assert recall >= 0 and recall <= 1, f'{unique_gt_hits} / {gt_intervals}'
    
    return recall, len(unique_gt_hits), len(gt_intervals)


def compute_metrics_for_video(args, pred_file, df_master):
    video_id = pred_file.split('/')[-1].split('.')[0]
    gt_intervals = get_gt_intervals(args, video_id, df_master)

    if gt_intervals is None:
        return None

    df_pred = pd.read_csv(pred_file)
    df_pred = df_pred[df_pred['label'] == args.pred_label]

    if args.between_steps2and5:
        df_pred = keep_preds_between_steps2and5(df_pred, df_master.loc[video_id, 'step_csv'], df_master.loc[video_id, 'duration_frames'])

    # different thresholds
    # report precision and recall at different absolute distances to the GT state change
    metrics = {}

    for threshold in args.thresholds:

        if len(df_pred) > 0:
            df_pred_ = df_pred[df_pred['score'] >= threshold]
            pred_start = df_pred_['start_frame'].values.tolist()
            pred_end = df_pred_['end_frame'].values.tolist()
            pred_intervals = list(zip(pred_start, pred_end))

            scores = df_pred_['score'].values.tolist()
        else:
            pred_intervals, scores = [], []

        precision, num_pred_hits, num_pred = compute_precision(pred_intervals, gt_intervals)
        recall, num_gt_hits, num_gt = compute_recall(pred_intervals, gt_intervals)

        metrics[threshold] = {'precision': precision, 'recall': recall, 'pred': pred_intervals, 'scores': scores, 'gt': gt_intervals, 'num_pred_hits': num_pred_hits, 'num_pred_total': num_pred, 'num_gt_hits': num_gt_hits, 'num_gt_total': num_gt}
    
    return metrics


def main(args):
    all_pred_files = glob(f'{args.pred_dir}/*.csv')

    df_split = pd.read_csv(args.datasplit_csv).set_index('video_id')
    df_split = df_split[df_split['error_subset'] == args.split]
    df_split = df_split.dropna(subset=[args.gt_col])

    if args.dataset_only:
        print(f"Restrict to {args.dataset}")
        args.dataset = args.dataset.split(',')
        df_split = df_split[df_split['dataset'].isin(args.dataset)]

    split_video_ids = df_split.index.values.tolist()

    args.label_lst = args.label_lst.split(',')

    if args.thresholds != "all":
        args.thresholds = args.thresholds.split(',')
        args.thresholds = [float(t) for t in args.thresholds]
    else:
        args.thresholds = np.linspace(args.min_threshold, 1.0, 100)
    
    all_metrics_dict = {'average': None, 'individual': {}}

    for pred_file in all_pred_files:
        video_id = pred_file.split('/')[-1].split('.')[0]
        if video_id not in split_video_ids:
            continue
        metrics_dict = compute_metrics_for_video(args, pred_file, df_split)

        if metrics_dict is not None:
            all_metrics_dict['individual'][video_id] = metrics_dict

    print(f"Number of videos being evaluated: {len(all_metrics_dict['individual'])}")

    all_metrics_dict['average'] = {}
    for t in args.thresholds:
        # print(f"Threshold: {t}")
        totals_dict = {'num_pred_hits': 0, 'num_pred_total': 0, 'num_gt_hits': 0, 'num_gt_total': 0}
        precision_lst = [all_metrics_dict['individual'][video_id][t]['precision'] for video_id in all_metrics_dict['individual']]
        recall_lst = [all_metrics_dict['individual'][video_id][t]['recall'] for video_id in all_metrics_dict['individual']]

        for video_id in all_metrics_dict['individual']:
            for key in totals_dict:
                totals_dict[key] += all_metrics_dict['individual'][video_id][t][key]

        if totals_dict['num_pred_total'] == 0:
            totals_dict['precision'] = None
        else:
            totals_dict['precision'] = totals_dict['num_pred_hits'] / totals_dict['num_pred_total']
            assert totals_dict['precision'] >= 0 and totals_dict['precision'] <= 1

        if totals_dict['num_gt_total'] == 0:
            totals_dict['recall'] = None
        else:
            totals_dict['recall'] = totals_dict['num_gt_hits'] / totals_dict['num_gt_total']
            assert totals_dict['recall'] >= 0 and totals_dict['recall'] <= 1

        no_pred = [p is None for i, p in enumerate(precision_lst)]
        no_gt = [r is None for i, r in enumerate(recall_lst)]

        acc_no_event = (np.array(no_pred) == np.array(no_gt)).sum() / len(no_pred)

        precision_lst = [p for p in precision_lst if p is not None]
        recall_lst = [r for r in recall_lst if r is not None]

        mean_precision, mean_recall = None, None
        if len(precision_lst) > 0:
            mean_precision = sum(precision_lst) / len(precision_lst)

        if len(recall_lst) > 0:
            mean_recall = sum(recall_lst) / len(recall_lst)

        # print(f"Precision: {mean_precision:0.4f} \t Recall: {mean_recall:.04f}")
        all_metrics_dict['average'][t] = {'precision_vid_avg': mean_precision, 'recall_vid_avg': mean_recall, 'precision_interval_avg': totals_dict['precision'], 'recall_interval_avg': totals_dict['recall'], 'accuracy_of_no_event': acc_no_event, 'num_gt_no_event': sum(no_gt)}
        
        if totals_dict['precision'] is not None and totals_dict['recall'] is not None and (totals_dict['precision'] + totals_dict['recall']) > 0:
            f1_interval_avg = 2 * (totals_dict['precision'] * totals_dict['recall']) / (totals_dict['precision'] + totals_dict['recall'])
            f2_interval_avg = ((1+4) * totals_dict['precision'] * totals_dict['recall']) / (4 * totals_dict['precision'] + totals_dict['recall'])
        else:
            f1_interval_avg = None
            f2_interval_avg = None
        all_metrics_dict['average'][t]['f1_interval_avg'] = f1_interval_avg
        all_metrics_dict['average'][t]['f2_interval_avg'] = f2_interval_avg

        if isinstance(mean_precision, float):
            assert mean_precision >= 0 and mean_precision <= 1
        
        if isinstance(mean_recall, float):
            assert mean_recall >= 0 and mean_recall <= 1

    args.metrics_dir = os.path.join(args.metrics_dir, args.pred_label)
    os.makedirs(args.metrics_dir, exist_ok=True)
    if args.dataset_only:
        save_path = os.path.join(args.metrics_dir, args.split + f"{'_'.join(args.dataset)}.json")
    else:
        save_path = os.path.join(args.metrics_dir, args.split + ".json")
    
    if args.easy_only:
        save_path = save_path.split('.json')[0] + "_easy_only.json"
    with open(save_path, 'w') as f:
        json.dump(all_metrics_dict, f, indent=4)
    
    print(f"Saved metrics to {save_path}")

    # Plot PR curve
    if args.plot_pr_curve:
        plot_path = save_path.replace('.json', '_pr_curve.png')
        plot_pr_curve(all_metrics_dict['average'], plot_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, default='postprocessed_predictions')
    parser.add_argument('--pred_label', type=str, default='thermal_injury')
    parser.add_argument('--label_lst', type=str, default="2.2InjuryToNontargetStructureOrTissue_Burn,3.2InjuryToNontargetStructureOrTissue_Burn,5.2InjuryToNontargetStructureOrTissue_Burn")
    parser.add_argument('--thresholds', type=str, default='0.4,0.45,0.5,0.6,0.7,0.8,0.9')
    parser.add_argument('--between_steps2and5', action='store_true')
    parser.add_argument('--datasplit_csv', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--gt_col', type=str, default=None)

    parser.add_argument('--metrics_dir', type=str, default='metrics')

    parser.add_argument('--dataset_only', action='store_true')
    parser.add_argument('--dataset', type=str, default=None)

    parser.add_argument('--plot_pr_curve', action='store_true')
    parser.add_argument('--min_threshold', type=float, default=0.40)

    args = parser.parse_args()
    main(args)