import numpy as np
import os
from yacs.config import CfgNode
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import cv2
import json

from .plot_utils import generate_distinct_colors, summarize_list, generate_image_for_segmentation, plot_single_video_segmentation

def intervals_extract(iterable):
    # code from https://www.geeksforgeeks.org/python-make-a-list-of-intervals-with-sequential-numbers/
    iterable = sorted(set(iterable))
    for key, group in itertools.groupby(enumerate(iterable),
    lambda t: t[1] - t[0]):
        group = list(group)
        yield [group[0][1], group[-1][1]]


def convert_framenum_to_time(frame_num: int, fps: int):
    seconds = frame_num / fps
    hours = int(seconds / 60**2)
    seconds -= hours * 60**2
    minutes = int(seconds / 60)
    seconds -= minutes * 60
    seconds = int(seconds)

    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def transform_array_to_df(pred: np.array, action_to_idx: dict, fps: int, active_surgery_pred_path: str):
    # pred: [bs]

    # Shift times depending on start and end
    start = 0
    if active_surgery_pred_path is not None:
        df_active = pd.read_csv(active_surgery_pred_path)
        start_row_idx = df_active[df_active['label'] == 'activeSurgery'].index.min()
        start_row = df_active.loc[start_row_idx]
        start = int(start_row['duration'].split('-')[0])

    data = []
    
    for action, idx in action_to_idx.items():
        # get frame num with idx
        indices = np.where(pred == idx)[0]
        intervals = intervals_extract(indices)

        for interval in intervals:
            interval = [val + start for val in interval]
            duration = f'{interval[0]}-{interval[1] + 1}'
            start_time = convert_framenum_to_time(interval[0], fps)
            end_time = convert_framenum_to_time(interval[1] + 1, fps)
            data.append({'label': action, 'duration': duration, 'start_time': start_time, 'end_time': end_time})
        
    df = pd.DataFrame(data)
    return df


def generate_legend(pred_lbl: list, idx_to_action: dict, colors: list, base_path: str):
    max_prediction = max(pred_lbl)

    color_to_label = {colors[max_prediction - i]:idx_to_action[max_prediction - i] for i in range(max_prediction + 1)}

    plt.figure()
    fig, ax = plt.subplots(figsize=(5, len(color_to_label)))

    # Turn off the axis
    ax.axis('off')

    # Add the legend
    for i, (color, label) in enumerate(color_to_label.items()):
        color = [int(c) / 255 for c in color]
        ax.add_patch(plt.Rectangle((0, i), 1, 1, color=color))
        ax.text(1.1, i + 0.5, label, verticalalignment='center')

    # Set the limits
    ax.set_xlim(0, 2)
    ax.set_ylim(0, len(color_to_label))

    vis_path = os.path.join(base_path, 'legend.png')
    plt.savefig(vis_path, bbox_inches='tight')

    print(f"Saved legend to {vis_path}")



def visualize_prediction(pred: np.array, video_info: dict, idx_to_action: dict, base_path: str, size: int = 256):
    # Output the legend in a different file!
    colors = generate_distinct_colors(n=7, random_seed=115)

    pred_lbl, pred_lens = summarize_list(list(pred))

    pred_res = generate_image_for_segmentation(pred_lbl, pred_lens,
                                                colors=colors,
                                                height=10,
                                                white_label=[-100])
    
    pred = cv2.resize(pred_res, dsize=(size, 20), interpolation=cv2.INTER_NEAREST)

    plot_single_video_segmentation(pred, video_info)

    vis_path = os.path.join(base_path, 'pred.png')
    plt.savefig(vis_path, bbox_inches='tight')

    print(f"Saved fig to {vis_path}")

    # Generate a legend
    generate_legend(pred_lbl, idx_to_action, colors, base_path)


def post_process(cfg: CfgNode, pred: np.array, video_info: dict, base_path: str, active_surgery_pred_path: str, active_surgery: bool):
    # Open mapping file
    with open(cfg.DATA.PATH_TO_MAPPING_FILE, 'r') as f:
        lines = map(lambda l: l.strip().split(), f.readlines())
        action_to_idx = {action: int(str_idx) for str_idx, action in lines}
        idx_to_action = {idx: action for action, idx in action_to_idx.items()}

    # Format into the csv with start and end times
    """
    clip_id	annotation_type	label	duration	fps	start_time	end_time	video_path
    """
    fps = 1
    df = transform_array_to_df(pred, action_to_idx, fps, active_surgery_pred_path)
    df['clip_id'] = video_info['video_name']
    df['annotation_type'] = 'steps' if not active_surgery else 'active_surgery'
    df['fps'] = fps
    # df['video_path'] = video_info['video_path']

    df = df[['clip_id', 'annotation_type', 'label', 'duration', 'fps', 'start_time', 'end_time']] #, 'video_path']]

    df = df.sort_values(by='start_time')

    csv_path = os.path.join(base_path, 'pred.csv')
    df.to_csv(csv_path, index=False)
    print(f"Dumped formatted predictions to {csv_path}")

    # Visualize predictions (save as png)
    visualize_prediction(pred, video_info, idx_to_action, base_path)