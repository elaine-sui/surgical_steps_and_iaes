import os
import argparse
from tqdm import trange
from glob import glob
import json
import numpy as np

from src.builder import build_model, build_framewise_sliding_window_dataloader
from src import train

def create_annotations_json(args):
    frame_file_lst = sorted(glob(f'{args.video_frames_dir}/{args.video_id}/*.jpeg'))
    frame_nums = [int(f.split('/')[-1].replace('.jpeg', '')) for f in frame_file_lst]
    idx_frame = np.argsort(frame_nums)
    frame_file_lst = [frame_file_lst[i] for i in idx_frame]
    num_frames = len(frame_file_lst)

    coco_ann_lst = []
    video_id = args.video_id
    for start_frame_batch in trange(0, num_frames, args.batch_size):
        frames = frame_file_lst[start_frame_batch:start_frame_batch+args.batch_size]

        for frame_path in frames:
            frame_num = frame_path.split('/')[-1].split('.')[0]

            item = {
                'id': int(frame_num),
                'file_name': frame_path,
                'video_id': video_id,
                'ds': None,
                'video_name': video_id
            }

            coco_ann_lst.append(item)

    with open(args.ann_json, 'w') as f:
        json.dump({'images': coco_ann_lst}, f)
    print(f'Dumped to {args.ann_json}')


def main(args):
    os.makedirs(args.blood_output_dir, exist_ok=True)
    os.makedirs(args.bile_output_dir, exist_ok=True)
    os.makedirs(args.ann_dir, exist_ok=True)

    # Create dummy annotation file
    args.ann_json = os.path.join(args.ann_dir, f"{args.video_id}.json")
    create_annotations_json(args)

    model = build_model(args)
    test_loader = build_framewise_sliding_window_dataloader(args)
    
    assert args.model in ['DINOv2-base-multiclass', 'DINOv2-base-2head'], f'must be a multiclass model!'
    blood_preds, bile_preds = train.predict_multiclass(model,test_loader,use_bf16=args.use_bf16)

    # Reformat predictions
    blood_reformatted_preds = {pred['id']: {'scores': pred['pred']} for pred in blood_preds}
    bile_reformatted_preds = {pred['id']: {'scores': pred['pred']} for pred in bile_preds}
    blood_json_output = {'annotations': blood_reformatted_preds}
    bile_json_output = {'annotations': bile_reformatted_preds}

    blood_save_path = os.path.join(args.blood_output_dir, args.video_id + ".json")
    with open(blood_save_path, 'w') as f:
        json.dump(blood_json_output, f)
    print(f'Blood predictions saved to {blood_save_path}')

    bile_save_path = os.path.join(args.bile_output_dir, args.video_id + ".json")
    with open(bile_save_path, 'w') as f:
        json.dump(bile_json_output, f)
    print(f'Bile predictions saved to {bile_save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_dir', type=str, default='annotations')
    parser.add_argument('--video_frames_dir', type=str, default=None)
    parser.add_argument('--sampled_fps', type=int, default=10)
    parser.add_argument('--blood_output_dir', type=str, default='results_sliding_window_errors')
    parser.add_argument('--bile_output_dir', type=str, default='results_sliding_window_errors')
    parser.add_argument('--video_id', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)

    # model
    parser.add_argument('--model', type=str, default='DINOv2')
    parser.add_argument('--use_backbone_only', action='store_true')
    parser.add_argument('--num_tcn_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--no_lora', action='store_true')
    parser.add_argument('--use_bf16', action='store_true')
    parser.add_argument('--no_pooling', action='store_true')

    parser.add_argument('--with_pnr', action='store_true')
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--fps', type=int, default=10)

    # train and test
    parser.add_argument('--checkpoint', type=str, default=None)

    args = parser.parse_args()

    main(args)