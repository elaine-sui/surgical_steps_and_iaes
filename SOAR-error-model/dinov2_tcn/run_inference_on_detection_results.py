import os
import argparse
import json
import pandas as pd

from src.builder import build_model, build_clipwise_sliding_window_dataloader
from src import train


def get_checkpoint_paths(args):
    df_ckpt = pd.read_csv(args.ckpt_csv)

    error_name = None
    if args.error_name.startswith('thermal_injury'):
        error_name = 'thermal_injury'
    elif args.error_name.startswith('perforation') or args.error_name.startswith('spillage'):
        error_name = 'spillage'
    elif args.error_name.startswith('bleeding'):
        error_name = 'bleeding'
    else:
        raise ValueError(f'{error_name} is None!')

    df_ckpt = df_ckpt[df_ckpt['error_type'] == error_name]
    df_ckpt = df_ckpt[df_ckpt['fps'] == args.fps]

    return list(df_ckpt['checkpoint_path'].values)


def main(args):
    ckpt_files = get_checkpoint_paths(args)
    os.makedirs(args.output_dir, exist_ok=True)
    for ckpt_file in ckpt_files:
        args.checkpoint = ckpt_file
        config_name = args.checkpoint.split('/')[-1].split('.')[0]
        model = build_model(args)
        test_loader = build_clipwise_sliding_window_dataloader(args)

        preds = train.predict(model,test_loader,use_bf16=args.use_bf16)

        json_output = {'predictions': preds, 'annotation_file': args.ann_json}

        save_path = os.path.join(args.output_dir, config_name + ".json")
        with open(save_path, 'w') as f:
            json.dump(json_output, f)
        print(f'Predictions saved to {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_json', type=str, default='annotations')
    parser.add_argument('--sampled_fps', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='results_sliding_window_errors')
    parser.add_argument('--error_name', type=str, default=None)

    # model
    parser.add_argument('--model', type=str, default='DINOv2')
    parser.add_argument('--use_backbone_only', action='store_true')
    parser.add_argument('--num_tcn_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--no_lora', action='store_true')
    parser.add_argument('--use_bf16', action='store_true')
    parser.add_argument('--no_pooling', action='store_true')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--with_pnr', action='store_true')
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--fps', type=int, default=10)

    # train and test
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--ckpt_csv', type=str, default=None)

    args = parser.parse_args()

    main(args)