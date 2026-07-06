import argparse
import os
from tqdm import tqdm
import pandas as pd

import numpy as np
import torch
import math
import cv2

from dataset.loader import get_video_loader

from model_builder import get_model, get_transform


def get_args():
    parser = argparse.ArgumentParser(
        'Extract video frame features', add_help=False)

    parser.add_argument(
        '--batch_size',
        default=16,
        type=int,
        help='number of frames in a single batch')

    parser.add_argument(
        '--video_lst',
        type=str,
        help='list of video ids')
    parser.add_argument(
        '--video_dir',
        type=str,
        help='directory to videos')

    parser.add_argument(
        '--video_lst_csv',
        type=str,
        help='video list csv (if video dir is not provided)'
    )

    parser.add_argument(
        '--save_path',
        default='YOUR_PATH/thumos14_video/th14_vit_g_16_4',
        type=str,
        help='path for saving features')

    parser.add_argument(
        '--model_name',
        default="CLIP",
        type=str,
        metavar='MODEL',
        help='Name of model')

    parser.add_argument(
        '--model_arch',
        default="ViT-B/32",
        type=str,
        metavar='ARCH',
        help='architecture of model')

    parser.add_argument(
        '--download_root',
        default=None,
        type=str,
    )

    parser.add_argument(
        '--ckpt_path',
        default=None,
        type=str,
    )

    parser.add_argument(
        '--local_center_crop',
        action='store_true'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=10
    )

    return parser.parse_args()


def get_start_idx_range(batch_size):
    def dataset_range(num_frames):
        return range(0, num_frames, batch_size)

    return dataset_range


def extract_feature(args):
    # preparation
    os.makedirs(args.save_path, exist_ok=True)
    video_loader = get_video_loader()

    transform = get_transform(args.model_name, args.local_center_crop)

    # get video path
    vid_list = args.video_lst.split(',')

    # load model
    model = get_model(args.model_name, args.model_arch, download_root=args.download_root, ckpt_path=args.ckpt_path)
    model.eval()
    model.cuda()
    print(f"Model {args.model_name} loaded!")

    start_idx_range = get_start_idx_range(args.batch_size)

    df_videos = None
    if args.video_lst_csv != 'None' and args.video_lst_csv is not None:
        df_videos = pd.read_csv(args.video_lst_csv, dtype={'video_id': str})
        df_videos['video_id'] = df_videos['video_id'].astype('string')
        df_videos = df_videos.set_index('video_id')

    # extract feature
    num_videos = len(vid_list)
    for idx, vid_name in enumerate(vid_list):
        url = os.path.join(args.save_path, vid_name.split('.')[0] + '.npy')
        if os.path.exists(url):
            continue
        
        if args.video_dir != 'None' and args.video_dir is not None:
            video_path = os.path.join(args.video_dir, vid_name + ".mp4")
        else:
            video_path = df_videos.loc[vid_name, 'video_path']

        try:
            vr = video_loader(video_path)
        except:
            print(f"Corrupted video: {video_path}. Skip.")
            continue

        feature_list = []

        if df_videos is None:
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
        else:
            fps = df_videos.loc[vid_name, 'fps']

        factor = math.ceil(fps / args.fps)
        print(f"Factor: {factor} FPS: {args.fps}")

        for start_idx in tqdm(start_idx_range(len(vr) // factor), total=len(start_idx_range(len(vr) // factor))):
            data = vr.get_batch(np.arange(start_idx, min(start_idx + args.batch_size, len(vr)//factor)) * factor).asnumpy()

            if args.model_name.startswith('DINOv3'):
                inputs = transform(images=data, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    out = model(**inputs).last_hidden_state # Take CLS token's embedding
                    cls_token = out[:, 0]
                    patch_tokens = out[:, 1 + model.config.num_register_tokens:]
                    feature = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
            else:
                frame = torch.from_numpy(data)  # torch.Size([batch_size, h, w, 3])
                input_data = transform(frame).cuda()  # torch.Size([batch_size, 3, 224, 224])

                if args.model_name == 'CLIP':
                    input_data = input_data.to(torch.float16)

                with torch.no_grad():
                    feature = model(input_data)
            
            feature_list.append(feature.cpu().numpy())

        # [N, C]
        np.save(url, np.concatenate(feature_list, axis=0))
        print(f'[{idx} / {num_videos}]: save feature on {url}')


if __name__ == '__main__':
    args = get_args()
    extract_feature(args)
