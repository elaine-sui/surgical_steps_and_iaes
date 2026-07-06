import argparse
import os
from glob import glob
from mmengine.dataset import BaseDataset
from mmdet.apis import init_detector
from mmengine.registry import DATASETS
from mmengine.runner import Runner

from typing import List
import imagesize
import json
import numpy as np

@DATASETS.register_module()
class FrameDataset(BaseDataset):
    """Dataset for Thermal Injury Detection"""

    METAINFO = {
        'classes': ('thermal_injury'),
        'palette': [(255, 255, 100)]
    }

    def __init__(self, data_root, metainfo, pipeline):
        self.height, self.width = None, None
        super().__init__(metainfo=metainfo, data_root=data_root, pipeline=pipeline, test_mode=True)


    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  
        image_paths = glob(f'{self.data_root}/*.jpeg')
        frame_nums = [int(os.path.split(f)[1].split('.')[0]) for f in image_paths]
        idx = np.argsort(frame_nums)
        image_paths = [image_paths[i] for i in idx]

        self.cat_ids = self.metainfo['classes']
        self.cat2label = {cat_id: i+1 for i, cat_id in enumerate(self.cat_ids)}

        data_list = []
        for img_id, img_path in enumerate(image_paths):

            if img_id == 0:
                self.width, self.height = imagesize.get(img_path)

            parsed_data_info = {
                'img_path': img_path,
                'img_id': img_id,
                'height': self.height,
                'width': self.width,

            }
            data_list.append(parsed_data_info)

        return data_list


def build_dataloader(data_root):
    test_pipeline = [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='Resize', scale=(399, 224), keep_ratio=True),
        dict(
            type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor'))
    ]

    metainfo = {
        'classes': ('thermal_injury'),
        'palette': [(255, 255, 100)]
    }

    dataloader_dict=dict(
        batch_size=16,
        sampler=dict(
            type='DefaultSampler',
            shuffle=False),
        dataset=dict(type='FrameDataset',
            data_root=data_root,
            metainfo=metainfo,
            pipeline=test_pipeline)
    )

    dataloader = Runner.build_dataloader(dataloader_dict)

    return dataloader


def load_model(config_file, ckpt_file):
    model = init_detector(config_file, ckpt_file, device='cuda:0')

    return model


def run_inference(model, batch_inputs): #, threshold):
    # Run inference
    results = model(**batch_inputs, mode='predict')

    boxes = [r.pred_instances.bboxes for r in results]
    labels = [r.pred_instances.labels for r in results]
    scores = [r.pred_instances.scores for r in results]

    # Restrict to only bboxes that > threshold confidence
    # classnames = [id2cat[l] for l in labels[0]]
    ret = []
    for box, label, score in zip(boxes, labels, scores):
        ret.append({'bboxes': box.detach().cpu().numpy().tolist(), 'scores': score.detach().cpu().numpy().tolist(), 'labels': label.detach().cpu().numpy().tolist()})
    return ret


def main(args):

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...")
    model = load_model(args.config_file, args.ckpt_file)
    print("Done loading model")

    print("Running inference...")
    video_id = args.video_id
    vid_frames_dir = os.path.join(args.all_video_frames_dir, video_id)
    output_json = os.path.join(args.output_dir, video_id + ".json")

    print("Building dataloader...")
    dataloader = build_dataloader(vid_frames_dir)
    print("Finished building dataloader")

    video_result = {}
    preds = []
    for i, batch_input in enumerate(dataloader):
        batch_input = model.data_preprocessor(batch_input, training=False)
        ret = run_inference(model, batch_input) #, threshold=args.threshold)

        for i, res in enumerate(ret):
            img_path = batch_input['data_samples'][i].img_path
            frame_id = os.path.split(img_path)[1].split('.')[0]
            video_result[frame_id] = res

            if res['bboxes'] is not None:
                preds.append(1)
            else:
                preds.append(0)

    with open(output_json, 'w') as f:
        json.dump({'annotations': video_result}, f)
        print(f"Dumped to {output_json}")



    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='mmdetection/configs/deformable_detr/deformable-detr-refine-twostage_r50_15xb2-50e_coco_thermal_injury_w_more_aug.py')
    parser.add_argument('--ckpt_file', type=str, default='ckpt.pth')
    parser.add_argument('--all_video_frames_dir', type=str, default='full_video_frames_10fps')
    parser.add_argument('--video_id', type=str, default='video33')
    parser.add_argument('--output_dir', type=str, default='detection_results_full_videos_test_10fps')
    parser.add_argument('--sampled_fps', type=int, default=10)

    args = parser.parse_args()

    main(args)