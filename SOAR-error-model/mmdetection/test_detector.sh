#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate mmdet

config=configs/deformable_detr/deformable-detr-refine-twostage_r50_15xb2-50e_coco_thermal_injury_w_more_aug.py
ckpt=/path/to/checkpoint.pth

python tools/test_detector.py ${config} ${ckpt} \
    --work-dir work_dirs/deformable-detr-refine-twostage_r50_15xb2-50e_coco_thermal_injury_w_more_aug \
    --data-root /path/to/frame_annotations/