#!/bin/bash
eval "$(conda shell.bash hook)"

video_id=$1
fps=$2
frames_dir=$3
bleeding_prefix=$4
spillage_prefix=$5

bleeding_output_dir=${bleeding_prefix}/bleeding_detection_results_full_videos_test_10fps
spillage_output_dir=${spillage_prefix}/spillage_detection_results_full_videos_test_10fps

# Activate conda environment
conda activate dino_tcn

## Run framewise binary classification model
echo Run framewise binary classification
python3 dinov2_tcn/run_framewise_classification_multiclass.py \
    --ann_dir dummy_frame_annotations \
    --video_frames_dir ${frames_dir} \
    --blood_output_dir ${bleeding_output_dir} \
    --bile_output_dir ${spillage_output_dir} \
    --fps ${fps} \
    --video_id ${video_id} \
    --use_backbone_only \
    --model DINOv2-base-multiclass \
    --checkpoint ckpts/ckpt_frame/multiclass_1head_dino_frame.pt

conda deactivate