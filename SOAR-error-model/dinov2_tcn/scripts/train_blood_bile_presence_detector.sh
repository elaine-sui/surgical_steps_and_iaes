#!/bin/bash
eval "$(conda shell.bash hook)"

# Activate conda environment
conda activate dino_tcn2

exp_name=blood_merged_and_bile_frame_classification_multiclass_dino_base_0.00003
annotations_dir=/path/to/frame_annotations/Blood_Merged_And_Bile

# training
python3 run.py \
    --annotations_dir ${annotations_dir} \
    --exp_name ${exp_name} \
    --model DINOv2-base-multiclass \
    --use_backbone_only \
    --dropout 0.1 \
    --lr 0.00003 \
    --num_epochs 40 \
    --batch_size 128 \
    --num_workers 8 \
    --train \
    --test
