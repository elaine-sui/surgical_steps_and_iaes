#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate step_model

# -------------------------------------------------------------------------------------
# User-configurable settings (edit these for your dataset)
# -------------------------------------------------------------------------------------
# Master CSV describing your dataset (columns: video_id,mode,feat_path,gt_path,fps).
# See the README ("Train the step model") for the expected format.
master_csv="/path/to/your/master.csv"

# Mapping file listing the step classes, one "<index> <name>" per line.
# Path is relative to the step_model/ directory.
mapping_file="mapping_files/chole_steps.txt"

# Number of step classes (must match the number of lines in mapping_file).
num_classes=5

# Directory where checkpoints, configs and results are written.
output_dir="experiments/step_model"

# Short tag included in the output/experiment paths.
exp_prefix="step_model"
# -------------------------------------------------------------------------------------

cd step_model

# Train, then evaluate on the test split.
python run_net_training.py \
    --cfg configs/LTContext_lovit.yaml \
    TRAIN.ENABLE True \
    TEST.ENABLE True \
    TRAIN.EVAL_SPLIT test \
    OUTPUT_DIR ${output_dir} \
    NUM_GPUS 1 \
    DATA.PATH_TO_MAPPING_FILE ${mapping_file} \
    MODEL.NUM_CLASSES ${num_classes} \
    DATA.PATH_TO_MASTER_CSV ${master_csv} \
    EXP_PREFIX ${exp_prefix}

# -------------------------------------------------------------------------------------
# Optional: evaluate a previously trained checkpoint without re-training.
# Set ckpt to your "best_checkpoint.pyth" and uncomment the block below.
# -------------------------------------------------------------------------------------
# ckpt="/path/to/best_checkpoint.pyth"
# python run_net_training.py \
#     --cfg configs/LTContext_lovit.yaml \
#     TRAIN.ENABLE False \
#     TEST.ENABLE True \
#     TRAIN.EVAL_SPLIT test \
#     OUTPUT_DIR ${output_dir} \
#     NUM_GPUS 1 \
#     DATA.PATH_TO_MAPPING_FILE ${mapping_file} \
#     MODEL.NUM_CLASSES ${num_classes} \
#     DATA.PATH_TO_MASTER_CSV ${master_csv} \
#     EXP_PREFIX ${exp_prefix} \
#     TEST.CHECKPOINT_PATH ${ckpt}
