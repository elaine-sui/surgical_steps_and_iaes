#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate step_model

# -------------------------------------------------------------------------------------
# User-configurable settings (edit these for your dataset)
# -------------------------------------------------------------------------------------
# Master CSV describing your dataset (columns: video_id,mode,feat_path,gt_path,fps).
# See the README ("Train the active surgery model") for the expected format.
master_csv="/path/to/your/master.csv"

# Mapping file with the two classes (background / activeSurgery).
# Path is relative to the step_model/ directory.
mapping_file="mapping_files/active_surgery.txt"

# Directory where checkpoints, configs and results are written.
output_dir="experiments/active_surgery"

# Short tag included in the output/experiment paths.
exp_prefix="active_surgery"
# -------------------------------------------------------------------------------------

cd step_model

# Train, then evaluate on the test split.
python run_net_training.py \
    --cfg configs/LTContext_lovit_active_surgery.yaml \
    TRAIN.ENABLE True \
    TEST.ENABLE True \
    TRAIN.EVAL_SPLIT test \
    OUTPUT_DIR ${output_dir} \
    NUM_GPUS 1 \
    DATA.PATH_TO_MAPPING_FILE ${mapping_file} \
    DATA.PATH_TO_MASTER_CSV ${master_csv} \
    EXP_PREFIX ${exp_prefix}
