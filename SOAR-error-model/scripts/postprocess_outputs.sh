#!/bin/bash
eval "$(conda shell.bash hook)"

video_path=$1
filename=$(basename "$video_path")      # Extract the filename (e.g., file.txt)
video_id="${filename%.*}"  # Remove the extension (e.g., file)

fps=$2
bleeding_preds_json=$3
spillage_preds_json=$4
thermal_injury_preds_json=$5
postprocessed_dir=$6

# Activate conda environment
conda activate dino_tcn

python3 postprocess_and_combine_preds.py \
    --postprocessed_csv ${postprocessed_dir}/${video_id}.csv \
    --bleeding_preds_json ${bleeding_preds_json} \
    --spillage_preds_json ${spillage_preds_json} \
    --thermal_injury_preds_json ${thermal_injury_preds_json} \
    --video_path ${video_path} \
    --fps ${fps}
conda deactivate