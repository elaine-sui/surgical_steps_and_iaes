#!/bin/bash
eval "$(conda shell.bash hook)"

video_path=$1
models_dir=$2
filename=$(basename "$video_path")      # Extract the filename (e.g., file.txt)
video_id="${filename%.*}"  # Remove the extension (e.g., file)

sampled_fps=1

features_dir=features
encoder_path=${models_dir}/step_encoder.pth
features_path=${features_dir}/${video_id}.npy

# Activate conda environment
conda activate feature_extract

echo Extract features

encoder_model=$3 #frame encoder model name
encoder_model_arch=$4 #frame encoder model architecture
ckpt=${encoder_path} # optional

bash ./scripts/1_preprocessing.sh ${encoder_model} ${encoder_model_arch} ${video_path} ${features_dir} ${sampled_fps} ${ckpt}

active_surgery_model_path=${models_dir}/active_surgery_model.pyth
active_surgery_output_dir=outputs_active_surgery_test

conda deactivate
conda activate step_model

echo Find start and end times of active surgery
bash ./scripts/1b_find_active_surgery_times.sh ${active_surgery_model_path} ${features_path} ${active_surgery_output_dir} ${video_id}

step_model_path=${models_dir}/step_model.pyth

output_dir=outputs
active_surgery_output_path=${active_surgery_output_dir}/results/preds/${video_id}/pred.csv

echo Run step model
bash ./scripts/2_run_segmentation.sh ${step_model_path} ${features_path} ${output_dir} ${active_surgery_output_path}

pred_file=${output_dir}/results/preds/${video_id}/pred.csv
echo Post-processing
bash ./scripts/3_postprocess_predictions.sh ${pred_file}

conda deactivate