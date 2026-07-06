#!/bin/bash

video_path=$1
filename=$(basename "$video_path")      # Extract the filename (e.g., file.txt)
video_id="${filename%.*}"  # Remove the extension (e.g., file)

frames_dir=full_video_frames_10fps
fps=10

ckpt_csv_path=checkpoints.csv

bleeding_prefix=bleeding_preds
spillage_prefix=spillage_preds
thermal_injury_prefix=thermal_injury_preds
postprocessed_dir=postprocessed_predictions

bleeding_preds_json=${bleeding_prefix}/bleeding_state_change_predictions/${video_id}/ensembled_preds_intersection.json
spillage_preds_json=${spillage_prefix}/spillage_state_change_predictions/${video_id}/ensembled_preds_intersection.json
thermal_injury_preds_json=${thermal_injury_prefix}/thermal_injury_state_change_predictions/${video_id}/ensembled_preds.json

echo Run frame extraction
bash ./scripts/extract_frames.sh ${video_path} ${fps} ${frames_dir}

echo Run blood and bile frame detection
bash ./scripts/blood_bile_detection.sh ${video_id} ${fps} ${frames_dir} ${bleeding_prefix} ${spillage_prefix}

echo Run bleeding
bash ./scripts/bleeding_single_video.sh ${video_id} ${fps} ${frames_dir} ${ckpt_csv_path} ${bleeding_prefix}

echo Run spillage
bash ./scripts/spillage_single_video.sh ${video_id} ${fps} ${frames_dir} ${ckpt_csv_path} ${spillage_prefix}

echo Run Thermal Injury
bash ./scripts/thermal_injury_single_video.sh ${video_id} ${fps} ${frames_dir} ${ckpt_csv_path} ${thermal_injury_prefix}

echo Run post-processing
bash ./scripts/postprocess_outputs.sh ${video_path} ${fps} ${bleeding_preds_json} ${spillage_preds_json} ${thermal_injury_preds_json} ${postprocessed_dir}