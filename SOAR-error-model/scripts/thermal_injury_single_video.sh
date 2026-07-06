#!/bin/bash
eval "$(conda shell.bash hook)"

error_name=thermal_injury

video_id=$1
fps=$2
frames_dir=$3
ckpt_csv_path=$4
prefix=$5

output_dir=${prefix}/${error_name}_detection_results_full_videos_test_10fps
json_output_dir=${prefix}/${error_name}_detection_result_clip_jsons
state_change_output_dir=${prefix}/${error_name}_state_change_predictions

## Run detection model
echo Run detection model
conda activate mmdet
python3 mmdetection/tools/run_inference.py \
    --all_video_frames_dir ${frames_dir} \
    --video_id ${video_id} \
    --output_dir ${output_dir} \
    --sampled_fps ${fps} \
    --ckpt_file ckpts/ckpt_frame/thermal_injury_detection_frame.pth
conda deactivate

# ## From the detection results, sample clips around the start of each detection at 
# ## different temporal resolutions (10 fps, 5 fps, 2 fps)

conda activate dino_tcn
echo Run state change models
for sampled_fps in 10 5 2
do
    python3 create_sampled_clip_jsons.py \
        --detection_json ${output_dir}/${video_id}.json \
        --original_fps ${fps} \
        --sampled_fps ${sampled_fps} \
        --output_dir ${json_output_dir} \
        --frames_dir ${frames_dir} \
        --video_id ${video_id} \
        --threshold 0.5
    
    # Run all the state change models
    python3 dinov2_tcn/run_inference_on_detection_results.py \
        --ann_json ${json_output_dir}/${video_id}/sampled_${sampled_fps}fps.json \
        --sampled_fps ${sampled_fps} \
        --output_dir ${state_change_output_dir}/${video_id} \
        --fps ${sampled_fps} \
        --error_name ${error_name} \
        --ckpt_csv ${ckpt_csv_path}
done

## Ensemble classification results
python3 ensemble_state_change_classification_results.py \
    --results_dir ${state_change_output_dir}/${video_id} \
    --annotations_dir ${json_output_dir}/${video_id} \
    --video_id ${video_id} \
    --threshold 0.5

conda deactivate