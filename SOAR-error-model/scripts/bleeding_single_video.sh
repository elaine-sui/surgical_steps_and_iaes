#!/bin/bash
eval "$(conda shell.bash hook)"

error_name=bleeding

video_id=$1
fps=$2
frames_dir=$3
ckpt_csv_path=$4
prefix=$5

output_dir=${prefix}/${error_name}_detection_results_full_videos_test_10fps
json_output_dir=${prefix}/${error_name}_detection_result_clip_jsons
state_change_output_dir=${prefix}/${error_name}_state_change_predictions

# Activate conda environment
conda activate dino_tcn

# From the detection results, sample clips around the start of each detection at 
# different temporal resolutions (5 fps, 2 fps, 1 fps)
for sampled_fps in 5 2 1
do
    python3 create_sampled_clip_jsons.py \
        --detection_json ${output_dir}/${video_id}.json \
        --original_fps ${fps} \
        --sampled_fps ${sampled_fps} \
        --output_dir ${json_output_dir} \
        --frames_dir ${frames_dir} \
        --video_id ${video_id} \
        --add_sliding_window_throughout_detected_segments \
        --threshold 0.3
    
    # Run all the state change models
    python3 dinov2_tcn/run_inference_on_detection_results.py \
        --ann_json ${json_output_dir}/${video_id}/sampled_${sampled_fps}fps.json \
        --sampled_fps ${sampled_fps} \
        --output_dir ${state_change_output_dir}/${video_id} \
        --fps ${sampled_fps} \
        --error_name ${error_name} \
        --ckpt_csv ${ckpt_csv_path}
done

# Ensemble with intersection of intervals
python3 ensemble_state_change_classification_results_v2.py \
    --results_dir ${state_change_output_dir}/${video_id} \
    --annotations_dir ${json_output_dir}/${video_id} \
    --video_id ${video_id} \
    --pool mean \
    --do_intersect \
    --threshold 0.5
conda deactivate