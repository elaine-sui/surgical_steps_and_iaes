#!/bin/bash
eval "$(conda shell.bash hook)"

video_path=$1
fps=$2
frames_dir=$3

# Extract video frames
conda activate dino_tcn
python3 extract_video_frames.py \
    --frames_dir ${frames_dir} \
    --sampling_frame_rate ${fps} \
    --video_path ${video_path}
conda deactivate