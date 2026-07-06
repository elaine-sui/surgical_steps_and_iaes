model=$1 #frame encoder model name
model_arch=$2 #frame encoder model architecture
video_path=$3 #/path/to/video
features_dir=$4 #/path/to/features
sampled_fps=$5
ckpt_path=$6

filename=$(basename "$video_path")      # Extract the filename (e.g., file.txt)
video_id="${filename%.*}"  # Remove the extension (e.g., file)
video_dir=$(dirname "$video_path")  # Extract the directory (e.g., /path/to)

cd extract_features
python3 extract_tad_feature.py \
    --model_name ${model} \
    --model_arch ${model_arch} \
    --video_dir ${video_dir} \
    --save_path ../${features_dir} \
    --video_lst ${video_id} \
    --fps ${sampled_fps} \
    --ckpt_path ${ckpt_path}