checkpoint=$1 #/path/to/activ_surgery_model.pyth
feats_path=$2 #/path/to/features/vid.npy
output_dir=$3 #/path/to/output_dir outputs_active_surgery
video_id=$4

python step_model/run_net_inference.py \
    --video_feats_path ${feats_path} \
    --cfg step_model/configs/LTContext_lovit_active_surgery.yaml \
    --save_preds \
    --output_dir ${output_dir} \
    TEST.CHECKPOINT_PATH ${checkpoint} \
    DATA.FRAME_SAMPLING_RATE 1 \
    NUM_GPUS 1

active_surgery_output_path=${output_dir}/results/preds/${video_id}/pred.csv

# post-process: if the prediction is only background, change to only active_surgery
python3 postprocess_active_surgery.py \
    --active_surgery_predictions ${active_surgery_output_path}