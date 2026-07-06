checkpoint=$1 #/path/to/step_model.pyth
feats_path=$2 #/path/to/features/vid.npy
output_dir=$3 #/path/to/output_dir outputs
active_surgery_pred_path=$4 #/path/to/output_dir_active_surgery/results/pred/vid.csv

python step_model/run_net_inference.py \
    --video_feats_path ${feats_path} \
    --cfg step_model/configs/LTContext_lovit.yaml \
    --save_preds \
    --output_dir ${output_dir} \
    --active_surgery_pred_path ${active_surgery_pred_path} \
    TEST.CHECKPOINT_PATH ${checkpoint} \
    DATA.FRAME_SAMPLING_RATE 1 \
    NUM_GPUS 1