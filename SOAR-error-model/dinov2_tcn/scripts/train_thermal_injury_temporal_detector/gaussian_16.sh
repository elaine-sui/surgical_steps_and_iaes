num_frames=16
distribution=gaussian
label=thermal_injury
suffix=around_start_${num_frames}_${distribution}_20-80_split_w_pnr

python3 run.py \
    --annotations_dir /path/to/annotations \
    --exp_name ${label}_temporal_dino_base_tcn_0.00003_4_layer_${suffix} \
    --model DINOv2 \
    --num_tcn_layers 4 \
    --dropout 0.3 \
    --lr 0.00003 \
    --num_epochs 40 \
    --train \
    --test \
    --with_pnr \
    --fps 10
