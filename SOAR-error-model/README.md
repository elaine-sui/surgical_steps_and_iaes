# Surgical Error Detection

This is the repo for the surgical error detection model. It is used to automatically detect instances of **bleeding, bile spillage, and thermal injury**.

## Getting Started

### GPU and CUDA requirements
- The models were developped using CUDA=12.2
- The GPUs types used were NVIDIA A6000s and L40s (48 GB) [TODO: figure out minimum mem requirements]

### Create conda environments
Create the `dino_tcn` conda environment:
```
conda env create -f dinov2_tcn/environment.yml
```

Create the `mmdet` conda environment and install MMCV/MMDetection (the thermal injury detector uses stock `mmdet`, so it is installed as a normal dependency rather than from a vendored source tree):
```
conda env create -f mmdetection/environment.yml
conda activate mmdet
mim install mmcv==2.1.0 mmdet==3.3.0 mmengine==0.10.7
```

## Training the model components
The error detection pipeline is built from several independently trained models. There are two families:
- **Frame-level models** classify or detect the error in a single frame. These are: (1) the blood and bile frame classifier, and (2) the thermal injury detector.
- **Temporal (clip-level) models** classify a short candidate clip as containing the error or not. For each error there are **6 temporal models** — two sampling distributions (`gaussian` and `uniform`) at each of three frame rates. These are: (3) the bleeding temporal classifier, (4) the bile spillage temporal classifier, and (5) the thermal injury temporal classifier.

All frame-level classifiers and temporal models live in `dinov2_tcn` and are trained through `dinov2_tcn/run.py`. The thermal injury *detector* is a Deformable DETR model trained with `mmdetection`.

Each training script trains, then evaluates on the test split (`--train --test`). Checkpoints for the `dinov2_tcn` models are written to `<prefix_dir>/<output_dir>/<exp_name>/<timestamp>/` (by default `prefix_dir=/path/to/soar_wellcome_save/dinov2_tcn/` and `output_dir=checkpoints`); the best checkpoint is reloaded and re-evaluated at the end of training. Edit `annotations_dir` (and the `source ~/miniconda3/...` line) in each script to point at your data and conda install before running.

### 1) Blood and bile frame classifier
A DINOv2-base multiclass frame classifier (the backbone with a single classification head) trained on COCO-format frame annotations (`Blood_Merged_And_Bile`). It produces the `multiclass_1head_dino_frame.pt` checkpoint used for frame-level blood/bile presence.
```
conda activate dino_tcn
cd dinov2_tcn
bash scripts/train_blood_bile_presence_detector.sh
```
Key settings (in the script): `--model DINOv2-base-multiclass --use_backbone_only --lr 0.00003 --num_epochs 40 --batch_size 128`.

### 2) Thermal injury detector
The frame-level thermal injury detector is a Deformable DETR model trained with `mmdet` via the lightweight `tools/train_detector.py` script (the model, dataset, and schedule are all defined in the config). Set `data_root` in `mmdetection/configs/_base_/datasets/coco_detection_detr_thermal_injury_w_more_aug.py` to the COCO-format annotations directory containing `training.json`, `validation.json`, and `testing.json`, then train (the sbatch script activates the `mmdet` env itself):
```
cd mmdetection
sbatch tools/train_thermal_injury_detector.bash
```
Checkpoints and logs are written to `work_dirs/<config_name>/`. Training logs to Weights & Biases (configured in `configs/_base_/default_runtime.py`).

Evaluate a trained checkpoint (COCO bbox mAP; add `--out preds.pkl` to dump per-frame predictions for `tools/compute_frame_presence_metrics.py`):
```
python tools/test_detector.py \
    configs/deformable_detr/deformable-detr-refine-twostage_r50_15xb2-50e_coco_thermal_injury_w_more_aug.py \
    work_dirs/deformable-detr-refine-twostage_r50_15xb2-50e_coco_thermal_injury_w_more_aug/epoch_25.pth
```

### 3) Bleeding temporal classifier
A DINOv2 backbone with a 4-layer TCN head trained on sampled candidate clips around state-change (`active_bleeding`). Train all 6 models — two distributions (`gaussian`, `uniform`) at 3 frame rates:

| Script | Frame rate | Clip length |
| --- | --- | --- |
| `gaussian_160.sh` / `uniform_160.sh` | 1 fps | 160 frames |
| `gaussian_80.sh` / `uniform_80.sh` | 2 fps | 80 frames |
| `gaussian_32.sh` / `uniform_32.sh` | 5 fps | 32 frames |

```
conda activate dino_tcn
cd dinov2_tcn
# run each of the 6 scripts (example: gaussian sampling at 1 fps)
bash scripts/train_bleeding_temporal_detector/gaussian_160.sh
```
Key settings (in each script): `--model DINOv2 --num_tcn_layers 4 --dropout 0.3 --lr 0.00003 --num_epochs 40 --with_pnr`, with `--fps` matching the table above. These produce the `bleeding_{1,2,5}fps_{1,2}.pt` checkpoints.

### 4) Bile spillage temporal classifier
Same architecture and frame-rate/distribution grid as the bleeding classifier, trained on `bile_spillage` clips. Produces the `spillage_{1,2,5}fps_{1,2}.pt` checkpoints.
```
conda activate dino_tcn
cd dinov2_tcn
# run each of the 6 scripts (example: uniform sampling at 2 fps)
bash scripts/train_bile_spillage_temporal_detector/uniform_80.sh
```

| Script | Frame rate | Clip length |
| --- | --- | --- |
| `gaussian_160.sh` / `uniform_160.sh` | 1 fps | 160 frames |
| `gaussian_80.sh` / `uniform_80.sh` | 2 fps | 80 frames |
| `gaussian_32.sh` / `uniform_32.sh` | 5 fps | 32 frames |

### 5) Thermal injury temporal classifier
Same architecture as the other temporal classifiers, trained on `thermal_injury` clips, but at higher frame rates (thermal injury is a faster event). Produces the `thermal_injury_{2,5,10}fps_{1,2}.pt` checkpoints.
```
conda activate dino_tcn
cd dinov2_tcn
# run each of the 6 scripts (example: gaussian sampling at 10 fps)
bash scripts/train_thermal_injury_temporal_detector/gaussian_16.sh
```

| Script | Frame rate | Clip length |
| --- | --- | --- |
| `gaussian_80.sh` / `uniform_80.sh` | 2 fps | 80 frames |
| `gaussian_32.sh` / `uniform_32.sh` | 5 fps | 32 frames |
| `gaussian_16.sh` / `uniform_16.sh` | 10 fps | 16 frames |

## Evaluating metrics
After running the pipeline over a test split, `evaluate.py` computes interval-level precision/recall (and F1/F2) of the post-processed predictions against the ground-truth annotations. A prediction is a *hit* if a ground-truth state change falls within the predicted interval; metrics are reported across a sweep of score thresholds and written as JSON to `<metrics_dir>/<pred_label>/<split>.json`.

The key arguments are the prediction directory (`--pred_dir`), the predicted error label being scored (`--pred_label`), the ground-truth labels it is matched against (`--label_lst`), the ground-truth column in the datasplit CSV (`--gt_col`), and the score thresholds to sweep (`--thresholds`). `--between_steps2and5` restricts evaluation to predictions between the *Expose Gallbladder* (step 2) and *Gallbladder Dissection* (step 5) steps.

### Bleeding
```
conda activate dino_tcn

python3 evaluate.py \
    --pred_dir postprocessed_predictions \
    --pred_label bleeding \
    --label_lst "Bleeding" \
    --gt_col bleeding_csv \
    --between_steps2and5 \
    --thresholds 0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.9 \
    --datasplit_csv /path/to/datasplit.csv \
    --split testing \
    --metrics_dir metrics
```

### Bile spillage
```
conda activate dino_tcn

python3 evaluate.py \
    --pred_dir postprocessed_predictions \
    --pred_label spillage \
    --label_lst "Bile Spillage" \
    --gt_col bile_spillage_csv \
    --between_steps2and5 \
    --thresholds 0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.9 \
    --datasplit_csv /path/to/datasplit.csv \
    --split testing \
    --metrics_dir metrics
```

### Thermal injury
```
conda activate dino_tcn

python3 evaluate.py \
    --pred_dir postprocessed_predictions \
    --pred_label thermal_injury \
    --label_lst "Thermal Injury" \
    --gt_col thermal_injury_csv \
    --between_steps2and5 \
    --thresholds 0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.9 \
    --datasplit_csv /path/to/datasplit.csv \
    --split testing \
    --metrics_dir metrics
```

## Inference

### Download model checkpoints and put them in the following directories:
Download the zip file of model checkpoints from [Google Drive](https://drive.google.com/file/d/1exLB0qatiZcL2-2zmgIkjitLje_Vo8i8/view?usp=sharing) and unzip it.

The directory structure should be as follows:
```
ckpts/
|–– ckpt_frame/
    |–– multiclass_1head_dino_frame.pt
    |–– thermal_injury_detection_frame.pth
|–– ckpt_state_change/
    |–– bleeding_1fps_1.pt
    |–– bleeding_1fps_2.pt
    |–– bleeding_2fps_1.pt
    |–– bleeding_2fps_2.pt
    |–– bleeding_5fps_1.pt
    |–– bleeding_5fps_2.pt
    |–– spillage_1fps_1.pt
    |–– spillage_1fps_2.pt
    |–– spillage_2fps_1.pt
    |–– spillage_2fps_2.pt
    |–– spillage_5fps_1.pt
    |–– spillage_5fps_2.pt
    |–– thermal_injury_2fps_1.pt
    |–– thermal_injury_2fps_2.pt
    |–– thermal_injury_5fps_1.pt
    |–– thermal_injury_5fps_2.pt
    |–– thermal_injury_10fps_1.pt
    |–– thermal_injury_10fps_2.pt
```

### Running the error detection pipeline
Run this line to run the error detection pipeline:
```
bash ./scripts/run_single_video.sh /path/to/[video_id].mp4
```

**Input:** video path (`/path/to/[video_id].mp4`)
    - Sample video: [here](https://drive.google.com/file/d/1JZsENae-vtagjOr72rdoqo6SJ7542fp2/view?usp=drive_link)

**Output:** CSV with the detected errors (`postprocessed_predictions/[video_id].csv`)
    - The output columns are: ['clip_id', 'annotation_type', 'label', 'duration', 'fps', 'start_frame', 'end_frame', 'start_time', 'end_time', 'score', 'video_path']


This script does the following in sequence for a single video:
- Extracts frames from `/path/to/[video_id].mp4` at 10 fps
- Runs the bleeding detection model
- Runs the bile spillage detection model
- Runs the thermal injury detection model
- Post-processes all the error predictions into a single CSV

Note: the models that detect each error are independent of one another and could be parallelized by running it on multiple GPUs simultaneously.

### More detail on how the error model works
To detect a specific error, we first
1. Classify each frame as having blood/bile/thermal injury with an ML model
2. Sample candidate clips from the full-length video where an error is most likely to occur
3. Classify each candidate clip with 6 ML models (2 each for 3 different frame rates) and combine the model predictions (Note: these models are run in sequence for simplicity, but could be parallelized)
4. Post-process the results and only keep the ones over a certain threshold (0.4 for bleeding ; 0.45 for bile spillage; 0.5 for thermal injury)