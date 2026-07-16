# Step Segmentation Model

## Configuration
- Conda version: 23.7.4
- CUDA version: 11.5
- GPU type: NVIDIA Titan RTX (24 GB) [Note: also works on NVIDIA A6000 and L40s]

## Environment setup
```
conda env create -f environment.yml
conda activate step_model
```

## Inference

### Download pre-trained models

Make models subdirectory
```
mkdir models
```

Feature extraction (LoVIT):
- Download the pre-trained encoder (Trained_VIT_Cholec80.pth) from https://github.com/MRUIL/LoViT
- Convert the checkpoint such that torch.load works (loads the pre-trained weights into the vit_base_patch16_224 architecture):

```
mkdir models
python3 convert_lovit_checkpoints.py \
   --src Trained_VIT_Cholec80.pth \
   --dst models/step_encoder.pth
```

Active surgery detection model:
- Download the pre-trained active surgery detection model `active_surgery_model.pyth` from [Google Drive](https://drive.google.com/file/d/1ksXfQewot5JOo2PlO2mnzYqNAn9X5_Wb/view?usp=drive_link) to `models/active_surgery_model.pyth`

Step model:
- Download the pre-trained step model `step_model.pyth` from [Google Drive](https://drive.google.com/file/d/1hQhFRD-nZulfT5IATKaXq3WQTXWI4fKS/view?usp=sharing) to `models/step_model.pyth`

Make sure the `models` directory looks as follows:
```
models/
|–– active_surgery_model.pyth
|–– step_encoder.pth
|–– step_model.pyth
```

### Prepare the paths
Optional paths to replace in the script `scripts/run_all.sh`:
- features_dir: Directory where features (output of the step encoder) should be saved
- encoder_path: Path to the step_encoder checkpoint
- step_model_path: Path to the step_model checkpoint
- active_surgery_model_path: Path to the active surgery detection model checkpoint
- active_surgery_output_dir: Directory where the active surgery model outputs will be saved
- outputs: Directory where the step model outputs will be saved

### Running the model
Run all the scripts in sequence
```
bash ./scripts/run_inference.sh /path/to/video_id.mp4 /path/to/models \[FEATURE_EXTRACTOR_NAME] [FEATURE_EXTRACTOR_ARCHITECTURE]
```
Note: `[FEATURE_EXTRACTOR_NAME]` and `[FEATURE_EXTRACTOR_ARCHITECTURE]` are what you would pass to --model_name and --model_arch when running `python3 extract_features/extract_tad_feature.py`

This script does the following:
- Extract frame-level features at 1 fps (LoVIT)
- Predicts the start and end of active surgery in the video (Active Surgery Detection Model)
- Predicts the step for each frame with the step encoder (LTContext)
- Post-processes the model outputs


### Model outputs
All step model outputs can be found in `outputs/results/preds/video_name`:
- pred.npy: NumPy array of the predictions
- pred.csv: Formatted CSV of the predictions (step names) with the columns: clip_id,annotation_type,label,duration,fps,start_time,end_time
- pred.png: Image of the temporal segmentations
- legend.png: Legend corresponding to pred.png

All outputs related to active surgery detection are found in `outputs_active_surgery/results/preds/video_name`
- pred.csv: Formatted CSV of the predictions (background vs. active surgery) with the columns: clip_id,annotation_type,label,duration,fps,start_time,end_time

## Training

Training has three stages: (1) extract per-frame features for every video, (2) train the
step segmentation model on those features, and (3) train the active surgery detection model.
Both models are [LTContext](https://github.com/sabaghzadeh/LTContext)-style temporal models
that operate on the pre-extracted features, so feature extraction only has to be done once.

### Dataset layout

All training stages are driven by a **master CSV** that describes your dataset, with one row
per video and the following columns:

| Column      | Description                                                                 |
|-------------|-----------------------------------------------------------------------------|
| `video_id`  | Unique id for the video (also the feature filename stem, `<video_id>.npy`).  |
| `mode`      | Which split the video belongs to: `train`, `validation`, or `test`.         |
| `feat_path` | Absolute path to the extracted feature file (`<features_dir>/<video_id>.npy`).|
| `gt_path`   | Absolute path to the ground-truth annotation CSV for this video (see below). |
| `fps`       | Frame rate the annotations are defined at (usually `1`).                     |

Each **ground-truth CSV** (`gt_path`) lists the labeled segments of one video with at least:

| Column     | Description                                                                  |
|------------|------------------------------------------------------------------------------|
| `label`    | The class name for the segment (must appear in the mapping file).            |
| `duration` | The frame range of the segment as `"<start>-<end>"` (inclusive start frame). |

A **mapping file** maps class names to integer indices, one `"<index> <name>"` per line.
The repository ships two examples under `step_model/mapping_files/`:
[`chole_steps.txt`](step_model/mapping_files/chole_steps.txt) (5 step classes) and
[`active_surgery.txt`](step_model/mapping_files/active_surgery.txt) (background /
activeSurgery). Create your own mapping file to match your dataset's classes.

### Extract framewise features for all videos

Feature extraction (LoVIT):
- We use LoVIT as our pre-trained image feature extractor. Download the pre-trained encoder (Trained_VIT_Cholec80.pth) from https://github.com/MRUIL/LoViT
- Convert the checkpoint such that torch.load works (loads the pre-trained weights into the vit_base_patch16_224 architecture):

```
mkdir models
python3 convert_lovit_checkpoints.py \
   --src Trained_VIT_Cholec80.pth \
   --dst models/step_encoder.pth
```

Extract one `[N, C]` feature array per video (`N` = number of sampled frames) with a frozen
image encoder. This is done by the standalone tool in
[`extract_features/`](extract_features/) — see [extract_features/README.md](extract_features/README.md)
for the full instructions, encoder options, and SLURM helpers.

In brief:

```bash
conda env create -f extract_features/environment.yaml
conda activate feature_extract

python3 extract_features/extract_tad_feature.py \
    --model_name LoVIT \
    --model_arch vit_base_patch16_224 \
    --ckpt_path models/step_encoder.pth \
    --video_lst_csv /path/to/videos.csv \
    --save_path features \
    --video_lst "vid001,vid002,vid003" \
    --batch_size 1 \
    --fps 1
```

This writes `features/<video_id>.npy`. Use the resulting paths as the `feat_path` column in
your master CSV. The released checkpoints were trained on LoVIT features sampled at 1 fps; if
you use a different encoder, update `MODEL.INPUT_DIM` in the configs to match the feature
dimension `C`.

### Train the step model

1. Build a master CSV and a step mapping file as described in [Dataset layout](#dataset-layout).
2. Edit the user-configurable variables at the top of
   [`scripts/train_step_model.sh`](scripts/train_step_model.sh):
   - `master_csv` — path to your master CSV
   - `mapping_file` — path to your step mapping file (relative to `step_model/`)
   - `num_classes` — number of step classes (must match the mapping file)
   - `output_dir` / `exp_prefix` — where outputs are written and a tag for this run
3. Launch training (trains, then evaluates on the `test` split):

   ```bash
   bash ./scripts/train_step_model.sh
   ```

Architecture and optimization settings live in
[`step_model/configs/LTContext_lovit.yaml`](step_model/configs/LTContext_lovit.yaml)
(learning rate, number of epochs, LTContext layers/stages, etc.); the script overrides the
dataset-specific values on the command line. Checkpoints (including `best_checkpoint.pyth`)
and per-epoch metrics are written under `output_dir`. To re-evaluate a saved checkpoint
without re-training, uncomment the optional block at the bottom of the script and set `ckpt`.

### Train the active surgery model

The active surgery model is a binary temporal model that predicts the start and end of active
surgery (background vs. activeSurgery). Its ground-truth CSV needs a single `activeSurgery`
segment whose `duration` marks the active-surgery `start-end`; all other frames are treated
as background.

1. Build a master CSV (same columns as above) and use the binary mapping file
   [`step_model/mapping_files/active_surgery.txt`](step_model/mapping_files/active_surgery.txt).
2. Edit the user-configurable variables at the top of
   [`scripts/train_active_surgery.sh`](scripts/train_active_surgery.sh)
   (`master_csv`, `mapping_file`, `output_dir`, `exp_prefix`).
3. Launch training:

   ```bash
   bash ./scripts/train_active_surgery.sh
   ```

Settings live in
[`step_model/configs/LTContext_lovit_active_surgery.yaml`](step_model/configs/LTContext_lovit_active_surgery.yaml).
Checkpoints and metrics are written under `output_dir`.