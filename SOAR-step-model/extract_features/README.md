# Frame-wise Feature Extraction

Extract frame-wise (per-frame) image features from full-length surgical videos for
downstream temporal action detection / step recognition. Each video is decoded with
[decord](https://github.com/dmlc/decord), sub-sampled to a target FPS, passed through a
frozen image encoder, and saved as a single `[N, C]` NumPy array (`N` = number of
sampled frames, `C` = feature dimension).

Supported encoders (`--model_name`):

| `model_name`    | `model_arch` example      | Notes                                                     |
|-----------------|---------------------------|----------------------------------------------------------|
| `CLIP`          | `ViT-L/14`, `ViT-B/32`    | OpenAI CLIP visual encoder (fp16).                        |
| `DINOv2`        | `dinov2_vitl14_reg`       | Loaded via `torch.hub` (`facebookresearch/dinov2`).      |
| `DINOv3-ViTB16` | *(arch fixed by name)*    | HF `facebook/dinov3-vitb16-pretrain-lvd1689m`.           |
| `DINOv3-ViTL16` | *(arch fixed by name)*    | HF `facebook/dinov3-vitl16-pretrain-lvd1689m`.           |
| `LoVIT`         | `vit_base_patch16_224`    | `timm` backbone; requires `--ckpt_path` to a checkpoint. |

## 1. Environment setup

A CUDA-capable GPU is required (the model is moved to `.cuda()`).

```bash
conda env create -f environment.yaml
conda activate feature_extract
```

For long videos with truncated streams, decord may need a larger retry limit:

```bash
export DECORD_EOF_RETRY_MAX=204800
```

## 2. Inputs

You point the extractor at videos in one of two ways:

1. **A directory of `.mp4` files** via `--video_dir`. File `XYZ.mp4` is treated as
   `video_id = XYZ`.
2. **A CSV manifest** via `--video_lst_csv`. The CSV must contain the columns:
   - `video_id` — unique id (also the output filename stem)
   - `video_path` — absolute path to the video file
   - `fps` — native frame rate of the video

   The extractor down-samples to the target `--fps` by taking every
   `ceil(native_fps / target_fps)`-th frame.

`--video_lst` is a comma-separated list of the `video_id`s to actually process in a given
run (this is what enables sharding across many jobs — see below).

## 3. Running a single extraction

Direct invocation of the extractor:

```bash
python3 extract_tad_feature.py \
    --model_name DINOv3-ViTB16 \
    --video_lst_csv /path/to/master.csv \
    --save_path features/DINOv3-ViTB16/full_videos \
    --video_lst "vid001,vid002,vid003" \
    --batch_size 16 \
    --fps 1
```

CLIP example (optionally downloads weights into `--download_root`):

```bash
python3 extract_tad_feature.py \
    --model_name CLIP \
    --model_arch ViT-L/14 \
    --download_root /path/to/pretrained_models \
    --video_lst_csv /path/to/master.csv \
    --save_path features/clip_ViT-L-14 \
    --video_lst "vid001,vid002" \
    --batch_size 50
```

LoVIT example (requires a checkpoint):

```bash
python3 extract_tad_feature.py \
    --model_name LoVIT \
    --model_arch vit_base_patch16_224 \
    --ckpt_path /path/to/lovit_encoder.pth \
    --video_lst_csv /path/to/master.csv \
    --save_path features/appy_lovit_feats \
    --video_lst "vid001" \
    --batch_size 1
```

### Key arguments

| Argument             | Default      | Description                                                       |
|----------------------|--------------|-------------------------------------------------------------------|
| `--model_name`       | `CLIP`       | Encoder family (see table above).                                |
| `--model_arch`       | `ViT-B/32`   | Architecture string for CLIP / DINOv2 / LoVIT.                   |
| `--download_root`    | `None`       | Cache dir for CLIP / hub weights.                                |
| `--ckpt_path`        | `None`       | Checkpoint path (required for `LoVIT`).                          |
| `--video_dir`        | `None`       | Directory of `.mp4`s (alternative to `--video_lst_csv`).        |
| `--video_lst_csv`    | `None`       | CSV manifest with `video_id,video_path,fps`.                    |
| `--video_lst`        | —            | Comma-separated `video_id`s to process this run.                |
| `--save_path`        | —            | Output directory for the `.npy` files.                          |
| `--batch_size`       | `16`         | Frames processed per forward pass.                              |
| `--fps`              | `10`         | Target sampling rate (frames/sec) after down-sampling.         |
| `--local_center_crop`| off          | Skip the resize step and center-crop only (CLIP/LoVIT).        |

Output: one file per video at `<save_path>/<video_id>.npy`, shape `[N, C]`. Videos whose
output already exists are skipped, and corrupted/undecodable videos are skipped with a
warning.

## 4. Large-scale extraction (SLURM)

For datasets with many videos, `create_scripts_for_feature_extraction.py` generates a set
of SLURM `sbatch` scripts that shard the video list into batches and a `queue_all.sh` that
submits them all. It also automatically excludes videos that have already been extracted
in `--output_dir`.

```bash
python3 create_scripts_for_feature_extraction.py \
    --model DINOv3-ViTB16 \
    --video_lst_csv /path/to/master.csv \
    --output_dir features/DINOv3-ViTB16/full_videos \
    --sbatch_scripts_dir scripts/sbatch_scripts/extract_features_dinov3/full_videos \
    --batch_size 5 \
    --fps 1
```

This writes per-shard scripts to
`<sbatch_scripts_dir>/<model>/extract{0,batch,2*batch,...}.sbatch` plus a
`queue_all.sh`. Submit everything with:

```bash
bash scripts/sbatch_scripts/extract_features_dinov3/full_videos/DINOv3-ViTB16/queue_all.sh
```

> The SLURM header, conda activation, and account/partition in
> `create_scripts_for_feature_extraction.py` (`GLOBAL_CROP_SBATCH_TEMPLATE`) are
> configured for a specific cluster — edit them to match your environment.

Ready-to-adapt launch examples per encoder live under
[scripts/extract_tad_features/](scripts/extract_tad_features/).
