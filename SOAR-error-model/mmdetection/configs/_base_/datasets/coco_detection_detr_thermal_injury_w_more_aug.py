# dataset settings
import os

# `data_root` is the directory holding the COCO-format annotation files
# (training.json / validation.json / testing.json). It is required for training
# and evaluation (the train/test tools set THERMAL_INJURY_DATA_ROOT from their
# required `--data-root` argument and enforce its presence), but not for
# inference: `run_inference.py` only loads the model from this config and builds
# its own dataloader, so the train/val/test dataloaders below are never
# instantiated. Default to '' so the config parses cleanly at inference time.
dataset_type = 'CocoDataset'
data_root = os.environ.get('THERMAL_INJURY_DATA_ROOT', '')

backend_args = None

metainfo = {
    'classes': ('thermal_injury'),
    'palette': [(255, 255, 100)]
}

rand_aug_surg = [
        [dict(type='ShearX', level=8)],
        [dict(type='ShearY', level=8)],
        [dict(type='Rotate', level=8)],
        [dict(type='TranslateX', level=8)],
        [dict(type='TranslateY', level=8)],
        [dict(type='AutoContrast', level=8)],
        [dict(type='Equalize', level=8)],
        [dict(type='Contrast', level=8)],
        [dict(type='Color', level=8)],
        [dict(type='Brightness', level=8)],
        [dict(type='Sharpness', level=8)],
]

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(399, 224), keep_ratio=True),
    # dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
            type='RandAugment',
            aug_space=rand_aug_surg,
        ),
    dict(
            type='Color',
            min_mag = 0.6,
            max_mag = 1.4,
        ),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(399, 224), keep_ratio=True),
    # dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'training.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=metainfo,
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'validation.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'validation.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    backend_args=backend_args)
# test_evaluator = val_evaluator

# inference on test dataset
test_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'testing.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline))
test_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    format_only=False,
    classwise=True,
    ann_file=data_root + 'testing.json',
    outfile_prefix='./work_dirs/coco_detection_thermal_injury/test')
