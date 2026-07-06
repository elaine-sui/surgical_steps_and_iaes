"""Train the thermal injury detector (an mmdetection model) from a config file.

Adapted from mmdetection's `tools/train.py
<https://github.com/open-mmlab/mmdetection/blob/main/tools/train.py>`_, since
training only requires the generic mmengine `Runner` plus the config files
under `configs/`. The model architecture (Deformable DETR), dataset, and
schedule are all defined declaratively in the config rather than in this
script, so `mmdet` only needs to be pip-installed (e.g. `mim install mmdet`).

Example:
    python tools/train_detector.py \\
        configs/deformable_detr/deformable-detr-refine-twostage_r50_15xb2-50e_coco_thermal_injury_w_more_aug.py \\
        --work-dir work_dirs/deformable-detr-refine-twostage_r50_15xb2-50e_coco_thermal_injury_w_more_aug \\
        --data-root /path/to/frame_annotations/2024-10-28/Thermal_Injury

`--data-root` is the directory holding the COCO-format annotation files
(training.json / validation.json / testing.json) and is required by the dataset
config.
"""
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(description='Train the thermal injury detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--data-root',
        required=True,
        help='directory holding the COCO-format annotation files '
        '(training.json / validation.json / testing.json). Sets `data_root` '
        'in the dataset config.')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher for multi-GPU/multi-node training')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    # The dataset config reads `data_root` from this env var at parse time, so
    # it must be set before `Config.fromfile` is called below.
    if args.data_root is not None:
        os.environ['THERMAL_INJURY_DATA_ROOT'] = osp.join(args.data_root, '')

    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve training speed.
    setup_cache_size_limit_of_dynamo()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.amp:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.train()


if __name__ == '__main__':
    main()
