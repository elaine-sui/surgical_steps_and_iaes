"""Evaluate a trained thermal injury detector on the test split.

Adapted from mmdetection's `tools/test.py
<https://github.com/open-mmlab/mmdetection/blob/main/tools/test.py>`_. Runs the
model against `test_dataloader` (the `testing.json` COCO split defined in
`configs/_base_/datasets/coco_detection_detr_thermal_injury_w_more_aug.py`) and
reports COCO bbox mAP. Pass `--out preds.pkl` to dump per-frame predictions for
offline frame-presence metrics (see `tools/compute_frame_presence_metrics.py`).

`mmdet` only needs to be pip-installed (e.g. `mim install mmdet`).

Example:
    python tools/test_detector.py \\
        configs/deformable_detr/deformable-detr-refine-twostage_r50_15xb2-50e_coco_thermal_injury_w_more_aug.py \\
        work_dirs/deformable-detr-refine-twostage_r50_15xb2-50e_coco_thermal_injury_w_more_aug/epoch_25.pth \\
        --data-root /path/to/frame_annotations/2024-10-28/Thermal_Injury \\
        --out work_dirs/deformable-detr-refine-twostage_r50_15xb2-50e_coco_thermal_injury_w_more_aug/test_preds.pkl

`--data-root` is the directory holding the COCO-format annotation files
(training.json / validation.json / testing.json) and is required by the dataset
config.
"""
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the thermal injury detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file to evaluate')
    parser.add_argument(
        '--data-root',
        required=True,
        help='directory holding the COCO-format annotation files '
        '(training.json / validation.json / testing.json). Sets `data_root` '
        'in the dataset config.')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results in a popup window')
    parser.add_argument(
        '--show-dir',
        help='directory where painted prediction images will be saved. '
        'If specified, it will be automatically saved '
        'to work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
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
        help='job launcher for multi-GPU/multi-node evaluation')
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

    # Reduce the number of repeated compilations and improve testing speed.
    setup_cache_size_limit_of_dynamo()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    assert args.checkpoint, "Need to load checkpoint to test!"
    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpDetResults(out_file_path=args.out))

    runner.test()


if __name__ == '__main__':
    main()
