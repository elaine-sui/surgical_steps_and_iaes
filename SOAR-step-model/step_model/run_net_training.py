import os
import sys
import argparse
from os.path import join, split, splitext
from yacs.config import CfgNode
from dateutil import tz
import datetime

import ltc.utils.checkpoint as cu
from ltc.config.defaults import get_cfg
import ltc.utils.misc as misc
from ltc.train_net import train
from ltc.test_net import test


def parse_args():
    parser = argparse.ArgumentParser(
        description="Provide the path to config and options. "
                    "See ltc/config/defaults.py for all options"
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/Assembly101/LTContext.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See ltc/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument(
        "--rng_seed",
        help="Random seed",
        default=1234,
        type=int
    )

    parser.add_argument(
        "--cv_split",
        help="Cross validation split",
        default=0,
        type=int,
        choices=range(0, 5)
    )

    parser.add_argument(
        "--num_layers",
        help="Number of layers in model",
        default=None,
        type=int
    )

    parser.add_argument(
        "--num_stages",
        help="Number of layers in model",
        default=None,
        type=int
    )

    parser.add_argument(
        "--test_last",
        action="store_true",
        help='whether to eval on the last epoch'
    )


    parser.add_argument(
        "--stratify_by_dataset",
        action='store_true'
    )

    parser.add_argument(
        "--active_surgery_pred_path",
        type=str,
        default=None,
        help='where the active surgery model predictions are saved'
    )

    parser.add_argument(
        "--active_surgery",
        action="store_true",
        help='whether this model is the active surgery model'
    )

    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    """

    :param args: arguments including `cfg_file`, and `opts`
    :return:
        config file
    """

    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir
    if hasattr(args, "cv_split"):
        cfg.DATA.CV_SPLIT_NUM = args.cv_split
    
    if cfg.MODEL.NAME == 'ltc':
        if args.num_layers:
            cfg.MODEL.LTC.NUM_LAYERS = args.num_layers
        
        if args.num_stages:
            cfg.MODEL.LTC.NUM_STAGES = args.num_stages
    else:
        if args.num_layers:
            cfg.MODEL.TCN.NUM_LAYERS = args.num_layers

        if args.num_stages:
            cfg.MODEL.TCN.NUM_STAGES = args.num_stages

    cfg.CONFIG_FILE = args.cfg_file
    cfg_file_name = splitext(split(args.cfg_file)[1])[0]
    cfg.OUTPUT_DIR = join(cfg.OUTPUT_DIR, cfg_file_name)
    cfg.stratify_by_dataset = args.stratify_by_dataset
    return cfg


def prep_output_paths(cfg: CfgNode):
    """
    Preparing the path for tensorboard summary, config log and checkpoints
    :param cfg:
    :return:
    """
    if cfg.TRAIN.ENABLE:
        summary_path = misc.check_path(join(cfg.OUTPUT_DIR, "summary"))

        now = datetime.datetime.now(tz.tzlocal())
        timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")

        exp_name = cfg.EXP_NAME # cfg.MODEL.ERROR_DETECTION if cfg.MODEL.ERROR_DETECTION is not None else 'step'

        exp_name += f"_{cfg.MODEL.NAME}_{cfg.TRAIN.DATASET}_{cfg.DATA.FEATURES}_feats"
        if cfg.MODEL.NAME == 'ltc':
            exp_name += f"_{cfg.MODEL.LTC.NUM_LAYERS}_layers_{cfg.MODEL.LTC.NUM_STAGES}_stages"
        else:
            exp_name += f"_{cfg.MODEL.TCN.NUM_LAYERS}_layers_{cfg.MODEL.TCN.NUM_STAGES}_stages"
        exp_name += f"_seed_{cfg.RNG_SEED}/split_{cfg.DATA.CV_SPLIT_NUM}/{timestamp}"

        cfg.EXPR_NUM = exp_name
        # if cfg.TRAIN.AUTO_RESUME and cfg.TRAIN.RESUME_EXPR_NUM > 0:
        #     cfg.EXPR_NUM = cfg.TRAIN.RESUME_EXPR_NUM
        cfg.SUMMARY_PATH = misc.check_path(join(summary_path, "{}".format(cfg.EXPR_NUM)))
        cfg.CONFIG_LOG_PATH = misc.check_path(
            join(cfg.OUTPUT_DIR, "config", "{}".format(cfg.EXPR_NUM))
        )
        # Create the checkpoint dir.
        cu.make_checkpoint_dir(cfg.OUTPUT_DIR, cfg.EXPR_NUM)
    if cfg.TEST.ENABLE:
        os.makedirs(cfg.TEST.SAVE_RESULT_PATH, exist_ok=True)

        if not cfg.TRAIN.ENABLE:
            cfg.EXPR_NUM = '/'.join(cfg.TEST.CHECKPOINT_PATH.split('/')[-4:-1])


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    prep_output_paths(cfg)
    if cfg.TRAIN.ENABLE:
        train(cfg=cfg)

    if cfg.TEST.CHECKPOINT_PATH == "":
        cfg.TEST.CHECKPOINT_PATH = os.path.join(cu.get_checkpoint_dir(cfg.OUTPUT_DIR.replace('results', 'checkpoints'), cfg.EXPR_NUM), "best_checkpoint.pyth")
    
    if cfg.TEST.ENABLE:
        test(cfg)


if __name__ == "__main__":
    main()
