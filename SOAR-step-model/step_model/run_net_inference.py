import sys
import argparse
import datetime
from dateutil import tz
from yacs.config import CfgNode

from ltc.config.defaults import get_cfg
from ltc.test_net_inference import test

def parse_args():
    parser = argparse.ArgumentParser(
        description="Provide the path to config and options. "
                    "See ltc/config/defaults.py for all options"
    )

    parser.add_argument(
        "--video_feats_path",
        help='path to video features',
        default=None,
        type=str,
    )

    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/LTContext_lovit.yaml",
        type=str,
    )

    parser.add_argument(
        "opts",
        help="See ltc/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument(
        "--save_preds",
        action="store_true",
        help='whether to save test preds'
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help='where to save test preds'
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
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    cfg.CONFIG_FILE = args.cfg_file

    cfg.TEST.SAVE_PREDICTIONS = args.save_preds

    return cfg


def prep_output_paths(cfg: CfgNode):
    """
    Preparing the path for tensorboard summary, config log and checkpoints
    :param cfg:
    :return:
    """
    now = datetime.datetime.now(tz.tzlocal())
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    cfg.EXPR_NUM = timestamp


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    prep_output_paths(cfg)
    test(cfg, video_feats_path=args.video_feats_path, active_surgery_pred_path=args.active_surgery_pred_path, active_surgery=args.active_surgery)

if __name__ == "__main__":
    main()
