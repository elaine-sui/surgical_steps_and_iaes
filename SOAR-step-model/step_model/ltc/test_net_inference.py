import os
from os.path import join
from tqdm import tqdm
from yacs.config import CfgNode

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader


import ltc.utils.checkpoint as cu
import ltc.utils.misc as misc
from ltc.dataset import loader
from ltc.model import model_builder

import ltc.utils.logging as logging
from ltc.utils.postprocess import post_process

logger = logging.get_logger(__name__)


@torch.no_grad()
def eval_model(
        val_loader: DataLoader,
        model: nn.Module,
        device: torch.device.type,
        cfg: CfgNode,
        active_surgery_pred_path: str,
        active_surgery: bool
    ):
    """
    Evaluate the model on the val set.
    :param val_loader: data loader to provide validation data.
    :param model: model to evaluate the performance.
    :param device: device to use (cuda or cpu)
    :param cfg:
    :return:
    """
    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    logger.info(f"Testing the trained model.")

    save_path = join(cfg.OUTPUT_DIR, "results")
    logger.info(save_path)
    os.makedirs(save_path, exist_ok=True)

    ignored_class_idx = cfg.TEST.IGNORED_CLASSES + [cfg.MODEL.PAD_IGNORE_IDX]
    logger.info(f"Ignored class idxs: {ignored_class_idx}")

    # Note: only a single item in the val_loader
    for batch_dict in tqdm(val_loader, total=len(val_loader)):
        # mb_size = batch_dict["targets"].shape[0]
        # assert mb_size == 1, "Validation batch size should be one."

        misc.move_to_device(batch_dict, device)

        logits = model(batch_dict['features'], batch_dict['masks'])

        prediction = misc.prepare_prediction(logits)

        prediction = prediction.cpu()
        video_name = batch_dict['video_name'][0][0]

        if cfg.TEST.SAVE_PREDICTIONS:
            base_path = join(save_path, "preds", video_name)
            os.makedirs(base_path, exist_ok=True)
            pred = prediction[0].long().numpy()
            np.save(join(base_path, "pred.npy"), pred)
            print(f"Preds at {base_path}/pred.npy. Shape: {pred.shape}")

            # Save CSV
            video_info = {'video_name': video_name}
            post_process(cfg, pred, video_info, base_path, active_surgery_pred_path, active_surgery)


def test(cfg: CfgNode, video_feats_path=None, active_surgery_pred_path=None, active_surgery=False):
    """
    Train an action segmentation model for many epochs on train set and validate it on val set
    :param cfg: config file. Details can be found in ltc/config/defaults.py
    :return:
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    logging.setup_logging(cfg.OUTPUT_DIR, cfg.EXPR_NUM)

    model = model_builder.build_model(cfg)
    logger.info(f"Number of params: {misc.params_to_string(misc.params_count(model))}")

    if cfg.NUM_GPUS > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Transfer the model to device(s)
    model = model.to(device)

    checkpoint_path = cfg.TEST.CHECKPOINT_PATH

    print(f"Loading checkpoint {checkpoint_path}")
    cu.load_model(
        checkpoint_path,
        model=model,
        num_gpus=cfg.NUM_GPUS,
    )
    test_loader = loader.construct_loader_inference(cfg, video_feats_path, active_surgery_pred_path)
    eval_model(test_loader, model, device, cfg, active_surgery_pred_path, active_surgery)