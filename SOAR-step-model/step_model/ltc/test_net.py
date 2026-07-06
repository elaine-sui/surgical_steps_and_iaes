import os
from os.path import join
from tqdm import tqdm
from yacs.config import CfgNode
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from ltc.utils.meters import ValMeter
import ltc.utils.checkpoint as cu
import ltc.utils.misc as misc
from ltc.dataset import loader
from ltc.model import model_builder
from ltc.utils.metrics import calculate_metrics, calculate_metrics_multilabel, compute_macro_f1, compute_phase_precision, compute_phase_recall

import ltc.utils.logging as logging
from ltc.utils.postprocess import post_process

logger = logging.get_logger(__name__)


@torch.no_grad()
def eval_model(
        val_loader: DataLoader,
        model: nn.Module,
        device: torch.device.type,
        cfg: CfgNode,
        visualize = False,
        threshold: float = 0.5):
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

    save_path = join(cfg.OUTPUT_DIR, "results", cfg.EXPR_NUM)
    logger.info(save_path)
    os.makedirs(save_path, exist_ok=True)

    multilabel = 'multilabel' in cfg.MODEL.LOSS_FUNC or 'steps_and_tasks' in cfg.MODEL.LOSS_FUNC
    num_classes = cfg.MODEL.NUM_CLASSES
    if multilabel:
        test_metrics = {'video_name': []}
        for class_idx in range(num_classes):
            test_metrics[f'MoF_class{class_idx}'] = []
            test_metrics[f'Edit_class{class_idx}'] = []
            test_metrics[f'VideoF1_class{class_idx}'] = []
            test_metrics[f'confusion_matrix_class{class_idx}'] = np.zeros((2, 2)) # binary per class
    else:
        test_metrics = {"video_name": [],
                    "MoF": [],
                    "Edit": [],
                    "F1@10": [],
                    "F1@25": [],
                    "F1@50": [],
                    "VideoF1": [],
                    "confusion_matrix": np.zeros((num_classes, num_classes))
                    }

    ignored_class_idx = cfg.TEST.IGNORED_CLASSES + [cfg.MODEL.PAD_IGNORE_IDX]
    logger.info(f"Ignored class idxs: {ignored_class_idx}")

    if num_classes == 1: # binary (foreground and background)
        num_classes += 1

    visualization_samples = []
    all_logits = []
    all_targets = []
    all_video_names = []
    for batch_dict in tqdm(val_loader, total=len(val_loader)):
        mb_size = batch_dict["targets"].shape[0]
        assert mb_size == 1, "Validation batch size should be one."

        misc.move_to_device(batch_dict, device)
        logits = model(batch_dict['features'], batch_dict['masks'])
        all_logits.append(logits)

        if multilabel or logits.shape[2] == 1: # binary:
            prediction = misc.prepare_binary_prediction(logits, threshold=threshold)
        else:
            prediction = misc.prepare_prediction(logits)

        target = batch_dict['targets'].cpu()
        all_targets.append(target)
        prediction = prediction.cpu()
        video_name = batch_dict['video_name'][0][0]

        if multilabel:
            calculate_metrics_fn = calculate_metrics_multilabel
        else:
            calculate_metrics_fn = calculate_metrics
        video_metrics = calculate_metrics_fn(target,
                                          prediction,
                                          ignored_class_idx,
                                          num_classes=num_classes)

        test_metrics['video_name'].append(video_name)
        all_video_names.append(video_name)

        for name, score in video_metrics.items():
            if name.startswith('confusion_matrix'):
                test_metrics[name] += score
            elif name != 'Edit':
                score = score * 100
                test_metrics[name].append(score)
            else:
                test_metrics[name].append(score)

        if visualize:
            visualization_samples.append({"target": target.squeeze(0).numpy(),
                                          "pred": prediction.squeeze(0).numpy(),
                                          "video_name": f"{batch_dict['video_name'][0][0]}"})

        if cfg.TEST.SAVE_PREDICTIONS:
            pred = prediction[0].long().numpy()
            base_path = join(save_path, "preds", str(video_name))
            os.makedirs(base_path, exist_ok=True)
            np.save(join(base_path, "pred.npy"), pred)
            np.save(join(base_path, "gt.npy"), target[0].long().numpy())

            print(f"Preds at {base_path}/pred.npy")
            print(f"GT at {base_path}/gt.npy")
    
    if visualize and not multilabel:
        val_meter = ValMeter(len(val_loader), cfg, None)

        filepath = f'visualize_threshold_{threshold:.6f}_training.png'
        val_meter.visualize_prediction_result(vis_data=visualization_samples, cur_epoch=f'threshold_tuning_{threshold}', save_visualization=True, filepath=filepath)
    
    test_metrics_confusion_matrix = {k:v for k,v in test_metrics.items() if k.startswith('confusion_matrix')}
    test_metrics = {k:v for k,v in test_metrics.items() if not k.startswith('confusion_matrix')}
    
    test_res_df = pd.DataFrame(test_metrics)
    test_res_df.round(5).to_csv(join(save_path, "testing_metrics.csv"))

    if multilabel:
        keys = [k for k in test_metrics if k != 'video_name' and not k.startswith('confusion_matrix')]
        mean_metrics = test_res_df[keys].mean()
    else:
        mean_metrics = test_res_df[['F1@10', 'F1@25', 'F1@50', 'Edit', 'MoF', 'VideoF1']].mean()
    
    if len(test_metrics_confusion_matrix) > 1:
        for key, confusion_matrix in test_metrics_confusion_matrix.items():
            if key.startswith('confusion_matrix'):
                class_idx = key.split('class')[1]
                print(confusion_matrix)
                mean_metrics[f'PhasePrecision_class{class_idx}'], _ = compute_phase_precision(confusion_matrix) * 100
                mean_metrics[f'PhaseRecall_class{class_idx}'], _ = compute_phase_recall(confusion_matrix) * 100
                mean_metrics[f'FramewiseF1_class{class_idx}'], _ = compute_macro_f1(confusion_matrix) * 100
    else:
        confusion_matrix = test_metrics_confusion_matrix['confusion_matrix']
        print(confusion_matrix)
        mean_metrics[f'PhasePrecision'], prec_lst = compute_phase_precision(confusion_matrix)
        mean_metrics[f'PhaseRecall'], rec_lst = compute_phase_recall(confusion_matrix)
        mean_metrics[f'FramewiseF1'], f1_lst = compute_macro_f1(confusion_matrix)

        for class_idx, (prec, rec, f1) in enumerate(zip(prec_lst, rec_lst, f1_lst)):
            mean_metrics[f'PhasePrecision_class{class_idx}'] = prec * 100
            mean_metrics[f'PhaseRecall_class{class_idx}'] = rec * 100
            mean_metrics[f'FramewiseF1_class{class_idx}'] = f1 * 100

        mean_metrics[f'PhasePrecision'] *= 100
        mean_metrics[f'PhaseRecall'] *= 100
        mean_metrics[f'FramewiseF1'] *= 100
    logger.info("Testing metric:")
    logging.log_json_stats(mean_metrics, precision=1)

    # Print metrics to console
    print("="*100)
    print("Frame-wise metrics")
    for metric, val in mean_metrics.items():
        print(f"test/{metric}:\t{val}")
    print("="*100)

    # Save metrics
    base_path = join(save_path, "metrics")
    if cfg.DATA.LEAVE_ONE_OUT_TEST_SPLIT is not None:
        base_path = join(base_path, str(video_name))

    os.makedirs(base_path, exist_ok=True)

    metrics_path = join(base_path, 'metrics.csv')
    df = pd.DataFrame([mean_metrics])
    df.to_csv(metrics_path, index=False)
    print(f"Save metrics to {metrics_path}")


    return all_logits, all_targets, all_video_names


def test(cfg: CfgNode, video_feats_path=None):
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
    test_loader, test_dataset = loader.construct_loader(cfg, cfg.TRAIN.EVAL_SPLIT)
    eval_model(test_loader, model, device, cfg, visualize=True)

    # # By data source
    # print("By data source")
    # test_loaders, test_datasets = loader.construct_loaders_by_datasource_groups(cfg, cfg.TRAIN.EVAL_SPLIT)
    # print('+'*80)
    # for test_loader, test_dataset in zip(test_loaders, test_datasets):
    #     print(f"Dataset: {test_dataset._datasource}")
    #     eval_model(test_loader, model, device, cfg)
