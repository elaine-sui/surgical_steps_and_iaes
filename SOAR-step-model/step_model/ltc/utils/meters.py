"""Meters."""

from typing import List
from collections import deque

import torch
import numpy as np
from yacs.config import CfgNode
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import matplotlib.pyplot as plt

from ltc.utils.metrics import calculate_metrics, compute_macro_f1, calculate_metrics_framewise, calculate_metrics_multilabel
import ltc.utils.misc as misc
import ltc.utils.plot_utils as plot_utils

import ltc.utils.logging as logging
logger = logging.get_logger(__name__)

import wandb
import ltc.utils.dist_utils as dist_utils
import os
import pickle


class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size: int):
        """

        :param window_size:
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

        self.fmt = "{median:.4f} ({global_avg:.4f})"

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value: float, n: int):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        if self.count == 0:
            return None
        return self.total / self.count
    
    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        Code from Llama repo
        """
        if not dist_utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    def __str__(self):
        """
        Code adapted from Llama repo
        """
        return self.fmt.format(
            median=self.get_win_median(),
            avg=self.get_win_avg(),
            global_avg=self.get_global_avg()
        )


class TrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self,
                 epoch_iters: int,
                 cfg: CfgNode,
                 global_step: int,
                 writer: SummaryWriter = None,
                 temp_eval_dir: str = 'temp_out'):
        """

        :param epoch_iters: the overall number of iterations of one epoch
        :param cfg:
        :param global_step:
        :param writer:
        """
        self._cfg = cfg
        self.num_classes = cfg.MODEL.NUM_CLASSES

        if isinstance(self.num_classes, list):
            self.num_classes = sum(self.num_classes)

        if self.num_classes == 1:
            self.num_classes += 1

        self.writer = writer
        self.epoch_iters = epoch_iters
        self.delimiter = "  "
        self.temp_eval_dir = temp_eval_dir

        keys = ['loss']
        loss_functions = self._cfg.MODEL.LOSS_FUNC.split('_')
        if self._cfg.MODEL.LOSS_FUNC == 'ce_mse_steps_and_tasks':
            keys += ['loss_ce_step', 'loss_ce_task', 'loss_mse_step', 'loss_mse_task']
        else:
            if 'ce' in loss_functions or 'bce' in loss_functions:
                keys += ['loss_ce']
            if 'mse' in loss_functions:
                keys += ['loss_mse']
            if 'focal' in loss_functions:
                keys += ['loss_f']
            if 'uncertainty' in loss_functions:
                keys += ['loss_u']
            if 'dice' in loss_functions:
                keys += ['loss_dice']

        self.loss_dict = {k:ScalarMeter(cfg.LOG_PERIOD) for k in keys}
        self.loss_sums = {k:0.0 for k in keys}

        # self.loss_dict = {
        #     "loss": ScalarMeter(cfg.LOG_PERIOD),
        #     "loss_ce": ScalarMeter(cfg.LOG_PERIOD),
        #     "loss_mse": ScalarMeter(cfg.LOG_PERIOD)
        # }

        # self.loss_sums = {
        #     "loss": 0.0,
        #     "loss_ce": 0.0,
        #     "loss_mse": 0.0
        # }

        self._iter_log_window = cfg.LOG_PERIOD
        self._ignore_idx = cfg.MODEL.PAD_IGNORE_IDX
        if isinstance(cfg.MODEL.PAD_IGNORE_IDX, int):
            self._ignore_idx = [cfg.MODEL.PAD_IGNORE_IDX]
        self._ignore_idx = self._ignore_idx + cfg.TRAIN.EVAL_IGNORE_LABELS
        
        if dist_utils.is_main_process():
            logger.info(f"Ignored label idxs: {self._ignore_idx}")


        self.multilabel = 'multilabel' in cfg.MODEL.LOSS_FUNC or 'steps_and_tasks' in cfg.MODEL.LOSS_FUNC

        if self.multilabel:
            self.metrics_dict = {}
            for class_idx in range(self.num_classes):
                self.metrics_dict[f'MoF_class{class_idx}'] = []
                self.metrics_dict[f'Edit_class{class_idx}'] = []
                self.metrics_dict[f'VideoF1_class{class_idx}'] = []
                self.metrics_dict[f'confusion_matrix_class{class_idx}'] = np.zeros((2, 2)) # binary per class
        else:
            self.metrics_dict = {
                "MoF": [],
                "Edit": [],
                "F1@10": [],
                "F1@25": [],
                "F1@50": [],
                "VideoF1": [],
                "confusion_matrix": np.zeros((self.num_classes, self.num_classes))
            }

        self.num_samples = 0
        self.lr = None
        self.global_iter = global_step
        self.rank = dist_utils.get_rank()

        if self._cfg.NUM_GPUS > 1:
            self.suffix = f"_rank{self.rank}"
        else:
            self.suffix = "_full"

        self.multilabel = 'multilabel' in cfg.MODEL.LOSS_FUNC or 'steps_and_tasks' in cfg.MODEL.LOSS_FUNC

        if self.multilabel:
            self.calculate_metrics_fn = calculate_metrics_multilabel
        else:
            self.calculate_metrics_fn = calculate_metrics

    def reset(self):
        """
        Reset the Meter.
        """
        for loss in self.loss_dict.values():
            loss.reset()
        for name in self.loss_sums.keys():
            self.loss_sums[name] = 0.0
        self.metrics_dict = {name: [] for name in self.metrics_dict.keys()}

        if self.multilabel:
            for class_idx in range(self.num_classes):
                self.metrics_dict[f'confusion_matrix_class{class_idx}'] = np.zeros((2, 2)) # binary per class
        else:
            self.metrics_dict["confusion_matrix"] = np.zeros((self.num_classes, self.num_classes))
        self.lr = None
        self.num_samples = 0

    def update_stats(self,
                     target: torch.Tensor,
                     prediction: torch.Tensor,
                     loss_dict: dict,
                     lr: float,
                     ):
        """

        :param target:
        :param prediction:
        :param loss_dict:
        :param lr:
        :return:
        """
        mb_size = target.shape[0]
        target = target.detach().cpu()
        prediction = prediction.detach().cpu()
        video_metrics = self.calculate_metrics_fn(target, prediction, self._ignore_idx, num_classes=self.num_classes)
        for name, score in video_metrics.items():
            if name.startswith('confusion_matrix'):
                self.metrics_dict[name] += score
            else:
                self.metrics_dict[name].append(score)

        if self.writer:
            for name, metric_val in video_metrics.items():
                if name.startswith('confusion_matrix'):
                    continue
                self.writer.add_scalar(f"4-Train_Metric/{name}", metric_val, global_step=self.global_iter)

        for name, loss in loss_dict.items():
            if not isinstance(loss, float):
                loss = loss.item()
            if name not in self.loss_dict:
                self.loss_dict[name] = ScalarMeter(self._cfg.LOG_PERIOD)
                self.loss_sums[name] = 0.0

            self.loss_dict[name].add_value(loss, mb_size)
            self.loss_sums[name] += loss * mb_size
            if self.writer:
                self.writer.add_scalar(f"2-Loss/train/{name}",
                                       loss,
                                       global_step=self.global_iter)
        self.lr = lr
        self.num_samples += mb_size
        self.global_iter += 1

    def synchronize_between_processes(self):
        """
        Code adapted from Llama repo
        """
        # print("Sync loss dict for train meter")
        for meter in self.loss_dict.values():
            meter.synchronize_between_processes()
    
    def __str__(self):
        loss_str = []
        for name, meter in self.loss_dict.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        Log the stats of the current iteration.

        :param cur_epoch: the number of current epoch.
        :param cur_iter: the number of current iteration.
        :return:
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "lr": f"{self.lr:.6f}",
            "mem": int(np.ceil(mem_usage)),
        }

        wandb_stats = dict()

        for name, metric_list in self.metrics_dict.items():
            if name.startswith('confusion_matrix'):
                continue
            metric_val = np.median(metric_list[-self._iter_log_window:])
            stats[name] = f"{metric_val:.4f}"
            wandb_stats[name + self.suffix] = metric_val
        for name, loss in self.loss_dict.items():
            stats[name] = f"{loss.get_win_median():.5f}"
            wandb_stats[name + self.suffix] = loss.get_win_median()

        wandb_stats = {**{f"train/{k}_step":v for k,v in wandb_stats.items()}}
        
        if dist_utils.is_main_process():
            logging.log_json_stats(stats)
            wandb.log(wandb_stats, step=self.global_iter)

    def save_metrics(self, cur_epoch):
        filepath = os.path.join(self.temp_eval_dir, f'epoch{cur_epoch}{self.suffix}.pkl')

        with open(filepath, 'wb') as f:
            pickle.dump(self.metrics_dict, f)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the end of epochs stats.

        :param cur_epoch:
        :return:
        """
        mem_usage = misc.gpu_mem_usage()
        avg_loss = {}
        wandb_stats = dict()

        for name, loss in self.loss_dict.items():
            if loss.count == 0:
                continue
            avg_loss_ = loss.get_global_avg()
            avg_loss[name] = np.round(avg_loss_, decimals=4)
            wandb_stats[name + self.suffix] = avg_loss_

        # for name, loss in self.loss_sums.items():
        #     avg_loss[name] = np.round(loss / self.num_samples, decimals=4)
        #     wandb_stats[name] = loss / self.num_samples

        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "loss": avg_loss,
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        stats.update(avg_loss)

        if self._cfg.NUM_GPUS > 1:
            self.save_metrics(cur_epoch)
        
        confusion_matrix = {k:v for k,v in self.metrics_dict.items() if 'confusion_matrix' in k}
        for name, metric_list in self.metrics_dict.items():
            if len(metric_list) == 0:
                continue

            if name.startswith('confusion_matrix'):
                continue
            else:
                metric_mean = np.mean(metric_list)
                stats[name] = metric_mean
            wandb_stats[name + self.suffix] = metric_mean
            if self.writer:
                self.writer.add_scalar(f"3-Avg_Train_Metric/{name}",
                                       metric_mean,
                                       global_step=cur_epoch)
                
        if self.writer:
            for name, loss_val in avg_loss.items():
                self.writer.add_scalar(f"2-Loss/train/avg_{name}", loss_val, global_step=cur_epoch)

        if self.writer:
            self.writer.add_scalar("5-LR/train_epoch/lr", self.lr, cur_epoch)

        if len(confusion_matrix) > 1:
            for cm_key, matrix in confusion_matrix.items():
                class_idx = cm_key.split('class')[1]
                wandb_stats[f'FramewiseF1_class{class_idx}' + self.suffix], _ = compute_macro_f1(matrix)
        else:
            wandb_stats[f'FramewiseF1' + self.suffix], per_class_f1 = compute_macro_f1(confusion_matrix['confusion_matrix'])
            for class_idx, f1 in enumerate(per_class_f1):
                wandb_stats[f'FramewiseF1_class{class_idx}' + self.suffix] = f1
        wandb_stats = {**{f"train/{k}_epoch":v for k,v in wandb_stats.items()}, "lr": self.lr, "epoch": cur_epoch}

        if dist_utils.is_main_process():
            logging.log_json_stats(stats)
            wandb.log(wandb_stats)

        return avg_loss['loss']


class ValMeter(object):
    """
    Measures validation stats.
    """
    def __init__(self, max_iter: int, cfg: CfgNode, writer: SummaryWriter = None, temp_eval_dir: str = 'temp_out'):
        """

        :param max_iter:  the max number of iteration of the current epoch.
        :param cfg:
        :param writer:
        """
        self._cfg = cfg
        self.num_classes = cfg.MODEL.NUM_CLASSES

        if self.num_classes == 1:
            self.num_classes += 1

        self.max_iter = max_iter
        self.colors = plot_utils.generate_distinct_colors(n=cfg.MODEL.NUM_CLASSES + 1,
                                                          random_seed=115)
        self._iter_log_window = cfg.LOG_PERIOD

        keys = ['loss']
        loss_functions = self._cfg.MODEL.LOSS_FUNC.split('_')
        if self._cfg.MODEL.LOSS_FUNC == 'ce_mse_steps_and_tasks':
            keys += ['loss_ce_step', 'loss_ce_task', 'loss_mse_step', 'loss_mse_task']
        else:
            if 'ce' in loss_functions or 'bce' in loss_functions:
                keys += ['loss_ce']
                self.avg_loss_type = 'loss_ce'
            if 'mse' in loss_functions:
                keys += ['loss_mse']
            if 'focal' in loss_functions:
                keys += ['loss_f']
                self.avg_loss_type = 'loss_f'
            if 'uncertainty' in loss_functions:
                keys += ['loss_u']
            if 'dice' in loss_functions:
                keys += ['loss_dice']
                self.avg_loss_type = 'loss_dice'

        self.loss_dict = {k:ScalarMeter(cfg.LOG_PERIOD) for k in keys}
        self.loss_sums = {k:0.0 for k in keys}

        # self.loss_dict = {
        #     "loss": ScalarMeter(cfg.LOG_PERIOD),
        #     "loss_ce": ScalarMeter(cfg.LOG_PERIOD),
        #     "loss_mse": ScalarMeter(cfg.LOG_PERIOD)
        # }

        # self.loss_sums = {
        #     "loss": 0.0,
        #     "loss_ce": 0.0,
        #     "loss_mse": 0.0
        # }

        self._ignore_idx = cfg.MODEL.PAD_IGNORE_IDX
        if isinstance(cfg.MODEL.PAD_IGNORE_IDX, int):
            self._ignore_idx = [cfg.MODEL.PAD_IGNORE_IDX]
        self._ignore_idx = self._ignore_idx + cfg.TRAIN.EVAL_IGNORE_LABELS

        if dist_utils.is_main_process():
            logger.info(f"Ignored label idxs: {self._ignore_idx}")

        self.multilabel = 'multilabel' in cfg.MODEL.LOSS_FUNC or 'steps_and_tasks' in cfg.MODEL.LOSS_FUNC

        self.writer = writer

        if self.multilabel:
            self.metrics_dict = {}
            for class_idx in range(self.num_classes):
                self.metrics_dict[f'MoF_class{class_idx}'] = []
                self.metrics_dict[f'Edit_class{class_idx}'] = []
                self.metrics_dict[f'VideoF1_class{class_idx}'] = []
                self.metrics_dict[f'confusion_matrix_class{class_idx}'] = np.zeros((2, 2)) # binary per class
        else:
            self.metrics_dict = {
                "MoF": [],
                "Edit": [],
                "F1@10": [],
                "F1@25": [],
                "F1@50": [],
                "VideoF1": [],
                "confusion_matrix": np.zeros((self.num_classes, self.num_classes))
            }
        self.num_samples = 0
        self.delimiter = "  "
        self.temp_eval_dir = temp_eval_dir
        self.rank = dist_utils.get_rank()

        if self._cfg.NUM_GPUS > 1:
            self.suffix = f"_rank{self.rank}"
        else:
            self.suffix = "_full"
        
        if self.multilabel:
            self.calculate_metrics_fn = calculate_metrics_multilabel
        else:
            self.calculate_metrics_fn = calculate_metrics

    def reset(self):
        """
        Reset the Meter.
        """
        self.metrics_dict = {name: [] for name in self.metrics_dict.keys()}

        if self.multilabel:
            for class_idx in range(self.num_classes):
                self.metrics_dict[f'confusion_matrix_class{class_idx}'] = np.zeros((2, 2)) # binary per class
        else:
            self.metrics_dict["confusion_matrix"] = np.zeros((self.num_classes, self.num_classes))
        for name in self.loss_sums.keys():
            self.loss_sums[name] = 0.0
        self.num_samples = 0

    def update_stats(self,
                     target: torch.Tensor,
                     prediction: torch.Tensor,
                     loss_dict: dict
                     ):

        video_metrics = self.calculate_metrics_fn(target,
                                          prediction,
                                          self._ignore_idx,
                                          num_classes=self.num_classes)
        mb_size = target.shape[0]
        for name, score in video_metrics.items():
            if name.startswith('confusion_matrix'):
                self.metrics_dict[name] += score
            else:
                self.metrics_dict[name].append(score)

        for name, loss in loss_dict.items():
            if not isinstance(loss, float):
                loss = loss.item()
            self.loss_sums[name] += loss * mb_size
            self.loss_dict[name].add_value(loss, mb_size)

        self.num_samples += mb_size
        return video_metrics

    def synchronize_between_processes(self):
        """
        Code adapted from Llama repo
        """
        # print("Sync loss dict for val meter")
        for meter in self.loss_dict.values():
            meter.synchronize_between_processes()
    
    def __str__(self):
        loss_str = []
        for name, meter in self.loss_dict.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)


    def log_iter_stats(self, cur_epoch: int, cur_iter: int):
        """

        :param cur_epoch:
        :param cur_iter:
        :return:
        """

        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "mem": int(np.ceil(mem_usage)),
        }
        
        wandb_stats = dict()
        for name, metric_list in self.metrics_dict.items():
            if len(metric_list) == 0:
                continue

            if name.startswith('confusion_matrix'):
                continue
            metric_val = np.median(metric_list[-self._iter_log_window:])
            stats[name] = metric_val
            wandb_stats[name + self.suffix] = metric_val
        
        wandb_stats = {**{f"val/{k}_step":v for k,v in wandb_stats.items()}}

        if dist_utils.is_main_process():
            logging.log_json_stats(stats)
            wandb.log(wandb_stats)


    def visualize_prediction_result(self, vis_data: List, cur_epoch: int, save_visualization: bool, filepath: str):
        """
        Generate and add visualization of predictions to tensorboard.
        :param vis_data: a list of dictionaries. Each dictionary contains the target
                         and prediction array (numpy) with video name
        :param cur_epoch:
        :return:
        """
        res_list = []
        if len(vis_data) > 1 and len(vis_data) % 2 != 0:
            vis_data = vis_data[:-1]

        # import pdb; pdb.set_trace()
        for data in vis_data:
            gt_lbl, gt_lens = plot_utils.summarize_list(data['target'].tolist())
            pred_lbl, pred_lens = plot_utils.summarize_list(data['pred'].tolist())

            gt_res = plot_utils.generate_image_for_segmentation(gt_lbl, gt_lens,
                                                                colors=self.colors,
                                                                height=10,
                                                                white_label=self._cfg.DATA.BACKGROUND_INDICES)
            pred_res = plot_utils.generate_image_for_segmentation(pred_lbl, pred_lens,
                                                                  colors=self.colors,
                                                                  height=10,
                                                                  white_label=self._cfg.DATA.BACKGROUND_INDICES)
            res_list.append({"gt": gt_res,
                             "pred": pred_res,
                             "video_name": data['video_name'],
                             "len": sum(gt_lens)})

        fig = plot_utils.create_result_fig(res_list)

        # save to disk
        if save_visualization:
            plt.savefig(filepath, bbox_inches='tight')
            print(f"Epoch {cur_epoch} visualization dumped to {filepath}")
            plt.close()

        if self.writer:
            self.writer.add_figure("Val_Pred", fig, global_step=cur_epoch)

    def save_metrics(self, cur_epoch):
        filepath = os.path.join(self.temp_eval_dir, f'epoch{cur_epoch}{self.suffix}.pkl')

        with open(filepath, 'wb') as f:
            pickle.dump(self.metrics_dict, f)

    def log_epoch_stats(self, cur_epoch: int):
        """
         Log the stats of the current epoch.
        :param cur_epoch: the number of current epoch.
        :return:
        """
        mem_usage = misc.gpu_mem_usage()
        avg_loss = {}
        wandb_stats = dict()
        # for name, loss in self.loss_sums.items():
        #     avg_loss[name] = np.round(loss / self.num_samples, decimals=4)
        #     wandb_stats[name] = loss / self.num_samples

        loss_ce = None
        for name, loss in self.loss_dict.items():
            if loss.count == 0:
                continue
            avg_loss_ = loss.get_global_avg()
            avg_loss[name] = np.round(avg_loss_, decimals=4)
            wandb_stats[name + self.suffix] = avg_loss_

            if name == self.avg_loss_type:
                loss_ce = avg_loss_

        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "avg_loss": avg_loss,
            "mem": int(np.ceil(mem_usage)),
        }

        if self.writer:
            for name, loss_val in avg_loss.items():
                self.writer.add_scalar(f"2-Val_Loss/avg_{name}", loss_val, global_step=cur_epoch)

        # will be used to select best model

        if self._cfg.NUM_GPUS > 1:
            self.save_metrics(cur_epoch)

        metrics_sum = 0
        frame_f1= 0

        confusion_matrix = {k:v for k,v in self.metrics_dict.items() if 'confusion_matrix' in k}

        for name, metric_list in self.metrics_dict.items():
            if len(metric_list) == 0:
                continue

            if name.startswith('confusion_matrix'):
                continue
            
            metric_mean = np.mean(metric_list)
            stats[name] = metric_mean
            wandb_stats[name + self.suffix] = metric_mean
            if self.writer:
                self.writer.add_scalar(f"1-Val_Metric/{name}", metric_mean, global_step=cur_epoch)
            if name == 'Edit':
                metric_mean = metric_mean / 100.0
            
            if name.startswith('VideoF1'):
                video_f1 = metric_mean
            
            metrics_sum += metric_mean

        wandb_stats[f'metrics_sum{self.suffix}'] = metrics_sum
        
        if len(confusion_matrix) > 1:
            for cm_key, matrix in confusion_matrix.items():
                class_idx = cm_key.split('class')[1]
                frame_f1, _ = compute_macro_f1(matrix)
                wandb_stats[f'FramewiseF1_class{class_idx}' + self.suffix] = frame_f1
        else:
            wandb_stats[f'FramewiseF1' + self.suffix], f1_lst = compute_macro_f1(confusion_matrix['confusion_matrix'])
            for class_idx, f1 in enumerate(f1_lst):
                 wandb_stats[f'FramewiseF1_class{class_idx}' + self.suffix] = f1
        out_metrics = {'metrics_sum': metrics_sum, 'VideoF1': video_f1, self.avg_loss_type: loss_ce}

        wandb_stats = {**{f"val/{k}_epoch":v for k,v in wandb_stats.items()}, "epoch": cur_epoch}

        if dist_utils.is_main_process():
            logging.log_json_stats(stats)
            wandb.log(wandb_stats)

        return out_metrics
