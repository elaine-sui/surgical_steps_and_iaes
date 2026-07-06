from typing import List, Dict
from torch import Tensor
import numpy as np

from .base import Metric
from .segmentation import MoFAccuracyMetric, FramewiseF1Metric
from .fully_supervised import F1Score, Edit


def calculate_metrics(target: Tensor, pred: Tensor, ignored_class_ids: List[int], num_classes: int) -> Dict:
    """
    Calculates the action segmentation metrics (MoF, Edit, F1@{10, 25, 50}) for a video
    :param target: a Tensor of shape [batch_size, sequence_len]
    :param pred: a Tensor of shape [batch_size, sequence_len]
    :param ignored_class_ids: a list of class ids to ignore during calculation
    :return:
      a dict of metrics values
    """
    assert target.shape[0] == 1, "Batch size should be one for validation due to limitation of metric functions."

    result_dict = {}
    mof_func = MoFAccuracyMetric(ignore_ids=ignored_class_ids)
    edit_func = Edit(ignore_ids=ignored_class_ids)
    f1_func = F1Score(ignore_ids=ignored_class_ids)
    framewise_f1 = FramewiseF1Metric(ignore_ids=ignored_class_ids, num_classes=num_classes)

    mof_func.add(target, pred)
    edit_func.add(target, pred)
    f1_func.add(target, pred)
    framewise_f1.add(target, pred)

    result_dict['MoF'] = mof_func.summary()
    result_dict['Edit'] = edit_func.summary()
    f1_dict = f1_func.summary()
    result_dict.update(f1_dict)

    framewise_f1_dict = framewise_f1.summary()
    result_dict.update(framewise_f1_dict)
    return result_dict


def calculate_metrics_multilabel(target: Tensor, pred: Tensor, ignored_class_ids: List[int], num_classes: int) -> Dict:
    """
    Calculates the action segmentation metrics (MoF, Edit, F1@{10, 25, 50}) for a video in the multilabel setting
    :param target: a Tensor of shape [batch_size, sequence_len, num_classes]
    :param pred: a Tensor of shape [batch_size, sequence_len, num_classes]
    :param ignored_class_ids: a list of class ids to ignore during calculation
    :return:
      a dict of metrics values
    """
    assert target.shape[0] == 1, "Batch size should be one for validation due to limitation of metric functions."

    result_dict = {}
    mof_func = MoFAccuracyMetric(ignore_ids=ignored_class_ids, multilabel=True, num_classes=num_classes)
    edit_func = Edit(ignore_ids=ignored_class_ids, num_classes=num_classes)
    # f1_func = F1Score(ignore_ids=ignored_class_ids)
    framewise_f1 = FramewiseF1Metric(ignore_ids=ignored_class_ids, num_classes=num_classes)

    mof_func.add_multilabel(target, pred)
    edit_func.add_multilabel(target, pred)
    # f1_func.add(target, pred)
    framewise_f1.add_multilabel(target, pred)

    mof_dict = mof_func.summary_multilabel()
    result_dict.update(mof_dict)
    edit_dict = edit_func.summary_multilabel()
    result_dict.update(edit_dict)
    # f1_dict = f1_func.summary()
    # result_dict.update(f1_dict)

    framewise_f1_dict = framewise_f1.summary_multilabel()
    result_dict.update(framewise_f1_dict)
    return result_dict


def calculate_metrics_mixed(target: Tensor, pred: Tensor, ignored_class_ids: List[int], num_classes: List[int]) -> Dict:
    num_step_classes = num_classes[0]
    target_steps = target[:, :, :num_step_classes]
    target_tasks = target[:, :, num_step_classes]

    pred_steps = pred[:, :, :num_step_classes]
    pred_tasks = pred[:, :, num_step_classes]

    step_metrics = calculate_metrics(target_steps, pred_steps, ignored_class_ids, num_classes[0])
    ignored_class_ids_task = [i - num_classes[0] for i in ignored_class_ids]
    task_metrics = calculate_metrics_multilabel(target_tasks, pred_tasks, ignored_class_ids_task, num_classes[1])

    return step_metrics, task_metrics
    

def get_metric(name):
    return {
        "MoF": MoFAccuracyMetric,
        "F1": F1Score,
        "Edit": Edit,
        "FramewiseF1": FramewiseF1Metric,
    }[name]


def calculate_metrics_framewise(target: Tensor, pred: Tensor, ignored_class_ids: List[int], num_classes: int) -> Dict:
    # compute accuracy and f1
    result_dict = {}
    mof_func = MoFAccuracyMetric(ignore_ids=ignored_class_ids)
    framewise_f1 = FramewiseF1Metric(ignore_ids=ignored_class_ids, num_classes=num_classes)

    mof_func.add(target, pred)
    framewise_f1.add(target, pred)

    result_dict['MoF'] = mof_func.summary()

    framewise_f1_dict = framewise_f1.summary()
    result_dict.update(framewise_f1_dict)

    return result_dict



def compute_macro_f1(confusion_matrix):
    print(confusion_matrix)
    tp_list = np.diagonal(confusion_matrix)
    row_sum = np.sum(confusion_matrix, axis=0)
    col_sum = np.sum(confusion_matrix, axis=1)

    f1_lst = (2 * tp_list) / (col_sum + row_sum)

    return np.mean(f1_lst), f1_lst

def compute_phase_recall(confusion_matrix):
    tp_list = np.diagonal(confusion_matrix)
    row_sum = np.sum(confusion_matrix, axis=0)

    rec_lst = tp_list / row_sum
    return np.mean(rec_lst), rec_lst

def compute_phase_precision(confusion_matrix):
    tp_list = np.diagonal(confusion_matrix)
    col_sum = np.sum(confusion_matrix, axis=1)
    
    prec_lst = tp_list / col_sum
    return np.mean(prec_lst), prec_lst
