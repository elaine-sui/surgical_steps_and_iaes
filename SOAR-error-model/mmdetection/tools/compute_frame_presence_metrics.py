"""Compute frame-level presence metrics (accuracy, AUROC) for the thermal
injury detector.

This treats detection as a binary per-frame presence task: a frame is positive
if it contains at least one ground-truth box, and the prediction score for a
frame is the highest-confidence detected box on that frame. Run
``tools/test_detector.py`` with ``--out <preds>.pkl`` first to produce the
prediction pickle.

Example:
    python3 tools/compute_frame_presence_metrics.py \
        --pred_pkl /path/to/test_preds.pkl \
        --gt_ann /path/to/testing.json
"""
import argparse
import pickle
import json
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


def convert_gt_to_binary(gt, all_image_ids):
    anns = gt['annotations']
    image_ids_with_bboxes = []
    for ann in anns:
        id = ann['image_id']
        image_ids_with_bboxes.append(id)

    image_ids_with_bboxes = list(set(image_ids_with_bboxes))

    img_id2label = {i: 0 for i in sorted(all_image_ids)}
    for id in image_ids_with_bboxes:
        img_id2label[id] = 1

    return img_id2label


def convert_pred_to_binary(pred, all_img_ids):
    img_id2highest_score = {i: 0. for i in sorted(all_img_ids)}
    for pred_item in pred:
        img_id = pred_item['img_id']
        scores = pred_item['pred_instances']['scores'].numpy()
        img_id2highest_score[img_id] = scores.max().item()

    return img_id2highest_score


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute frame-level presence metrics for the thermal '
        'injury detector')
    parser.add_argument(
        '--pred_pkl',
        type=str,
        default='work_dirs/deformable-detr-refine-twostage_r50_15xb2-50e_coco_thermal_injury_w_more_aug/test_preds.pkl',
        help='prediction pickle produced by tools/test_detector.py --out')
    parser.add_argument(
        '--gt_ann',
        type=str,
        default='/path/to/frame_annotations/2024-10-28/Thermal_Injury/testing.json',
        help='COCO-format ground-truth annotation json')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.pred_pkl, 'rb') as f:
        pred = pickle.load(f)

    with open(args.gt_ann, 'r') as f:
        gt = json.load(f)

    all_image_ids = sorted([img['id'] for img in gt['images']])

    img_id2label = convert_gt_to_binary(gt, all_image_ids)
    pred_img_id2highest_score = convert_pred_to_binary(pred, all_image_ids)

    labels = np.array(list(img_id2label.values()))
    pred = np.array(list(pred_img_id2highest_score.values()))

    auroc = roc_auc_score(labels, pred)
    acc = accuracy_score(labels, np.round(pred))

    print(f"Acc: {acc} \t AUROC: {auroc}")


if __name__ == '__main__':
    main()
