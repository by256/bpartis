"""
## Evaluation utils
--------------------------------------------------
## Author: Batuhan Yildirim
## Email: by256@cam.ac.uk
## Version: 1.0.0
--------------------------------------------------
## License: MIT
## Copyright: Copyright Batuhan Yildirim, 2020, BPartIS
-------------------------------------------------
"""

import torch
import numpy as np


def compute_iou(pred_mask, gt_mask):
    """
    Computes IoU between predicted instance mask and 
    ground-truth instance mask
    """
    intersection = pred_mask & gt_mask
    union = pred_mask | gt_mask
    return float(intersection.sum()) / float(union.sum())

def compute_matches(pred, gt, t=0.5):
    matches = []
    ious = []
    for pred_idx, pred_mask in enumerate(pred):
        for gt_idx, gt_mask in enumerate(gt):
            iou = compute_iou(pred_mask, gt_mask)
            if iou > t:
                match = (pred_idx, gt_idx)
                matches.append(match)
                ious.append(iou)
                break
    return matches, np.mean(ious)

def metrics(pred, gt, t=0.5):

    print('pred', pred.shape, 'gt', gt.shape)

    fig, axes = plt.subplots(1, 2)
    axes[0].matshow(pred)
    axes[1].matshow(gt)
    plt.show()

    if len(gt.shape) == 2:
        # turn instance maps into N binary instance maps each
        pred = torch.stack([(pred == i).byte() for i in torch.unique(pred) if i != 0], dim=0)
        gt = torch.stack([(gt == i).byte() for i in torch.unique(gt) if i != 0], dim=0)
    print('pred', pred.shape, 'gt', gt.shape)

    matches, mean_iou = compute_matches(pred, gt)
    print(matches)
    print('mean_iou', mean_iou)
    tp = len(matches)
    fp = np.maximum(len(pred) - tp, 0)
    fn = np.maximum(len(gt) - tp, 0)

    print('tp', tp, 'fp', fp, 'fn', fn)

    precision = tp / (tp + fp)

    return mean_iou, precision

def average_precision_range(pred, gt):

    if len(gt.shape) == 2:
        # turn instance maps into N binary instance maps each
        pred = torch.stack([(pred == i).byte() for i in torch.unique(pred) if i != 0], dim=0)
        gt = torch.stack([(gt == i).byte() for i in torch.unique(gt) if i != 0], dim=0)

    aps = []
    thresholds = np.linspace(0.5, 0.95, 10)
    for t in thresholds:
        matches, _ = compute_matches(pred, gt, t=t)
        print(len(matches))
        tp = len(matches)
        fp = np.maximum(len(pred) - tp, 0.0)
        fn = np.maximum(len(gt) - tp, 0.0)
        precision = tp / (tp + fp)
        aps.append(precision)
    return np.mean(aps)
