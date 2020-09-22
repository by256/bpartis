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
    # intersection = pred_mask & gt_mask
    # union = pred_mask | gt_mask
    pred_mask = pred_mask.byte()
    gt_mask = gt_mask.byte()
    intersection = torch.bitwise_and(pred_mask, gt_mask).sum().float()
    union = torch.bitwise_or(pred_mask, gt_mask).sum().float()
    print(intersection, union)
    return intersection / union

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
    if len(ious) == 0:
        mean_iou = 0.0
    else:
        mean_iou = np.mean(ious)
    return matches, mean_iou

def metrics(pred, gt, t=0.5, eps=1e-12):

    pred = pred.detach()

    if len(gt.shape) == 2:
        # turn instance maps into N binary instance maps each
        pred = torch.stack([(pred == i).byte() for i in torch.unique(pred) if i != 0], dim=0)
        gt = torch.stack([(gt == i).byte() for i in torch.unique(gt) if i != 0], dim=0)

    matches, mean_iou = compute_matches(pred, gt)
    tp = len(matches)
    fp = np.maximum(len(pred) - tp, 0)
    fn = np.maximum(len(gt) - tp, 0)
    print('tp', tp)

    # if tp > 0:
    precision = tp / (tp + fp + eps)
    # else: precision = 0.0

    return mean_iou, precision

def average_precision_range(pred, gt, eps=1e-12):

    if len(gt.shape) == 2:
        # turn instance maps into N binary instance maps each
        pred = torch.stack([(pred == i).byte() for i in torch.unique(pred) if i != 0], dim=0)
        gt = torch.stack([(gt == i).byte() for i in torch.unique(gt) if i != 0], dim=0)

    aps = []
    thresholds = np.linspace(0.5, 0.95, 10)
    for t in thresholds:
        matches, _ = compute_matches(pred, gt, t=t)
        tp = len(matches)
        fp = np.maximum(len(pred) - tp, 0.0)
        fn = np.maximum(len(gt) - tp, 0.0)
        precision = tp / (tp + fp + eps)
        aps.append(precision)
    return np.mean(aps)



# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# from data import EMPSMaskRCNN
# from utils.train import train_test_split_emps

# device='cpu'
# train_dataset, val_dataset = train_test_split_emps(EMPSMaskRCNN, 
#                                                    '/home/by256/Documents/Projects/particle-seg-dataset/elsevier/', 
#                                                    im_size=(512, 512), 
#                                                    device=device)

# # model = BranchedERFNet(num_classes=[4, 1]).to(device)
# # model = load_pretrained(model, path='/home/by256/Documents/Projects/bpartis/bpartis/saved_models/emps-model.pt', device=device)

# # image, instance, _ = val_dataset[0]

# image, targets = val_dataset[0]
# masks = targets['masks']

# image, targets = val_dataset[4]
# masks_2 = targets['masks']

# # print(targets)
# # print(masks.shape)

# iou, ap = metrics(masks, masks_2)

# print(iou, ap)