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

    pred_mask = pred_mask.byte().squeeze()
    gt_mask = gt_mask.byte().squeeze()
    # print('pred_masks', pred_mask.shape, 'gt_masks', gt_mask.shape)
    intersection = torch.bitwise_and(pred_mask, gt_mask).sum().float()
    union = torch.bitwise_or(pred_mask, gt_mask).sum().float()
    return intersection / union

def compute_tp_fp_fn(pred, gt, t=0.5):
    ious = []
    tp = 0
    fp = 0
    fn = 0
                  # gt_idx, pred_idx, match, iou
    match_table = []

    for gt_idx, gt_mask in enumerate(gt):
        for pred_idx, pred_mask in enumerate(pred):
            iou = compute_iou(gt_mask, pred_mask).item()
            match = 1 if iou > t else 0
            row = [gt_idx, pred_idx, match, iou]
            match_table.append(row)
    
    match_table = np.array(match_table)
    print(match_table, '\n')

    # get tp and fn
    for gt_idx in range(len(gt)):
        current_idx_mt = match_table[match_table[:, 0] == gt_idx]
        n_matches = np.sum(current_idx_mt[:, 2], dtype=int)
        if n_matches > 0:
            tp += 1
        else:
            fn += 1
        
        # penalize duplicates
        n_iou_gt_thresh = np.sum((current_idx_mt[:, 3] > t).astype(int))
        if n_iou_gt_thresh > 1:
            fp += n_matches - 1

    # get fp
    for pred_idx in range(len(pred)):
        current_idx_mt = match_table[match_table[:, 1] == pred_idx]
        n_matches = np.sum(current_idx_mt[:, 2], dtype=int)

        if n_matches > 0:
            max_iou = np.max(current_idx_mt[:, 3])
            ious.append(max_iou)

        if (n_matches == 0):
            fp += 1
            ious.append(0.0)
        
    mean_iou = np.mean(ious) if len(ious) > 0  else 0.0

    return mean_iou, tp, fp, fn


def metrics(pred, gt, eps=1e-12):

    if isinstance(pred, np.ndarray):
        pred = torch.Tensor(pred)
    if isinstance(gt, np.ndarray):
        gt = torch.Tensor(gt)

    pred = pred.detach()

    if len(gt.shape) == 2:
        # turn instance maps into N binary instance maps each
        pred = torch.stack([(pred == i).byte() for i in torch.unique(pred) if i != 0], dim=0)
        gt = torch.stack([(gt == i).byte() for i in torch.unique(gt) if i != 0], dim=0)

    ap = []
    ar = []
    ious = []
    
    # thresholds = np.linspace(0.5, 0.95, 10)
    thresholds = np.linspace(0.0, 0.95, 20)
    for t in thresholds:
        t = np.round(t, decimals=3)  # for float precision errors
        mean_iou, tp, fp, fn = compute_tp_fp_fn(pred, gt, t=t)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        # print(f't {t :.2f}  iou {mean_iou :.3f}  precision {precision :.3f}  recall {recall :.5f}  tp {tp}  fp {fp}  fn {fn}')

        ap.append(precision)
        ar.append(recall)
        if t == 0.5:
            ap_50 = precision
            ious.append(mean_iou)
        elif t == 0.75:
            ap_75 = precision
    
    ap = np.array(ap)[::-1]
    ap = make_monotonic(ap)
    ar = np.array(ar)[::-1]

    if np.all(np.array(ar, dtype=np.float16) == 1.0) and np.all(np.array(ap, dtype=np.float16) == 1.0):
        ap = 1.0
    elif np.all(np.array(ar, dtype=np.float16) == np.mean(ar)) and np.all(np.array(ap, dtype=np.float16) == np.mean(ap)):
        ap = np.mean(ap)
    else:
        ap = np.sum((ar[1:] - ar[:-1]) * (ap[1:] + ap[:-1]) / 2)
    mean_iou = np.mean(ious)

    return mean_iou, ap, ap_50, ap_75


def make_monotonic(x):
    for i in range(len(x)-2, -1, -1):
        x[i] = max(x[i], x[i+1])
    return x


# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# import matplotlib.pyplot as plt
# from data import EMPSMaskRCNN, EMPSDataset
# from models import BranchedERFNet
# from utils.train import train_test_split_emps, load_pretrained
# from cluster import Cluster

# device='cpu'

# train_dataset, val_dataset = train_test_split_emps(EMPSDataset, 
#                                                    '/home/by256/Documents/Projects/particle-seg-dataset/elsevier/', 
#                                                    im_size=(512, 512), 
#                                                    device=device)

# model = BranchedERFNet(num_classes=[4, 1]).to(device)
# model.load_state_dict(torch.load('/home/by256/Documents/Projects/bpartis/bpartis/saved_models/emps-model.pt', map_location=device))
# model.eval()

# cluster = Cluster(n_sigma=2, device=device)

# ious = []
# aps = []
# ap_50s = []
# ap_75s = []

# for i in range(len(val_dataset)):
#     # if i < 46: # 7 doesn't work
#     #     continue
#     image, instance, _ = val_dataset[i]  # 37, 38
#     image = image.unsqueeze(0)

#     output = model(image).squeeze()

#     pred, _ = cluster.cluster(output)
#     pred = pred.to(device)

#     iou, ap, ap_50, ap_75 = metrics(pred, instance)
#     print('{}/{}    IoU: {:.5f}   AP: {:.5f}   AP_50: {:.5f}   AP_75: {:.5f}'.format(i+1, len(val_dataset), iou, ap, ap_50, ap_75))

#     ious.append(iou)
#     aps.append(ap)
#     ap_50s.append(ap_50)
#     ap_75s.append(ap_75)
#     # print('{}/{}'.format(i+1, len(val_dataset)), flush=True, end='\r')

#     # fig, axes = plt.subplots(1, 3)
#     # for ax in axes:
#     #     ax.axis('off')
#     # axes[0].imshow(image.squeeze().permute(1, 2, 0).cpu())
#     # axes[1].matshow(pred.cpu(), cmap='tab20')
#     # axes[1].set_title('Pred') 
#     # axes[2].matshow(instance.cpu(), cmap='tab20')
#     # axes[2].set_title('GT')
#     # plt.show()
#     # break

# print('IoU: {:.5f}   AP: {:.5f}   AP_50: {:.5f}   AP_75: {:.5f}'.format(np.mean(ious), 
#                                                                         np.mean(aps), 
#                                                                         np.mean(ap_50s), 
#                                                                         np.mean(ap_75s)))
