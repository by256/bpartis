"""
## Train Mask R-CNN on EMPS dataset.
--------------------------------------------------
## Author: Batuhan Yildirim
## Email: by256@cam.ac.uk
## Version: 1.0.0
--------------------------------------------------
## License: MIT
## Copyright: Copyright Batuhan Yildirim, 2020, BPartIS
-----
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

sys.path.append('../../')
from bpartis.data import EMPSMaskRCNN
from bpartis.utils.train import train_test_split_emps, compute_decay_rate


parser = argparse.ArgumentParser(description='Train Mask R-CNN model on EMPS dataset.')
parser.add_argument('--data-dir', metavar='data_dir', type=str, help='Directory which contains the data.')
parser.add_argument('--device', metavar='device', type=str, default='cuda', help='device to train on (cuda or cpu)')
parser.add_argument('--im-size', metavar='im_size', type=tuple, default=(512, 512), help='Image size to load for training.')
parser.add_argument('--save-dir', metavar='save_dir', type=str, default='./saved_models/', help='directory to save and load weights from.')
parser.add_argument('--batch-size', metavar='batch_size', type=int, default=5, help='Batch size for training.')
parser.add_argument('--lr', metavar='lr', type=float, default=3e-4, help='Learning rate.')
parser.add_argument('--end-lr', metavar='end_lr', type=float, default=None, help='Learning rate to decay to.')
parser.add_argument('--epochs', metavar='epochs', type=int, default=300, help='No. of epochs to train.')
namespace = parser.parse_args()

train_dataset, val_dataset = train_test_split_emps(EMPSMaskRCNN, 
                                                   namespace.data_dir, 
                                                   im_size=namespace.im_size, 
                                                   device=namespace.device)

# train_dataset.image_fns = train_dataset.image_fns[:20]
# val_dataset.image_fns = val_dataset.image_fns[:4]


print('Train: {}    Val: {}'.format(len(train_dataset), len(val_dataset)))

collate_fn = lambda x: tuple(zip(*x))
train_loader = DataLoader(train_dataset, batch_size=namespace.batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=namespace.batch_size, collate_fn=collate_fn)

backbone = resnet_fpn_backbone('resnet101', pretrained=True)
backbone.requires_grad = False
model = MaskRCNN(backbone=backbone, num_classes=2).to(namespace.device)

optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=namespace.lr)
if namespace.end_lr is not None:
    decay = compute_decay_rate(start_lr=namespace.lr, end_lr=namespace.end_lr, epochs=int(namespace.epochs*0.75))
    lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=decay)

from engine import train_one_epoch, evaluate
import utils

for epoch in range(namespace.epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, train_loader, namespace.device, epoch, print_freq=10)
    # update the learning rate
    if (namespace.end_lr is not None) & (epoch+1 <= int(namespace.epochs*0.75)):
        lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, val_loader, device=namespace.device)

    torch.save(model.state_dict(), '{}mask-rcnn.pt'.format(namespace.save_dir))

# losses = {'train': [], 'val-iou': [], 'val-ap': [], 'val-ap-50': []}

# for epoch in range(namespace.epochs):

#     epoch_train_losses = []
#     start = time.time()
#     model.train()

#     for (images, targets) in train_loader:
#         images = list(image.to(namespace.device) for image in images)
#         targets = [{k: v.to(namespace.device) for k, v in t.items()} for t in targets]
#         loss_dict = model(images, targets)
#         loss = loss_dict['loss_classifier'] + loss_dict['loss_box_reg'] + loss_dict['loss_mask']
#         epoch_train_losses.append(loss.item())

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     losses['train'].append(np.mean(epoch_train_losses))

#     epoch_val_ious = []
#     epoch_val_aps = []
#     epoch_val_ap_50s = []
#     model.eval()

#     if epoch > 10:

#         for (images, targets) in val_loader:
#             images = list(image.to(namespace.device) for image in images)
#             targets = [{k: v.to(namespace.device) for k, v in t.items()} for t in targets]
#             outputs = model(images)
#             for i, output in enumerate(outputs):
#                 pred_masks = output['masks'].detach()
#                 gt_masks = targets[i]['masks'].detach()
#                 # print('pred_masks', pred_masks.shape, 'gt_masks', gt_masks.shape)
#                 iou, ap, ap_50, ap_75 = metrics(pred_masks, gt_masks)
#                 epoch_val_ious.append(iou)
#                 epoch_val_aps.append(ap)
#                 epoch_val_ap_50s.append(ap_50)
        
#         if (namespace.end_lr is not None) & (epoch+1 <= int(namespace.epochs*0.75)):
#             lr_scheduler.step()

#     save_model = (np.mean(epoch_val_ious) < np.min(losses['val-iou']) if epoch > 0 else False)
#     losses['val-iou'].append(np.mean(epoch_val_ious))
#     losses['val-ap'].append(np.mean(epoch_val_aps))
#     losses['val-ap-50'].append(np.mean(epoch_val_ap_50s))

#     print('{}/{}    Train: {:.5f}    Val IOU: {:.5f}    Val AP: {:.5f}    Val AP_50: {:.5f}    lr: {:.9f}    T: {:.2f} s'.format(epoch+1, namespace.epochs, losses['train'][-1], losses['val-iou'][-1], losses['val-ap'][-1], losses['val-ap-50'][-1], optimizer.param_groups[-1]['lr'], time.time()-start))

#     if save_model:
#         torch.save(model.state_dict(), '{}mask-rcnn.pt'.format(namespace.save_dir))