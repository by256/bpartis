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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from PIL import Image

sys.path.append('../../')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


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

def split_train_val_dois(dataset, dois):
    new_fns = []
    for fn in dataset.image_fns:
        keep_fn = False
        for doi in dois:
            if doi in fn:
                keep_fn = True
        if keep_fn:
            new_fns.append(fn)
    dataset.image_fns = new_fns
    return dataset

def freeze_batchnorm_layers(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            module.eval()

def compute_decay_rate(start_lr, end_lr, epochs):
    return np.exp(np.log(end_lr/start_lr)/epochs)

def train_test_split_emps(dataset, data_dir, im_size=(512, 512), device='cuda'):
    train_dataset = dataset('{}/processed-images/'.format(data_dir), '{}/segmaps/'.format(data_dir), im_size=im_size, device=device)
    val_dataset = dataset('{}/processed-images/'.format(data_dir), '{}/segmaps/'.format(data_dir), im_size=im_size, transform=False, device=device)

    unique_dois = sorted(list(set([x.split('.png')[0].split(' (')[0].split('.gr')[0] for x in train_dataset.image_fns])))

    np.random.seed(15)
    indices = np.arange(len(unique_dois))
    np.random.shuffle(indices)

    # train_indices = indices[:int(0.88*len(indices))]
    # val_indices = indices[int(0.88*len(indices)):]
    train_indices = indices[:int(0.77*len(indices))]
    val_indices = indices[int(0.77*len(indices)):]
    train_dois = [unique_dois[i] for i in train_indices]
    val_dois = [unique_dois[i] for i in val_indices]

    train_dataset = split_train_val_dois(train_dataset, train_dois)
    val_dataset = split_train_val_dois(val_dataset, val_dois)
    return train_dataset, val_dataset

class EMPSMaskRCNN(Dataset):
    
    def __init__(self, image_dir, mask_dir, im_size=(256, 256), device='cuda', transform=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.im_size = im_size
        self.device = device
        self.transform = transform

        self.image_fns = os.listdir(image_dir)
        self.image_fns = [x for x in self.image_fns if x.endswith('.png')]

        np.random.seed(9)
        shuffle_idx = np.arange(len(self.image_fns)).astype(int)
        np.random.shuffle(shuffle_idx)

        self.image_fns = list(np.array(self.image_fns)[shuffle_idx])
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

        self.colour_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
        self.crop = transforms.RandomCrop((self.im_size[0]//2, self.im_size[1]//2))
        self.hor_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.vert_flip = transforms.RandomVerticalFlip(p=1.0)

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, idx):
        image = Image.open(self.image_dir + self.image_fns[idx]).resize(self.im_size, resample=Image.BICUBIC)
        mask = Image.open(self.mask_dir + self.image_fns[idx]).resize(self.im_size, resample=Image.NEAREST)
        obj_ids = np.unique(mask)[1:]

        if self.transform:

            # hor-ver flip
            image, mask = self.horizontal_flip(image, mask)
            image, mask = self.vertical_flip(image, mask)

            # rotate
            image, mask = self.random_rotation(image, mask)

            # colour jitter
            image = self.colour_jitter(image)

            # random crop
            image, mask = self.random_crop(image, mask)
        
        image = np.array(image)
        mask = np.array(mask)
        
        image = torch.Tensor(image).permute(2, 0, 1) 
        image = image / 255.0
        image = (image - self.mean) / self.std
        
        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.Tensor(boxes).float()
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.Tensor(masks).byte()

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target

    def horizontal_flip(self, image, instances, p=0.5):
        if np.random.uniform() < p:
            image = self.hor_flip(image)
            instances = self.hor_flip(instances)
        return image, instances

    def vertical_flip(self, image, instances, p=0.5):
        if np.random.uniform() < p:
            image = self.vert_flip(image)
            instances = self.vert_flip(instances)
        return image, instances

    def random_rotation(self, image, instances):
        random_number = np.random.uniform()
        if random_number < 0.25:
            image = image.rotate(90)
            instances = instances.rotate(90)
        elif (random_number >= 0.25) & (random_number < 0.5):
            image = image.rotate(180)
            instances = instances.rotate(180)
        elif (random_number >= 0.5) & (random_number < 0.75):
            image = image.rotate(270)
            instances = instances.rotate(270)
        return image, instances

    def random_crop(self, image, instances):
        if np.random.uniform() <= 0.333:
            i, j, h, w = self.crop.get_params(image, self.crop.size)
            image_cropped = transforms.functional.crop(image, i, j, h, w).resize(self.im_size, resample=Image.BICUBIC)
            instances_cropped = transforms.functional.crop(instances, i, j, h, w).resize(self.im_size, resample=Image.NEAREST)
            # recursively random crop until n instances > 0 (not a problem in almost all cases.)
            if len(np.unique(np.array(instances_cropped))) == 1:
                image_cropped, instances_cropped = self.random_crop(image, instances)
            return image_cropped, instances_cropped
        else:
            return image, instances



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