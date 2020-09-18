"""
## Train model on EMPS dataset.
--------------------------------------------------
## Author: Batuhan Yildirim
## Email: by256@cam.ac.uk
## Version: 1.0.0
--------------------------------------------------
## License: MIT
## Copyright: Copyright Batuhan Yildirim, 2020, BPartIS
-----
"""

import time
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from data import EMPSDataset
from models import BranchedERFNet
from losses import SpatialEmbLoss
from utils.train import train_test_split_emps, freeze_batchnorm_layers, compute_decay_rate, load_pretrained


parser = argparse.ArgumentParser(description='Train model on EMPS dataset.')
parser.add_argument('--data-dir', metavar='data_dir', type=str, help='Directory which contains the data.')
parser.add_argument('--device', metavar='device', type=str, default='cuda', help='device to train on (cuda or cpu)')
parser.add_argument('--im-size', metavar='im_size', type=tuple, default=(256, 256), help='Image size to load for training.')
parser.add_argument('--finetune', metavar='finetune', type=bool, default=True, help='Load pretrained weights.')
parser.add_argument('--save-dir', metavar='save_dir', type=str, default='./saved_models/', help='directory to save and load weights from.')
parser.add_argument('--batch-size', metavar='batch_size', type=int, default=5, help='Batch size for training.')
parser.add_argument('--lr', metavar='lr', type=float, default=3e-4, help='Learning rate.')
parser.add_argument('--epochs', metavar='epochs', type=int, default=300, help='No. of epochs to train.')
namespace = parser.parse_args()

train_dataset, val_dataset = train_test_split_emps(EMPSDataset, 
                                                   namespace.data_dir, 
                                                   im_size=namespace.im_size, 
                                                   device=namespace.device)

print('Train: {}    Val: {}'.format(len(train_dataset), len(val_dataset)))

train_loader = DataLoader(train_dataset, batch_size=namespace.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=namespace.batch_size)

model = BranchedERFNet(num_classes=[4, 1]).to(namespace.device)
model.init_output(n_sigma=2)

if namespace.finetune:
    model = load_pretrained(model, path=namespace.save_dir, device=namespace.device)

loss_w = {
        'w_inst': 1,
        'w_var': 10,
        'w_seed': 1,
    }
criterion = SpatialEmbLoss(to_center=False, n_sigma=2, foreground_weight=10).to(namespace.device)
optimizer = Adam(model.parameters(), lr=namespace.lr)

losses = {'train': [], 'val': [], 'val-iou': []}

for epoch in range(namespace.epochs):

    epoch_train_losses = []
    start = time.time()
    model.train()
    
    if namespace.finetune:
        freeze_batchnorm_layers(model.encoder)
        for decoder in model.decoders:
            freeze_batchnorm_layers(decoder)

    for (image, instances, class_labels) in train_loader:
        output = model(image)
        loss, _ = criterion(output, instances, class_labels, **loss_w)
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_losses.append(loss.item())
    losses['train'].append(np.mean(epoch_train_losses))

    epoch_val_losses = []
    epoch_val_ious = []
    model.eval()
    for (image, instances, class_labels) in val_loader:
        output = model(image)
        loss, ious = criterion(output, instances, class_labels, **loss_w, iou=True)
        loss = loss.mean()
        epoch_val_losses.append(loss.item())
        epoch_val_ious.append(ious)

    save_model = (np.mean(epoch_val_losses) < np.min(losses['val']) if epoch > 0 else False)
    losses['val'].append(np.mean(epoch_val_losses))
    losses['val-iou'].append(np.mean(epoch_val_ious))

    print('{}/{}    Train: {:.5f}    Val: {:.5f}    Val IOU: {:.5f}    lr: {:.9f}    T: {:.2f} s'.format(epoch+1, namespace.epochs, losses['train'][-1], losses['val'][-1], losses['val-iou'][-1], optimizer.param_groups[-1]['lr'], time.time()-start))

    if save_model:
        torch.save(model.state_dict(), '{}emps-model.pt'.format(namespace.save_dir))

with open('{}logs/emps-losses.pkl'.format(namespace.save_dir), 'wb') as f:
    pickle.dump(losses, f)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(losses['train'], label='train')
axes[0].plot(losses['val'], label='val')
axes[0].set_title('End Train: {:.5f}    End Val: {:.5f}    Best Val: {:.5f}'.format(losses['train'][-1], losses['val'][-1], np.min(losses['val'])))
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Spatial Embedding Loss')
axes[1].plot(losses['val-iou'])
axes[1].set_title('End IOU: {:.5f}    Best IOU: {:.5f}'.format(losses['val-iou'][-1], np.max(losses['val-iou'])))
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Mean IOU')
plt.savefig('{}logs/emps-losses.png'.format(namespace.save_dir), bbox_inches='tight', pad_inches=0.1)
plt.close()
