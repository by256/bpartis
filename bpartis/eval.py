"""
## Eval model on EMPS dataset.
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
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from data import EMPSDataset
from models import BranchedERFNet
from losses import SpatialEmbLoss
from utils.train import train_test_split_emps
from uncertainty import enable_eval_dropout, monte_carlo_predict


parser = argparse.ArgumentParser(description='Train model on EMPS dataset.')
parser.add_argument('--data-dir', metavar='data_dir', type=str, help='Directory which contains the data.')
parser.add_argument('--model-path', metavar='model_path', type=str, help='Path to saved model parameters.')
parser.add_argument('--device', metavar='device', type=str, default='cuda', help='device to train on (cuda or cpu)')
parser.add_argument('--im-size', metavar='im_size', type=tuple, default=(256, 256), help='Image size to load for training.')
parser.add_argument('--batch-size', metavar='batch_size', type=int, default=1, help='Batch size for training.')
namespace = parser.parse_args()

_, val_dataset = train_test_split_emps(EMPSDataset, 
                                      namespace.data_dir, 
                                      im_size=namespace.im_size, 
                                      device=namespace.device)

print('Val: {}'.format(len(val_dataset)))

val_loader = DataLoader(val_dataset, batch_size=namespace.batch_size)

model = BranchedERFNet(num_classes=[4, 1]).to(namespace.device)
model.load_state_dict(torch.load(namespace.model_path, map_location=namespace.device))
criterion = SpatialEmbLoss(to_center=False, n_sigma=2, foreground_weight=10).to(namespace.device)

val_losses = []
val_ious = []
model.eval()

# colormap for segmentation maps 
seg_cmap = matplotlib.cm.tab20
seg_cmap.set_bad(color='k')

for (image, instances, class_labels) in val_loader:
    output = model(image)
    loss, ious = criterion(output, instances, class_labels, iou=True)
    loss = loss.mean()
    val_losses.append(loss.item())
    val_ious.append(ious)

    predictions, uncertainty = monte_carlo_predict(model, image[0].unsqueeze(0), device=namespace.device)
    predictions = predictions.float().cpu().numpy()
    predictions[predictions == 0] = np.nan
    instances = instances.float().cpu().numpy()
    instances[instances == 0] = np.nan
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.25))
    for ax in axes:
        ax.axis('off')
    axes[0].imshow(image[0].permute(1, 2, 0).cpu())
    axes[1].matshow(predictions, cmap=seg_cmap)
    axes[1].set_title('Pred - Loss: {:.5f}    IOU: {:.5f}'.format(loss.item(), ious))
    axes[2].matshow(instances[0], cmap=seg_cmap)
    axes[2].set_title('GT')
    axes[3].matshow(uncertainty.cpu())
    axes[3].set_title('Uncertainty')
    plt.show()

