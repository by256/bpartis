"""
## Train utils
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