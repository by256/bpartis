import numpy as np
import torch.nn as nn
from pytorch_msssim import MS_SSIM


def train_test_split_sem(dataset, data_dir, device='cuda'):
    train_dataset = dataset(data_dir, device=device)
    val_dataset = dataset(data_dir, device=device)

    np.random.seed(11)
    indices = np.arange(len(train_dataset))
    np.random.shuffle(indices)

    train_indices = indices[:int(0.9*len(indices))]
    val_indices = indices[int(0.9*len(indices)):]
    train_dataset.image_fns = [train_dataset.image_fns[i] for i in train_indices]
    val_dataset.image_fns = [val_dataset.image_fns[i] for i in val_indices]
    return train_dataset, val_dataset
