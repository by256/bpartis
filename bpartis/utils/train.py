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

def train_test_split_emps(dataset, data_dir, im_size=(256, 256), device='cuda'):
    train_dataset = dataset('{}/processed-images/'.format(data_dir), '{}/segmaps/'.format(data_dir), im_size=im_size, device=device)
    val_dataset = dataset('{}/processed-images/'.format(data_dir), '{}/segmaps/'.format(data_dir), im_size=im_size, transform=False, device=device)

    unique_dois = sorted(list(set([x.split('.png')[0].split(' (')[0].split('.gr')[0] for x in train_dataset.image_fns])))

    np.random.seed(9)
    indices = np.arange(len(unique_dois))
    np.random.shuffle(indices)

    train_indices = indices[:int(0.86*len(indices))]
    val_indices = indices[int(0.86*len(indices)):]
    train_dois = [unique_dois[i] for i in train_indices]
    val_dois = [unique_dois[i] for i in val_indices]

    train_dataset = split_train_val_dois(train_dataset, train_dois)
    val_dataset = split_train_val_dois(val_dataset, val_dois)
    return train_dataset, val_dataset

def load_pretrained(model, path='./saved_models/', device='cuda'):
    # load encoder
    model.encoder.load_state_dict(torch.load('{}pretrained-encoder.pt'.format(path), map_location=device))
    # load decoders
    pretrained_dict = torch.load('{}pretrained-decoder.pt'.format(path), map_location=device)
    for decoder in model.decoders:
        model_dict = decoder.state_dict()
        del model_dict['output_conv.weight']
        del model_dict['output_conv.bias']
        pretrained_dict_copy = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict_copy) 
        decoder.load_state_dict(pretrained_dict_copy, strict=False)
    
    return model
