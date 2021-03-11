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

import csv
import copy
import torch
import numpy as np


def get_split_from_csv(dataset, csv_path):
    """Obtain a subset of the EMPS dataset for train test split purposes"""
    dataset = copy.deepcopy(dataset)
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        split_fns = list(reader)
    split_fns = [x[0] for x in split_fns]

    split_image_fns = [x for x in dataset.image_fns if x.split('.png')[0] in split_fns]

    dataset.image_fns = split_image_fns
    return dataset

def split_train_test_dois(dataset, dois):
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

def train_test_split_emps_old(dataset, data_dir, im_size=(512, 512), device='cuda'):
    train_dataset = dataset('{}/processed-images/'.format(data_dir), '{}/segmaps/'.format(data_dir), im_size=im_size, device=device)
    test_dataset = dataset('{}/processed-images/'.format(data_dir), '{}/segmaps/'.format(data_dir), im_size=im_size, transform=False, device=device)

    unique_dois = sorted(list(set([x.split('.png')[0].split(' (')[0].split('.gr')[0] for x in train_dataset.image_fns])))

    np.random.seed(15)
    indices = np.arange(len(unique_dois))
    np.random.shuffle(indices)

    train_indices = indices[:int(0.77*len(indices))]
    test_indices = indices[int(0.77*len(indices)):]
    train_dois = [unique_dois[i] for i in train_indices]
    test_dois = [unique_dois[i] for i in test_indices]

    train_dataset = split_train_test_dois(train_dataset, train_dois)
    test_dataset = split_train_test_dois(test_dataset, test_dois)
    return train_dataset, test_dataset

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
