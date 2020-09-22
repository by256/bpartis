"""
## Data Loaders for EMPS and SEM datasets.
--------------------------------------------------
## Author: Batuhan Yildirim
## Email: by256@cam.ac.uk
## Version: 1.0.0
--------------------------------------------------
## License: MIT
## Copyright: Copyright Batuhan Yildirim, 2020, BPartIS
-------------------------------------------------
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms


class EMPSDataset(Dataset):
    def __init__(self, image_dir, segmap_dir, im_size=(512, 512), transform=True, device='cuda'):
        self.image_dir = image_dir
        self.segmap_dir = segmap_dir
        self.im_size = im_size
        self.transform = transform
        self.device = device

        self.image_fns = os.listdir(image_dir)
        self.image_fns = [x for x in self.image_fns if x.endswith('.png')]

        np.random.seed(9)
        shuffle_idx = np.arange(len(self.image_fns)).astype(int)
        np.random.shuffle(shuffle_idx)

        self.image_fns = list(np.array(self.image_fns)[shuffle_idx])

        self.colour_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
        self.crop = transforms.RandomCrop((self.im_size[0]//2, self.im_size[1]//2))
        self.hor_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.vert_flip = transforms.RandomVerticalFlip(p=1.0)

    def __getitem__(self, idx):
        image = Image.open(self.image_dir + self.image_fns[idx]).resize(self.im_size, resample=Image.BICUBIC)
        instances = Image.open(self.segmap_dir + self.image_fns[idx]).resize(self.im_size, resample=Image.NEAREST)
        labels = Image.fromarray((np.array(instances) > 0).astype(np.uint8))
        
        # transforms

        if self.transform:

            # hor-ver flip
            image, instances, labels = self.horizontal_flip(image, instances, labels)
            image, instances, labels = self.vertical_flip(image, instances, labels)

            # rotate
            image, instances, labels = self.random_rotation(image, instances, labels)

            # colour jitter
            image = self.colour_jitter(image)

            # random crop
            image, instances, labels = self.random_crop(image, instances, labels)

            # reduce quality
            # image = self.random_reduce_quality(image)

            # invert
            # if np.random.uniform() < 0.5:
            #     image = 255.0 - np.array(image)

        # scale
        image = np.array(image) / 255.0

        image = torch.FloatTensor(image).permute(2, 0, 1)
        
        instances = torch.LongTensor(np.array(instances))
        labels = torch.ByteTensor(np.array(labels))

        return image.to(self.device), instances.to(self.device), labels.to(self.device)

    def __len__(self):
        return len(self.image_fns)

    def horizontal_flip(self, image, instances, labels, p=0.5):
        if np.random.uniform() < p:
            image = self.hor_flip(image)
            instances = self.hor_flip(instances)
            labels = self.hor_flip(labels)
        return image, instances, labels

    def vertical_flip(self, image, instances, labels, p=0.5):
        if np.random.uniform() < p:
            image = self.vert_flip(image)
            instances = self.vert_flip(instances)
            labels = self.vert_flip(labels)
        return image, instances, labels

    def random_rotation(self, image, instances, labels):
        random_number = np.random.uniform()
        if random_number < 0.25:
            image = image.rotate(90)
            instances = instances.rotate(90)
            labels = labels.rotate(90)
        elif (random_number >= 0.25) & (random_number < 0.5):
            image = image.rotate(180)
            instances = instances.rotate(180)
            labels = labels.rotate(180)
        elif (random_number >= 0.5) & (random_number < 0.75):
            image = image.rotate(270)
            instances = instances.rotate(270)
            labels = labels.rotate(270)
        return image, instances, labels

    def random_crop(self, image, instances, labels):
        if np.random.uniform() <= 0.333:
            i, j, h, w = self.crop.get_params(image, self.crop.size)
            image_cropped = transforms.functional.crop(image, i, j, h, w).resize(self.im_size, resample=Image.BICUBIC)
            instances_cropped = transforms.functional.crop(instances, i, j, h, w).resize(self.im_size, resample=Image.NEAREST)
            labels_cropped = transforms.functional.crop(labels, i, j, h, w).resize(self.im_size, resample=Image.NEAREST)
            # recursively random crop until n instances > 0 (not a problem in almost all cases.)
            if len(np.unique(np.array(labels_cropped))) == 1:
                image_cropped, instances_cropped, labels_cropped = self.random_crop(image, instances, labels)
            return image_cropped, instances_cropped, labels_cropped
        else:
            return image, instances, labels

    def random_reduce_quality(self, image):
        scale_factor = np.random.choice([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1/2, 1/2, 1/2, 1/2, 1/4, 1/4])
        image = image.resize((int(self.im_size[0]*scale_factor), int(self.im_size[1]*scale_factor)), resample=Image.BICUBIC)
        image = image.resize(self.im_size, resample=Image.NEAREST)
        return image


class SEMDataset(Dataset):
    def __init__(self, path, im_size=(256), rotate=True, device='cuda'):
        self.path = path
        self.im_size = im_size
        self.rotate = rotate
        self.device = device
        self.image_fns = sorted(os.listdir(self.path))
        np.random.seed(9)
        shuffle_idx = np.arange(len(self.image_fns)).astype(int)
        np.random.shuffle(shuffle_idx)
        self.image_fns = list(np.array(self.image_fns)[shuffle_idx])

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.path + self.image_fns[idx]))
        h, w = image.shape[:2]
        # get random self.im_size[0] x self.im_size[1] patch
        h_start = np.random.randint(h-self.im_size[0])
        w_start = np.random.randint(w-self.im_size[1])
        image = image[h_start:h_start+self.im_size[0], w_start:w_start+self.im_size[1], :]
        if self.rotate:
            image = self.random_rotation(image)
        image = image / 255.0
        image = torch.FloatTensor(image).permute(2, 0, 1)
        return image.to(self.device)


    def random_rotation(self, image):
        image = Image.fromarray(image)
        random_number = np.random.uniform()
        if random_number < 0.25:
            image = image.rotate(90)
        elif (random_number >= 0.25) & (random_number < 0.5):
            image = image.rotate(180)
        elif (random_number >= 0.5) & (random_number < 0.75):
            image = image.rotate(270)
        return np.array(image)


class EMPSMaskRCNN(Dataset):
    
    def __init__(self, image_dir, mask_dir, im_size=(256, 256), device='cuda', transform=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.im_size = im_size
        self.device = device

        self.image_fns = os.listdir(image_dir)
        self.image_fns = [x for x in self.image_fns if x.endswith('.png')]

        np.random.seed(9)
        shuffle_idx = np.arange(len(self.image_fns)).astype(int)
        np.random.shuffle(shuffle_idx)

        self.image_fns = list(np.array(self.image_fns)[shuffle_idx])

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_dir + self.image_fns[idx]).resize(self.im_size, resample=Image.BICUBIC))
        mask = np.array(Image.open(self.mask_dir + self.image_fns[idx]).resize(self.im_size, resample=Image.NEAREST))
        obj_ids = np.unique(mask)[1:]
        
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
        labels = torch.ones((num_objs,)).int()
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
