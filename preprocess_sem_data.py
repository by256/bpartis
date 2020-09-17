"""
## Preprocesses the electron microscopy images from
## https://www.nature.com/articles/sdata2018172.pdf used for 
## unsupervised pre-training by getting rid of the 
## scalebar/microscope info.
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
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def remove_microscope_information(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    # print(contours)
    if (contours is None) or (len(contours) == 0):
        image = None
    else:
        areas = [cv2.contourArea(c) for c in contours]
        max_idx = np.argmax(areas)
        info_contour = contours[max_idx].squeeze()
        min_h = np.min(info_contour[:, 1])
        image = image[:min_h-20, :]

    return image

def filter_thin_images(image):
    if image is not None:
        h, w = image.shape[:2]
        if h < 512:
            image = None
    return image

def filter_all_white_images(image):
    """filter image if mostly white"""
    if image is not None:
        n_pixels = np.prod(image.shape)
        if np.sum((image == 255).astype(int)) > 0.95*n_pixels:
            image = None
    return image


parser = argparse.ArgumentParser(description='Preprocess the SEM dataset before training.')
parser.add_argument('--cat-dir', metavar='cat_dir', type=str, help='Directory which contains the individual category directories.')
parser.add_argument('--save-dst', metavar='save_dst', type=str, help='Directory to save preprocessed images to.')

namespace = parser.parse_args()

categories = os.listdir(namespace.cat_dir)
categories = [x for x in categories if not x.endswith('.tar')]

valid_extenstions = ['.png', '.jpg', '.jpeg', '.tif']

for cat in categories:
    cat_image_fns = os.listdir('{}{}/'.format(namespace.cat_dir, cat))
    cat_image_fns = [x for x in cat_image_fns if os.path.splitext(x)[1]][40:]
    for i, fn in enumerate(cat_image_fns):
        path = namespace.cat_dir + cat + '/' + fn
        try:
            image = np.array(Image.open(path))
            image = remove_microscope_information(image)
            image = filter_thin_images(image)
            image = filter_all_white_images(image)
            Image.fromarray(image).save('{}{}_{}'.format(namespace.save_dst, cat, fn))
            if image is None:
                continue
        except (IndexError, AttributeError) as e:
            continue
        print('{}: {}/{}'.format(cat, i+1, len(cat_image_fns)), end='\r', flush=True)
