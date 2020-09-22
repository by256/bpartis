import sys
import time
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import m2py.segmentation.segmentation_gmm as seg_gmm
import m2py.segmentation.segmentation_watershed as seg_water
from m2py.utils import seg_label_utils as slu

sys.path.append('..')
from data import EMPSDataset
from utils.train import train_test_split_emps


parser = argparse.ArgumentParser(description='Train model on EMPS dataset.')
parser.add_argument('--data-dir', metavar='data_dir', type=str, help='Directory which contains the data.')
parser.add_argument('--device', metavar='device', type=str, default='cuda', help='device to train on (cuda or cpu)')
parser.add_argument('--im-size', metavar='im_size', type=tuple, default=(256, 256), help='Image size to load for training.')
parser.add_argument('--save-dir', metavar='save_dir', type=str, default='./saved_models/', help='directory to save and load weights from.')
namespace = parser.parse_args()

_, val_dataset = train_test_split_emps(EMPSDataset, 
                                       namespace.data_dir, 
                                       im_size=namespace.im_size, 
                                       device=namespace.device)

seg = seg_gmm.SegmenterGMM(n_components=2, embedding_dim=3,
                           nonlinear=True, normalize=True,
                           padding=0, zscale=False)

cmap = matplotlib.cm.tab20
cmap.set_bad(color='black')

for i, (image, instance, _) in enumerate(val_dataset):
    # if i <= 20:
    #     continue
    image = image.permute(1, 2, 0).cpu().numpy()
    instance = instance.cpu().numpy()
    start = time.time()
    labels = seg.fit_transform(image)#, pers_thresh=0.15)
    labels = seg.get_grains(labels)
    labels = slu.get_significant_labels(labels, bg_contrast_flag=True, label_thresh=50)
    uniq, counts = np.unique(labels, return_counts=True)
    labels[labels == uniq[np.argmax(counts)]] = np.nan
    print('{}/{}    Time Elapsed: {}'.format(i+1, len(val_dataset), time.time()-start))
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image, cmap='gray')
    axes[1].matshow(labels, cmap=cmap)
    plt.show()
