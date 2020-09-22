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

import time
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from ..data import EMPSMaskRCNN
from ..utils.train import train_test_split_emps, compute_decay_rate


