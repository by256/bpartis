import numpy as np

def shuffle_segmap(x):
    insts = np.unique(x)
    shuffle_idx = np.arange(1, len(insts))
    np.random.shuffle(shuffle_idx)
    shuffled_segmap = np.zeros_like(x)
    for i, idx in enumerate(shuffle_idx):
        inst_mask = x == idx
        shuffled_segmap[inst_mask] = i + 1
        
    return shuffled_segmap
