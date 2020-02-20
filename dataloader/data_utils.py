# -*-coding=utf-8-*-
import torch
import torch.utils.data as data_utils
import h5py
import numpy as np
import random
import os
from .nyu_dataloader import NYUDataset

data_sets = ['nyudepth', 'make3d', 'kitti']

def create_relative_depth(depth, num_samples):
    if num_samples == 0:
        return None
		
    prob = get_prob(depth, num_samples)
    mask_keep = np.random.uniform(0, 1, depth.shape) < prob
    return mask_keep

def read_list(filename):
    f = open(filename, 'r')
    data = f.read().strip('\n')
    f.close()
    data_list = data.split('\n')
    return data_list

def get_dataloader(data_set, data_dir, batch_size, num_samples, workers):
    val_file = os.path.join(data_dir, 'val.txt')
    val_list = read_list(val_file)

    if data_set == 'nyudepth':
        val_dataset = NYUDataset(val_list, type='val', num_samples = num_samples)
        h, w = NYUDataset.h, NYUDataset.w
    else:
        print("please use the NYU-Depth-v2 dataset")


    val_loader = data_utils.DataLoader(val_dataset, 1, shuffle = False, num_workers = workers, drop_last = False)
    return len(val_list), val_loader, h, w
