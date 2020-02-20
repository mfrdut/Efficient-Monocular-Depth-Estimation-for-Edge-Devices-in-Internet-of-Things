# -*-coding=utf-8-*-
import torch
import torch.utils.data as data
import h5py
import numpy as np
import os
from . import data_utils
from . import transforms

iheight, iwidth = 480, 640 
oheight, owidth = 224, 224 
to_tensor = transforms.ToTensor()

def val_transform(rgb, depth):
    depth_trans = depth

    transform = transforms.Compose([
        # Resize
        transforms.Resize(240.0 / iheight),
        # CenterCrop
        transforms.CenterCrop((oheight, owidth)),
    ])
    depth_trans = transform(depth_trans)
    rgb_trans = transform(rgb)
    rgb_trans = np.asfarray(rgb_trans, dtype = 'float') / 255

    return rgb_trans, depth_trans

class NYUDataset(data.Dataset):
    h, w = oheight, owidth

    def __init__(self, data_list, type, num_samples):

        self.data_list = data_list
        if type == 'val':
            self.transform = val_transform

        self.num_samples = num_samples

    def __getitem__(self, index):
        h5_filename = self.data_list[index] 
        h5f = h5py.File(h5_filename, "r") 
        rgb = np.array(h5f['rgb'])
        rgb = np.transpose(rgb, (1, 2, 0)) 
        depth = np.array(h5f['depth'])

        rgb_raw = rgb.copy()
        rgb_raw_tensor = torch.from_numpy(rgb_raw)
		
        if self.transform is not None:
            rgb_np, depth_np = self.transform(rgb, depth)

        input_np = rgb_np
        mask_keep = data_utils.create_relative_depth(depth_np, self.num_samples) # num_samples ---relative loss

        input_tensor = to_tensor(input_np)
		
        if mask_keep is not None:
            mask_tensor = torch.from_numpy(mask_keep.astype(np.uint8).copy())
        else:
            mask_tensor = torch.Tensor([0])
			
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        depth_tensor = to_tensor(depth_np)
        depth_tensor = depth_tensor.unsqueeze(0)
		
        h5f.close()
        return rgb_raw_tensor, input_tensor, depth_tensor, mask_tensor, h5_filename.replace('/', '_') # filename

    def __len__(self):
        return len(self.data_list)