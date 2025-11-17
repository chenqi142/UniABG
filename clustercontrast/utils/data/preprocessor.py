from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image
import cv2
from clustercontrast.utils.data.color_conversion import apply_color_transfer_to_drone


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, id = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = cv2.imread(fpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(image=img)['image']

        return img, fname, id, index
    
class Preprocessor_drone(Dataset):
    def __init__(self, dataset, root=None, transform=None, global_satellite_stats=None):
        super(Preprocessor_drone, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.global_satellite_stats = global_satellite_stats

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, id = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = cv2.imread(fpath)
        transformed_drone = apply_color_transfer_to_drone(img, self.global_satellite_stats)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transformed_drone = cv2.cvtColor(transformed_drone, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(image=img)['image']
            transformed_drone = self.transform(image=transformed_drone)['image']

        return img, transformed_drone, fname, id, index

