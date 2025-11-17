from __future__ import print_function, absolute_import
import time
import collections
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import copy

from .utils.meters import AverageMeter
from .utils import to_torch

import pdb

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_cnn_feature(model, inputs, device):
    inputs = to_torch(inputs).to(device)
    outputs = model(inputs) 
    feature = outputs[-2]
    feature = feature.data.cpu()
    return feature

def extract_features(model, data_loader, print_freq=50, flip=True, device="cuda:0", logger=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs, device)


            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] =  output.detach()
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                logger.info('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels

def extract_features_drone(model, data_loader, print_freq=50, flip=True, device="cuda:0", logger=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features1 = OrderedDict()
    features2 = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs1, imgs2, fnames, pids, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs1 = extract_cnn_feature(model, imgs1, device)
            outputs2 = extract_cnn_feature(model, imgs2, device)


            for fname, output1, output2, pid in zip(fnames, outputs1, outputs2, pids):
                features1[fname] =  output1.detach()
                features2[fname] =  output2.detach()
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                logger.info('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features1, features2, labels