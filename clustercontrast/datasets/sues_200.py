import pickle
import json
import os
import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import copy
from tqdm import tqdm
import time
import re
import random
from typing import List, Dict, Tuple, Optional, Any # For type hinting
from clustercontrast.utils.data.color_conversion import apply_color_transfer_to_drone

import pdb


def get_data(path):
    data = {}
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            data[name] = {"path": os.path.join(root, name)}
            for _, _, files in os.walk(data[name]["path"], topdown=False):
                data[name]["files"] = files

    return data

# class U1652DatasetTrain(Dataset):
    
#     def __init__(self,
#                  data_list=None,
#                  transforms_query=None,
#                  transforms_gallery=None,
#                  prob_flip=0.5,
#                  shuffle_batch_size=128):
#         super().__init__()
 

#         self.pairs = data_list  # List of tuples (label, gallery_path, uav_path)
#         self.transforms_query = transforms_query
#         self.transforms_gallery = transforms_gallery
#         self.prob_flip = prob_flip
#         self.shuffle_batch_size = shuffle_batch_size
        
#         self.samples = copy.deepcopy(self.pairs)
#         #self.samples = torch.load("/data1/chenqi/My_ADCA/samples.pth")
        
#     def __getitem__(self, index):
        
#         idx, query_img_path, gallery_img_path = self.samples[index]
        
#         # for query there is only one file in folder
#         query_img = cv2.imread(query_img_path)
#         query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        
#         gallery_img = cv2.imread(gallery_img_path)
#         gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)
        
#         if np.random.random() < self.prob_flip:
#             query_img = cv2.flip(query_img, 1)
#             gallery_img = cv2.flip(gallery_img, 1) 
        
#         # image transforms
#         if self.transforms_query is not None:
#             query_img = self.transforms_query(image=query_img)['image']
            
#         if self.transforms_gallery is not None:
#             gallery_img = self.transforms_gallery(image=gallery_img)['image']
        
#         return query_img, gallery_img, idx

#     def __len__(self):
#         return len(self.samples)
    
    
#     def shuffle(self, ):

#             '''
#             custom shuffle function for unique class_id sampling in batch
#             '''
            
#             print("\nShuffle Dataset:")
            
#             pair_pool = copy.deepcopy(self.pairs)
              
#             # Shuffle pairs order
#             random.shuffle(pair_pool)
           
            
#             # Lookup if already used in epoch
#             pairs_epoch = set()   
#             idx_batch = set()
     
#             # buckets
#             batches = []
#             current_batch = []
             
#             # counter
#             break_counter = 0
            
#             # progressbar
#             pbar = tqdm()
    
#             while True:
                
#                 pbar.update()
                
#                 if len(pair_pool) > 0:
#                     pair = pair_pool.pop(0)
                    
#                     idx, _, _ = pair
                    
#                     if idx not in idx_batch and pair not in pairs_epoch:
                        
#                         idx_batch.add(idx)
#                         current_batch.append(pair)
#                         pairs_epoch.add(pair)
            
#                         break_counter = 0
                        
#                     else:
#                         # if pair fits not in batch and is not already used in epoch -> back to pool
#                         if pair not in pairs_epoch:
#                             pair_pool.append(pair)
                            
#                         break_counter += 1
                        
#                     if break_counter >= 512:
#                         break
                   
#                 else:
#                     break

#                 if len(current_batch) >= self.shuffle_batch_size:
                
#                     # empty current_batch bucket to batches
#                     batches.extend(current_batch)
#                     idx_batch = set()
#                     current_batch = []
       
#             pbar.close()
            
#             # wait before closing progress bar
#             time.sleep(0.3)
            
#             self.samples = batches
            
#             print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples))) 
#             print("Break Counter:", break_counter)
#             print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
#             print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))  

class U1652DatasetTrain(Dataset):

    def __init__(self,
                 samples=None,  # List of tuples (label, query_img_path, gallery_img_path)
                 transforms_query=None,
                 transforms_gallery=None,
                 prob_flip=0.5,
                 shuffle_batch_size=128):
        super().__init__()


        self.pairs = samples

        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size

        self.samples = copy.deepcopy(self.pairs)

    def __getitem__(self, index):

        label, query_img_path, gallery_img_path = self.samples[index]

        # for query there is only one file in folder
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        gallery_img = cv2.imread(gallery_img_path)
        gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)

        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1)

            # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']

        if self.transforms_gallery is not None:
            gallery_img = self.transforms_gallery(image=gallery_img)['image']

        return query_img, gallery_img, label, label

    def __len__(self):
        return len(self.samples)

    def shuffle(self, ):

        '''
        custom shuffle function for unique class_id sampling in batch
        '''

        print("\nShuffle Dataset:")

        pair_pool = copy.deepcopy(self.pairs)

        # Shuffle pairs order
        random.shuffle(pair_pool)

        # Lookup if already used in epoch
        pairs_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm()

        while True:

            pbar.update()

            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)

                idx, _, _ = pair

                if idx not in idx_batch and pair not in pairs_epoch:

                    idx_batch.add(idx)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)

                    break_counter = 0

                else:
                    # if pair fits not in batch and is not already used in epoch -> back to pool
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)

                    break_counter += 1

                if break_counter >= 512:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.3)

        self.samples = batches

        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))


def create_filtered_confidence_dataset_train(
    results_data: List[Tuple[int, str, str, float]],  # List of tuple as described above
    confidence_threshold: float = 0.0,
    transforms_query=None,
    transforms_gallery=None,
    prob_flip: float = 0.5,
    shuffle_batch_size: int = 128,
    output_samples_path: str = "samples2.pth"  # Path to save the filtered samples
):
    """
    Create a filtered confidence dataset for training.

    Args:
    - results_data (List[Tuple[int, str, str, float]]): List of tuples containing label, UAV path, gallery path, and confidence.
    - confidence_threshold (float): Minimum confidence required for a pair to be included.
    - transforms_query: Transformations to apply to query images.
    - transforms_gallery: Transformations to apply to gallery images.
    - prob_flip (float): Probability of flipping images horizontally.
    - shuffle_batch_size (int): Batch size for shuffling.
    - output_samples_path (str): Path to save the filtered samples.

    Returns:
    - samples (List[Tuple[Any, str, str]]): List of filtered samples with label, gallery path, and UAV path.
    """

    print(f"Original number of potential pairs from input data: {len(results_data[0])}")

    pairs = []  # (label, uav_path, gallery_path)
    correct_matches = 0  # Initialize correct matches counter
    for item in results_data[0]:
        uav_path = item[1]
        gallery_path = item[2]
        confidence = item[3]  # Default to -1 if not present
        label = item[0]  # Direct numerical label

        if uav_path is None or gallery_path is None or label is None:
            print(f"Warning: Skipping item due to missing path or label: {item}")
            continue

        if not isinstance(label, (int,)):  # Ensure label is a suitable type
            try:
                label = int(label)
            except ValueError:
                print(f"Warning: Skipping item due to non-integer label that cannot be converted: {item}")
                continue
        
        if confidence >= confidence_threshold and gallery_path == results_data[1].get(uav_path, None):
            pairs.append((label, gallery_path, uav_path))

            uav_path = uav_path
            sat_path = gallery_path

            # 从路径中提取数字ID，例如 /path/to/drone/123/image.jpg -> 123
            # 正则表达式 r'/drone/(\d+)' 匹配 "/drone/" 后紧跟的一串数字
            uav_id_match = re.search(r'/drone/(\d+)', uav_path)
            sat_id_match = re.search(r'/satellite/(\d+)', sat_path)

            if uav_id_match and sat_id_match:
                uav_id_str = uav_id_match.group(1) # group(1) 获取第一个括号匹配的内容
                sat_id_str = sat_id_match.group(1)

                if uav_id_str == sat_id_str:
                    correct_matches += 1
            else:
                # 如果路径格式不符合预期，打印警告
                print(f"警告：无法从以下路径中提取ID进行匹配度检查: UAV='{uav_path}', Satellite='{sat_path}'")
            
    accuracy = (correct_matches / len(pairs)) * 100 if len(pairs) > 0 else 0.0

    print(f"Number of pairs after applying confidence threshold ({confidence_threshold}): {len(pairs)} and accuracy: {accuracy:.2f}%")

    if not pairs:
        print("Warning: No pairs left after filtering. Dataset will be empty.")
        return []

    samples = copy.deepcopy(pairs)
    if not samples:
        print("Warning: samples is unexpectedly empty after initialization despite pairs not being empty.")
        return []
    # --- 使用 torch.save() 保存 self.samples ---
    if samples:
        try:
            torch.save(samples, output_samples_path)
            print(f"Successfully saved {len(samples)} samples to {output_samples_path} using torch.save().")
        except IOError as e: # torch.save can raise IOError
            print(f"Error: Could not write samples to {output_samples_path}. Reason: {e}")
        except Exception as e: # Catch other potential errors during pickling by torch.save
            print(f"An unexpected error occurred while saving samples with torch.save(): {e}")
    else:
        print(f"No samples to save to {output_samples_path} (self.samples is empty).")
    # --- 保存结束 ---

    return U1652DatasetTrain(samples,
                                      transforms_query=transforms_query,
                                      transforms_gallery=transforms_gallery,
                                      prob_flip=prob_flip,
                                      shuffle_batch_size=shuffle_batch_size,
                                      )


class U1652DatasetEval(Dataset):

    def __init__(self,
                 data_folder,
                 mode,
                 transforms=None,
                 sample_ids=None,
                 gallery_n=-1):
        super().__init__()

        self.data_dict = get_data(data_folder)

        # use only folders that exists for both gallery and query
        self.ids = list(self.data_dict.keys())

        self.transforms = transforms

        self.given_sample_ids = sample_ids

        self.images = []
        self.sample_ids = []

        self.mode = mode

        self.gallery_n = gallery_n

        for i, sample_id in enumerate(self.ids):

            for j, file in enumerate(self.data_dict[sample_id]["files"]):
                self.images.append("{}/{}".format(self.data_dict[sample_id]["path"],
                                                  file))

                self.sample_ids.append(sample_id)

    def __getitem__(self, index):

        img_path = self.images[index]
        sample_id = self.sample_ids[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # if self.mode == "sat":

        #    img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        #    img180 = cv2.rotate(img90, cv2.ROTATE_90_CLOCKWISE)
        #    img270 = cv2.rotate(img180, cv2.ROTATE_90_CLOCKWISE)

        #    img_0_90 = np.concatenate([img, img90], axis=1)
        #    img_180_270 = np.concatenate([img180, img270], axis=1)

        #    img = np.concatenate([img_0_90, img_180_270], axis=0)

        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        label = int(sample_id)
        if self.given_sample_ids is not None:
            if sample_id not in self.given_sample_ids:
                label = -1

        return img, label

    def __len__(self):
        return len(self.images)

    def get_sample_ids(self):
        return set(self.sample_ids)
    

def get_transforms(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):
    weather_id = 0
    val_transforms = A.Compose([A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),

                                # Multi-weather U1652 settings.
                                # iaa_weather_list[weather_id],

                                A.Normalize(mean, std),
                                ToTensorV2(),
                                ])

    train_sat_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),

                                      # Multi-weather U1652 settings.
                                      # A.OneOf(iaa_weather_list, p=1.0),

                                      A.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.3,
                                                    always_apply=False, p=0.5),
                                      A.OneOf([
                                          A.AdvancedBlur(p=1.0),
                                          A.Sharpen(p=1.0),
                                      ], p=0.3),
                                      A.OneOf([
                                          A.GridDropout(ratio=0.4, p=1.0),
                                          A.CoarseDropout(max_holes=25,
                                                          max_height=int(0.2 * img_size[0]),
                                                          max_width=int(0.2 * img_size[0]),
                                                          min_holes=10,
                                                          min_height=int(0.1 * img_size[0]),
                                                          min_width=int(0.1 * img_size[0]),
                                                          p=1.0),
                                      ], p=0.3),
                                      A.RandomRotate90(p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                      ])

    train_drone_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),

                                        # Multi-weather U1652 settings.
                                        # A.OneOf(iaa_weather_list, p=1.0),

                                        A.ColorJitter(brightness=0.15, contrast=0.7, saturation=0.3, hue=0.3,
                                                      always_apply=False, p=0.5),
                                        A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                        ], p=0.3),
                                        A.OneOf([
                                            A.GridDropout(ratio=0.4, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2 * img_size[0]),
                                                            max_width=int(0.2 * img_size[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1 * img_size[0]),
                                                            min_width=int(0.1 * img_size[0]),
                                                            p=1.0),
                                        ], p=0.3),
                                        A.Normalize(mean, std),
                                        ToTensorV2(),
                                        ])

    return val_transforms, train_sat_transforms, train_drone_transforms