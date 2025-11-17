import argparse
import os.path as osp
import random
import re
import shutil
from clustercontrast.evaluate.university import evaluate
from clustercontrast.models.objectives import InfoNCE
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
import numpy as np
import sys
import collections
import time
import datetime
from datetime import timedelta
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

import logging

from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
import json

from sample4geo.loss.DSA_loss import DSA_loss
from sample4geo.loss.blocks_infoNCE import blocks_InfoNCE
from sample4geo.loss.triplet_loss import TripletLoss
from sample4geo.model import TimmModel

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 设置日志格式
    datefmt='%Y-%m-%d %H:%M:%S',  # 设置日期格式
    handlers=[
        logging.FileHandler('app.log'),  # 将日志写入文件
    ]
)
import os
import sys

from sklearn.cluster import DBSCAN
import torch
from torch.cuda.amp import GradScaler
import cv2
import torch.nn.functional as F
from torch.backends import cudnn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional, Union

from clustercontrast.datasets.sues_200_drone import sues_drone
from clustercontrast.datasets.sues_200_satellite import sues_satellite
from clustercontrast.trainners import ClusterContrastTrainer_intra_view, get_test_loader, create_model, get_test_loader_drone, get_train_loader_satellite, get_train_loader_drone, train
from clustercontrast.evaluators import extract_features, extract_features_drone
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.color_conversion import calculate_global_lab_stats
from clustercontrast.datasets.university_1652 import U1652DatasetEval, create_filtered_confidence_dataset_train, get_transforms
from clustercontrast.models.model import view_Classifier
import pdb


# 创建一个 logger
logger = logging.getLogger(__name__)

def graph_filtering(
    features_drone1: torch.Tensor,
    features_drone2: torch.Tensor,
    pseudo_labels_drone: np.ndarray,
    uav_image_paths_list: List[str],
    features_satellite: torch.Tensor,
    satellite_image_paths_list: List[str],
    k_neighbors: int = 2,
    output_json_path: Optional[str] = None, # Path to save the output JSON file
) -> List[Tuple[int, str, str, float]]:
    """
    通过图过滤和聚类投票为无人机数据生成高质量的伪标签对。

    该函数执行以下步骤：
    1. 计算两个相似度矩阵：(无人机 vs 卫星) 和 (伪卫星 vs 卫星)。
    2. 使用相互近邻法（mutual k-nearest neighbor）进行图过滤，找到高置信度的匹配。
    3. 根据提供的无人机聚类标签，对每个簇进行加权投票，确定最佳的卫星目标。
    4. 为每个无人机图像生成一个包含目标索引、路径和置信度的结构化元组。

    Args:
        features_drone1 (torch.Tensor): 真实无人机视角的特征 (F_U). Shape: [B_uav, D].
        features_drone2 (torch.Tensor): 伪卫星视角的特征 (F_C). Shape: [B_uav, D].
        pseudo_labels_drone (np.ndarray): DBSCAN等算法生成的无人机聚类标签. Shape: [B_uav].
        uav_image_paths_list (List[str]): 无人机图片的路径列表，与特征一一对应.
        features_satellite (torch.Tensor): 真实卫星图片的特征 (F_S). Shape: [B_sat, D].
        satellite_image_paths_list (List[str]): 卫星图片的路径列表，与特征一一对应.
        k_neighbors (int, optional): 在相互近邻过滤中使用的K值. Defaults to 2.

    Returns:
        List[Tuple[int, str, str, float]]: 一个包含所有生成的伪标签对的列表。
            每个元组的格式为: (target_satellite_idx, uav_item_path, satellite_item_path, confidence)。
    """
    print("--- Starting Graph Filtering Process ---")
    device = features_drone1.device

    # --- 步骤 0: 特征归一化 ---
    # L2归一化是计算余弦相似度的前提
    features_drone1 = torch.nn.functional.normalize(features_drone1, p=2, dim=1)
    features_drone2 = torch.nn.functional.normalize(features_drone2, p=2, dim=1)
    features_satellite = torch.nn.functional.normalize(features_satellite, p=2, dim=1)
    
    num_uav = features_drone1.size(0)
    num_sat = features_satellite.size(0)
    print(f"Processing {num_uav} drone images and {num_sat} satellite images.")

    # --- 步骤 1: 计算相似度矩阵 ---
    print("Step 1: Calculating similarity matrices...")
    # Sim_AB: F_U vs F_S
    sim_ab = torch.matmul(features_drone1, features_satellite.T)

    # best_satellite_indices 形状: [B_uav], 值为匹配到的卫星图片在 satellite_features_tensor 中的索引
    best_satellite_similarities, best_satellite_indices = torch.max(sim_ab, dim=1)

    uav_to_satellite_map: Dict[str, str] = {}

    confidence_threshold = 0.6

    for uav_idx in range(num_uav):
        current_uav_path = uav_image_paths_list[uav_idx]
        matched_sat_idx = best_satellite_indices[uav_idx].item()
        similarity_score = best_satellite_similarities[uav_idx].item()

        # 判断相似度是否超过置信度阈值
        if similarity_score >= confidence_threshold:
            matched_satellite_path = satellite_image_paths_list[matched_sat_idx]
            uav_to_satellite_map[current_uav_path] = matched_satellite_path

    # Sim_CB: F_C vs F_S
    sim_cb = torch.matmul(features_drone2, features_satellite.T)

    # --- 步骤 2: 相互近邻过滤 ---
    print(f"Step 2: Performing mutual k-nearest neighbor filtering with K={k_neighbors}...")
    confident_matches = {}  # key: uav_idx, value: list of matched sat_indices
    for i in range(num_uav):
        # 从 Sim_AB 中找到 Top-K
        _, nn_ab_indices = torch.topk(sim_ab[i], k=k_neighbors)
        
        # 从 Sim_CB 中找到 Top-K
        _, nn_cb_indices = torch.topk(sim_cb[i], k=k_neighbors)
        
        # 计算交集。需要移动到CPU并转换为set进行操作
        intersection = set(nn_ab_indices.cpu().numpy()) & set(nn_cb_indices.cpu().numpy())
        
        if intersection:
            confident_matches[i] = list(intersection)

    print(f"Found {len(confident_matches)} drone images with high-confidence matches.")

    # --- 步骤 3: 根据伪标签对无人机进行分组 ---
    print("Step 3: Grouping drone images by their cluster labels...")
    # 忽略噪声点 (label == -1)
    num_clusters = pseudo_labels_drone.max() + 1
    uav_clusters = [[] for _ in range(num_clusters)]
    for uav_idx, label in enumerate(pseudo_labels_drone):
        if label != -1:
            uav_clusters[label].append(uav_idx)
    
    print(f"Grouped into {len(uav_clusters)} clusters (excluding noise).")

    # --- 步骤 4 & 5: 加权投票与生成最终训练对 ---
    print("Step 4 & 5: Performing weighted voting and generating final training pairs...")
    final_labeled_pairs = []
    
    for cluster_id, cluster_of_uav_indices in enumerate(uav_clusters):
        if not cluster_of_uav_indices:  # 跳过空簇
            continue

        vote_scores = {} # key: sat_idx, value: cumulative confidence score
        
        # 对簇内所有高置信度匹配进行投票
        for uav_idx in cluster_of_uav_indices:
            if uav_idx in confident_matches:
                for sat_idx in confident_matches[uav_idx]:
                    # 权重/置信度 = 融合后的相似度
                    weight = sim_ab[uav_idx, sat_idx].item() * sim_cb[uav_idx, sat_idx].item()
                    vote_scores[sat_idx] = vote_scores.get(sat_idx, 0) + weight
        
        # 如果该簇没有任何投票，则跳过
        if not vote_scores:
            continue
            
        # 确定获胜者
        target_satellite_idx = max(vote_scores, key=vote_scores.get)
        
        # 为该簇内的每个无人机样本生成训练对
        for uav_idx in cluster_of_uav_indices:
            # 获取文件路径
            uav_item_path = uav_image_paths_list[uav_idx]
            target_satellite_path = satellite_image_paths_list[target_satellite_idx]
            
            # 计算这个特定(uav, target_satellite)对的置信度
            pair_confidence = sim_ab[uav_idx, target_satellite_idx].item() * sim_cb[uav_idx, target_satellite_idx].item()

            labeled_pair = (
                target_satellite_idx,
                uav_item_path,
                target_satellite_path,
                pair_confidence
            )
            final_labeled_pairs.append(labeled_pair)

    print(f"--- Process Finished. Generated {len(final_labeled_pairs)} high-quality training pairs. ---")


    # 4. 计算正确匹配率并保存到 JSON 文件 (如果指定了路径)
    if output_json_path:
        if final_labeled_pairs:
            correct_matches = 0
            correct_matches_9 = 0 # 0.9 confidence threshold
            correct_matches_8 = 0 # 0.8 confidence threshold
            correct_matches_7 = 0 # 0.7 confidence threshold
            correct_matches_6 = 0 # 0.6 confidence threshold
            correct_matches_5 = 0 # 0.5 confidence threshold
            correct_matches_4 = 0 # 0.4 confidence threshold
            correct_matches_3 = 0 # 0.3 confidence threshold
            correct_matches_2 = 0 # 0.2 confidence threshold
            correct_matches_1 = 0 # 0.1 confidence threshold
            total_pairs = len(final_labeled_pairs)
            total_pairs_9 = 0
            total_pairs_8 = 0
            total_pairs_7 = 0
            total_pairs_6 = 0
            total_pairs_5 = 0
            total_pairs_4 = 0
            total_pairs_3 = 0
            total_pairs_2 = 0
            total_pairs_1 = 0

            for pair_data in final_labeled_pairs:
                if pair_data[3] >= 0.9:
                    total_pairs_9 += 1
                if pair_data[3] >= 0.8:
                    total_pairs_8 += 1
                if pair_data[3] >= 0.7:
                    total_pairs_7 += 1
                if pair_data[3] >= 0.6:
                    total_pairs_6 += 1
                if pair_data[3] >= 0.5:
                    total_pairs_5 += 1
                if pair_data[3] >= 0.4:
                    total_pairs_4 += 1
                if pair_data[3] >= 0.3:
                    total_pairs_3 += 1
                if pair_data[3] >= 0.2:
                    total_pairs_2 += 1
                if pair_data[3] >= 0.1:
                    total_pairs_1 += 1
                # pair_data is (uav_label, uav_item_path, satellite_item_path, confidence)
                uav_path = pair_data[1]
                sat_path = pair_data[2]

                # 从路径中提取数字ID，例如 /path/to/drone/123/image.jpg -> 123
                # 正则表达式 r'/drone/(\d+)' 匹配 "/drone/" 后紧跟的一串数字
                uav_id_match = re.search(r'/drone/(\d+)', uav_path)
                sat_id_match = re.search(r'/satellite/(\d+)', sat_path)

                if uav_id_match and sat_id_match:
                    uav_id_str = uav_id_match.group(1) # group(1) 获取第一个括号匹配的内容
                    sat_id_str = sat_id_match.group(1)

                    if uav_id_str == sat_id_str:
                        correct_matches += 1
                        if pair_data[3] >= 0.9:
                            correct_matches_9 += 1
                        if pair_data[3] >= 0.8:
                            correct_matches_8 += 1
                        if pair_data[3] >= 0.7:
                            correct_matches_7 += 1
                        if pair_data[3] >= 0.6:
                            correct_matches_6 += 1
                        if pair_data[3] >= 0.5:
                            correct_matches_5 += 1
                        if pair_data[3] >= 0.4:
                            correct_matches_4 += 1
                        if pair_data[3] >= 0.3:
                            correct_matches_3 += 1
                        if pair_data[3] >= 0.2:
                            correct_matches_2 += 1
                        if pair_data[3] >= 0.1:
                            correct_matches_1 += 1
                else:
                    # 如果路径格式不符合预期，打印警告
                    print(f"警告：无法从以下路径中提取ID进行匹配度检查: UAV='{uav_path}', Satellite='{sat_path}'")
            
            accuracy = (correct_matches / total_pairs) * 100 if total_pairs > 0 else 0.0

            # 获取当前时间
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # 准备要写入JSON的数据
            record_entry = {
                "timestamp": current_time,  # 添加当前时间戳
                "total_correct_matches": correct_matches,
                "total_pairs": total_pairs,
                "accuracy_percent": round(accuracy, 2), # 保留两位小数
                "0.9_correct_matches": correct_matches_9,
                "0.9_total_pairs": total_pairs_9,
                "0.9_accuracy_percent": round((correct_matches_9 / total_pairs_9) * 100, 2) if total_pairs_9 > 0 else 0.0, # 保留两位小数
                "0.8_correct_matches": correct_matches_8,
                "0.8_total_pairs": total_pairs_8,
                "0.8_accuracy_percent": round((correct_matches_8 / total_pairs_8) * 100, 2) if total_pairs_8 > 0 else 0.0, # 保留两位小数
                "0.7_correct_matches": correct_matches_7,
                "0.7_total_pairs": total_pairs_7,
                "0.7_accuracy_percent": round((correct_matches_7 / total_pairs_7) * 100, 2) if total_pairs_7 > 0 else 0.0,
                "0.6_correct_matches": correct_matches_6,
                "0.6_total_pairs": total_pairs_6,
                "0.6_accuracy_percent": round((correct_matches_6 / total_pairs_6) * 100, 2) if total_pairs_6 > 0 else 0.0,
                "0.5_correct_matches": correct_matches_5,
                "0.5_total_pairs": total_pairs_5,
                "0.5_accuracy_percent": round((correct_matches_5 / total_pairs_5) * 100, 2) if total_pairs_5 > 0 else 0.0,
                "0.4_correct_matches": correct_matches_4,
                "0.4_total_pairs": total_pairs_4,
                "0.4_accuracy_percent": round((correct_matches_4 / total_pairs_4) * 100, 2) if total_pairs_4 > 0 else 0.0,
                "0.3_correct_matches": correct_matches_3,
                "0.3_total_pairs": total_pairs_3,
                "0.3_accuracy_percent": round((correct_matches_3 / total_pairs_3) * 100, 2) if total_pairs_3 > 0 else 0.0,
                "0.2_correct_matches": correct_matches_2,
                "0.2_total_pairs": total_pairs_2,
                "0.2_accuracy_percent": round((correct_matches_2 / total_pairs_2) * 100, 2) if total_pairs_2 > 0 else 0.0,
                "0.1_correct_matches": correct_matches_1,
                "0.1_total_pairs": total_pairs_1,
                "0.1_accuracy_percent": round((correct_matches_1 / total_pairs_1) * 100, 2) if total_pairs_1 > 0 else 0.0
            }
            
            all_records = []
            try:
                # 尝试读取现有记录
                with open(output_json_path, 'r', encoding='utf-8') as f:
                    all_records = json.load(f)
                # 确保读取到的是一个列表
                if not isinstance(all_records, list):
                    print(f"警告：JSON文件 {output_json_path} 的内容不是一个列表。将创建一个新的列表。")
                    all_records = []
            except FileNotFoundError:
                # print(f"信息：JSON文件 {output_json_path} 未找到，将创建新文件。")
                all_records = [] # 文件不存在，则初始化为空列表
            except json.JSONDecodeError:
                print(f"警告：无法解码JSON文件 {output_json_path}。文件可能已损坏或为空。将创建一个新的列表。")
                all_records = [] # 文件内容损坏或为空，则初始化为空列表

            # 追加当前轮次的记录
            all_records.append(record_entry)

            try:
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(all_records, f, indent=4, ensure_ascii=False)
                print(f"正确匹配率信息已成功追加到: {output_json_path}")
            except IOError as e:
                print(f"错误：无法写入JSON文件 {output_json_path}. 原因: {e}")

        else: # contrastive_pairs_with_confidence 为空
            print(f"警告：没有生成图片对，因此无法计算匹配率。JSON文件 {output_json_path} 未更新。")
    
    return final_labeled_pairs, uav_to_satellite_map

# 假设cluster_labels是聚类后的结果
def log_cluster_statistics(cluster_labels, stage=""):
    cluster_count = collections.Counter(cluster_labels)
    logger.info(f"==> {stage} Clustering Statistics:")
    logger.info(f"    Number of clusters: {len(cluster_count)}")
    logger.info(f"    Cluster size distribution: {dict(cluster_count)}")

def associated_analysis_for_all(all_origin, all_pred, image_paths_for_all, log_dir):
    label_count_all = -1
    all_label_set = list(set(all_pred))
    all_label_set.sort()
    class_NIRVIS_list_modal_all = []
    associate = 0
    flag_satellite_list = collections.defaultdict(list)
    flag_drone_list = collections.defaultdict(list)
    for idx_, lab_ in enumerate(all_label_set):
        label_count_all += 1
        class_NIRVIS_list_modal = []
        flag_ir = 0
        flag_rgb = 0
        for idx, lab in enumerate(all_pred):
            if lab_ == lab:
                if 'satellite' in image_paths_for_all[idx]:
                    flag_ir = 1
                    flag_satellite_list[idx_] = 1
                elif 'drone' in image_paths_for_all[idx]:
                    flag_rgb = 1
                    flag_drone_list[idx_] = 1
        class_NIRVIS_list_modal_all.extend([class_NIRVIS_list_modal])

        if flag_ir == 1 and flag_rgb == 1:
            associate = associate + 1

    logger.info('associate rate', associate / len(all_label_set))

    return flag_satellite_list, flag_drone_list

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    
    main_worker_stage1_intra_view(args)
    main_worker_stage2_inter_view(args)

def main_worker_stage1_intra_view(args):
    start_time = time.monotonic()

    iters = args.iters if (args.iters > 0) else None
    #pdb.set_trace()
    logger.info("==> Load unlabeled dataset")
    drone_dataset = sues_drone(root=args.dataset, altitude=args.altitude, logger=logger)
    logger.info(len(drone_dataset.dataset))
    #pdb.set_trace()
    satellite_dataset = sues_satellite(root=args.dataset, altitude=args.altitude, logger=logger)
    logger.info(len(satellite_dataset.dataset))
    #pdb.set_trace()


    #-----------------------------------------------------------------------------#
    # create model                                                                   #
    #-----------------------------------------------------------------------------#
    if args.handcraft_model is not True:
        print("\nModel: {}".format(args.model))
        model = TimmModel(args.model,
                          pretrained=True,
                          img_size=args.img_size)

    else:
        from sample4geo.hand_convnext.model import make_model

        model = make_model(args)
        print("\nModel:{}".format("adjust model: handcraft convnext-base"))
        
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (args.img_size, args.img_size)

    # Activate gradient checkpointing
    if args.grad_checkpointing:
        model.set_grad_checkpointing(True)

        # Data parallel
    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)

    # Model to device   
    model = model.to(args.device)

    print("\nImage Size Query:", img_size)
    print("Image Size Ground:", img_size)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std))

    # define view classifier
    net_view_classifier = view_Classifier(embed_dim=1024, modal_class=3)
    net_view_classifier = net_view_classifier.to(args.device)

    #-----------------------------------------------------------------------------#
    # optimizer                                                                   #
    #-----------------------------------------------------------------------------#

    if args.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_parameters, lr=args.lr)

    elif args.lr_mlp is not None:
        model_params = []
        mlp_params = []
        for name, param in model.named_parameters():
            if 'back_mlp' in name:  # 根据参数名中是否包含 'mlp' 区分模型和 MLP 层的参数
                mlp_params.append(param)
            else:
                model_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': model_params, 'lr': args.lr},
            {'params': mlp_params, 'lr': args.lr_mlp}
        ])

    elif args.lr_decouple is not None:
        model_params = []
        logit_scale = []
        weights_params = []
        for name, param in model.named_parameters():
            if 'logit_scale' in name:
                logit_scale.append(param)
            elif 'w_blocks' in name:
                weights_params.append(param)
            else:
                model_params.append(param)

        optimizer = torch.optim.AdamW([{'params': model_params, 'lr': args.lr},
                                       {'params': logit_scale, 'lr': args.lr_decouple},
                                       {'params': weights_params, 'lr': args.lr_blockweights}])


    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    view_classifier_optimizer = torch.optim.SGD(net_view_classifier.parameters(), lr=0.01, weight_decay=5e-4,
                                                   momentum=0.9, nesterov=True)


    #-----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    #-----------------------------------------------------------------------------#

    train_steps_per = (len(drone_dataset.dataset) / args.batch_size)
    train_steps = (len(drone_dataset.dataset) / args.batch_size) * args.epochs
    # warmup_steps = len(train_dataloader) * config.warmup_epochs
    warmup_steps = train_steps * args.warmup_epochs
       
    if args.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(args.lr, args.lr_end))
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end=args.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)

    elif args.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(args.lr))
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)

    elif args.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(args.lr))
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=warmup_steps)

    else:
        scheduler = None

    lr_scheduler_view = torch.optim.lr_scheduler.StepLR(view_classifier_optimizer, step_size=args.step_size,
                                                         gamma=0.1)

    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(args.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(args.epochs, train_steps))


    # Trainer
    trainer = ClusterContrastTrainer_intra_view(encoder=model, net_view_classifer=net_view_classifier, device=args.device)

    # color transfer
    global_satellite_stats = calculate_global_lab_stats(satellite_dataset.dataset)

    best_score = 0

    for epoch in range(args.epochs):
        with torch.no_grad():
            if epoch == 0:
                # DBSCAN cluster
                d_eps = 0.77
                logger.info('Drone Clustering criterion: eps: {:.3f}'.format(d_eps))
                cluster_d = DBSCAN(eps=d_eps, min_samples=5, metric='precomputed', n_jobs=-1)
                s_eps = 0.7
                logger.info('Satellite Clustering criterion: eps: {:.3f}'.format(s_eps))
                cluster_s = DBSCAN(eps=s_eps, min_samples=6, metric='precomputed', n_jobs=-1)

            #pdb.set_trace()
            logger.info('==> Create pseudo labels for unlabeled drone data')

            cluster_loader_drone = get_test_loader(sorted(drone_dataset.dataset), height=args.height, width=args.width, batch_size=args.batch_size, num_workers=args.num_workers)
            features_drone, _ = extract_features(model, cluster_loader_drone, print_freq=50, device=args.device, logger=logger)
            del cluster_loader_drone
            features_drone = torch.cat([features_drone[f].unsqueeze(0) for f, _ in sorted(drone_dataset.dataset)], 0)

            
            logger.info('==> Create pseudo labels for unlabeled satellite data')
            
            cluster_loader_satellite = get_test_loader(sorted(satellite_dataset.dataset), height=args.height, width=args.width, batch_size=args.batch_size, num_workers=args.num_workers)
            features_satellite, _ = extract_features(model, cluster_loader_satellite, print_freq=50, device=args.device, logger=logger)
            del cluster_loader_satellite
            features_satellite = torch.cat([features_satellite[f].unsqueeze(0) for f, _ in sorted(satellite_dataset.dataset)], 0)

            # Replicate the satellite features 50 times
            features_satellite_replicated = features_satellite.repeat(50, 1)
            #pdb.set_trace()

            rerank_dist_satellite = compute_jaccard_distance(features_satellite_replicated, k1=args.k1, k2=args.k2, search_option=3, logger=logger)#rerank_dist_all_jacard[features_rgb.size(0):,features_rgb.size(0):]#
            pseudo_labels_satellite = cluster_s.fit_predict(rerank_dist_satellite)
            rerank_dist_drone = compute_jaccard_distance(features_drone, k1=args.k1, k2=args.k2, search_option=3, logger=logger)#rerank_dist_all_jacard[:features_rgb.size(0),:features_rgb.size(0)]#
            pseudo_labels_drone = cluster_d.fit_predict(rerank_dist_drone)


            # 在伪标签生成后调用
            log_cluster_statistics(pseudo_labels_satellite, stage="Satellite")
            log_cluster_statistics(pseudo_labels_drone, stage="Drone")
            del rerank_dist_drone
            del rerank_dist_satellite

            num_cluster_s = len(set(pseudo_labels_satellite)) - (1 if -1 in pseudo_labels_satellite else 0)
            num_cluster_d = len(set(pseudo_labels_drone)) - (1 if -1 in pseudo_labels_drone else 0)
            logger.info('Number of clusters in satellite: %d' % num_cluster_s)
            logger.info('Number of clusters in drone: %d' % num_cluster_d)

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features_satellite = generate_cluster_features(pseudo_labels_satellite, features_satellite_replicated)
        cluster_features_drone = generate_cluster_features(pseudo_labels_drone, features_drone)


        del features_satellite, features_drone, features_satellite_replicated


        # Define clustering memory
        memory_satellite = ClusterMemory(1024, num_cluster_s, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).to(args.device)
        memory_drone = ClusterMemory(1024, num_cluster_d, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).to(args.device)
        memory_satellite.features = F.normalize(cluster_features_satellite, dim=1).to(args.device)
        memory_drone.features = F.normalize(cluster_features_drone, dim=1).to(args.device)

        trainer.memory_satellite = memory_satellite
        trainer.memory_drone = memory_drone

        pseudo_labeled_dataset_satellite = []
        satellite_label=[]
        for i, ((fname, _), label) in enumerate(zip(sorted(satellite_dataset.dataset), pseudo_labels_satellite[0: len(satellite_dataset.dataset)])):
            if label != -1:
                pseudo_labeled_dataset_satellite.append((fname, label.item()))
                satellite_label.append(label.item())
        logger.info('==> Statistics for Satellite epoch {}: {} clusters'.format(epoch, num_cluster_s))

        pseudo_labeled_dataset_drone = []
        drone_label=[]
        for i, ((fname, _), label) in enumerate(zip(sorted(drone_dataset.dataset), pseudo_labels_drone)):
            if label != -1:
                pseudo_labeled_dataset_drone.append((fname, label.item()))
                drone_label.append(label.item())

        logger.info('==> Statistics for Drone epoch {}: {} clusters'.format(epoch, num_cluster_d))

        ########################
        
        # Transforms
        val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)

        train_loader_satellite = get_train_loader_satellite(args, satellite_dataset.dataset, args.height, args.width,
                                        args.batch_size, args.num_workers, iters,
                                        trainset=pseudo_labeled_dataset_satellite, train_transformer=train_sat_transforms)

        train_loader_drone = get_train_loader_drone(args, drone_dataset.dataset, args.height, args.width,
                                        args.batch_size, args.num_workers, iters,
                                        trainset=pseudo_labeled_dataset_drone, train_transformer=train_drone_transforms, global_satellite_stats=global_satellite_stats)
        
        train_loader_satellite.new_epoch()
        train_loader_drone.new_epoch()

        trainer.train(epoch, train_loader_satellite, train_loader_drone, optimizer, view_classifier_optimizer, print_freq=args.print_freq, train_iters=iters, adv_flag=args.adv_flag, logger=logger)

        if epoch>=0 and ( (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
##############################
            test_batch=64
            query_folder_test=f'{args.data_folder}/{args.dataset_name}/Testing/{args.altitude}/query_drone'
            gallery_folder_test=f'{args.data_folder}/{args.dataset_name}/Testing/{args.altitude}/gallery_satellite'

            # Reference Satellite Images
            query_dataset_test = U1652DatasetEval(data_folder=query_folder_test,
                                                    mode="query",
                                                    transforms=val_transforms,
                                                    )
            
            query_dataloader_test = DataLoader(query_dataset_test,
                                            batch_size=test_batch,
                                            num_workers=args.num_workers,
                                            shuffle=False,
                                            pin_memory=True)
            
            # Query Ground Images Test
            gallery_dataset_test = U1652DatasetEval(data_folder=gallery_folder_test,
                                                    mode="gallery",
                                                    transforms=val_transforms,
                                                    sample_ids=query_dataset_test.get_sample_ids())
            
            gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                            batch_size=test_batch,
                                            num_workers=args.num_workers,
                                            shuffle=False,
                                            pin_memory=True)
            
            logger.info("Query Images Test: %d", len(query_dataset_test))
            logger.info("Gallery Images Test: %d", len(gallery_dataset_test))
        

            logger.info("\n{}[{}]{}".format(30*"-", "University-1652", 30*"-"))  

            r1_test = evaluate(config=args,
                       model=model,
                       query_loader=query_dataloader_test,
                       gallery_loader=gallery_dataloader_test, 
                       ranks=[1, 5, 10],
                       step_size=1000,
                       cleanup=True,
                       logger=logger)
            if r1_test > best_score:
                best_score = r1_test
                logger.info("Best score: %f", best_score)
                # Save the model
                if torch.cuda.device_count() > 1 and len(args.gpu_ids) > 1:
                    torch.save(model.module.state_dict(), '{}/model_best.pth'.format(args.logs_dir))
                else:
                    torch.save(model.state_dict(), '{}/model_best.pth'.format(args.logs_dir)) 
       
        scheduler.step()
        lr_scheduler_view.step()
        
    end_time = time.monotonic()
    print('Total running timr: ', timedelta(seconds=end_time - start_time))


def main_worker_stage2_inter_view(args):
    start_time = time.monotonic()

    model_path = "{}/{}/{}".format(args.model_path,
                                   args.model,
                                   time.strftime("%m%d%H%M%S"))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))

    args.checkpoint_start = osp.join(args.logs_dir, 'model_best.pth')
    args.epochs = 6

    iters = args.iters if (args.iters > 0) else None
    #pdb.set_trace()
    logger.info("==> Load unlabeled dataset")
    drone_dataset = sues_drone(root=args.dataset, altitude=args.altitude, logger=logger)
    logger.info(len(drone_dataset.dataset))
    #pdb.set_trace()
    satellite_dataset = sues_satellite(root=args.dataset, altitude=args.altitude, logger=logger)
    logger.info(len(satellite_dataset.dataset))

    # color transfer
    global_satellite_stats = calculate_global_lab_stats(satellite_dataset.dataset)

    #-----------------------------------------------------------------------------#
    # create model                                                                   #
    #-----------------------------------------------------------------------------#
    if args.handcraft_model is not True:
        print("\nModel: {}".format(args.model))
        model = TimmModel(args.model,
                          pretrained=True,
                          img_size=args.img_size)

    else:
        from sample4geo.hand_convnext.model import make_model

        model = make_model(args)
        print("\nModel:{}".format("adjust model: handcraft convnext-base"))
        
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (args.img_size, args.img_size)

    # Activate gradient checkpointing
    if args.grad_checkpointing:
        model.set_grad_checkpointing(True)

    # Load pretrained Checkpoint    
    if args.checkpoint_start is not None:
        print("Start from:", args.checkpoint_start)
        model_state_dict = torch.load(args.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)

        # Data parallel
    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)

    # Model to device   
    model = model.to(args.device)

    print("\nImage Size Query:", img_size)
    print("Image Size Ground:", img_size)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std))

    # -----------------------------------------------------------------------------#
    # Loss                                                                        #
    # -----------------------------------------------------------------------------#
    # 1.infoNCE
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    loss_fn1 = InfoNCE(loss_function=loss_fn, device=args.device)

    # 2.Triplet
    loss_fn2 = TripletLoss(margin=args.triplet_loss)

    # 3.block infoNCE
    loss_fn3 = blocks_InfoNCE(loss_function=loss_fn, device=args.device)

    # 4.DSA loss infoNCE
    loss_fn4 = DSA_loss(loss_function=loss_fn, device=args.device)

    # all loss functions
    loss_functions = {"infoNCE": loss_fn1, "Triplet": loss_fn2, "blocks_infoNCE": loss_fn3, "DSA_loss": loss_fn4}

    if args.mixed_precision:
        scaler = GradScaler(init_scale=2. ** 10)
    else:
        scaler = None

    #-----------------------------------------------------------------------------#
    # optimizer                                                                   #
    #-----------------------------------------------------------------------------#

    if args.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_parameters, lr=args.lr)

    elif args.lr_mlp is not None:
        model_params = []
        mlp_params = []
        for name, param in model.named_parameters():
            if 'back_mlp' in name:  # 根据参数名中是否包含 'mlp' 区分模型和 MLP 层的参数
                mlp_params.append(param)
            else:
                model_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': model_params, 'lr': args.lr},
            {'params': mlp_params, 'lr': args.lr_mlp}
        ])

    elif args.lr_decouple is not None:
        model_params = []
        logit_scale = []
        weights_params = []
        for name, param in model.named_parameters():
            if 'logit_scale' in name:
                logit_scale.append(param)
            elif 'w_blocks' in name:
                weights_params.append(param)
            else:
                model_params.append(param)

        optimizer = torch.optim.AdamW([{'params': model_params, 'lr': args.lr},
                                       {'params': logit_scale, 'lr': args.lr_decouple},
                                       {'params': weights_params, 'lr': args.lr_blockweights}])


    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)



    #-----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    #-----------------------------------------------------------------------------#

    train_steps_per = (len(drone_dataset.dataset) / args.batch_size)
    train_steps = (len(drone_dataset.dataset) / args.batch_size) * args.epochs
    # warmup_steps = len(train_dataloader) * config.warmup_epochs
    warmup_steps = train_steps * args.warmup_epochs
       
    if args.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(args.lr, args.lr_end))
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end=args.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)

    elif args.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(args.lr))
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)

    elif args.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(args.lr))
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=warmup_steps)

    else:
        scheduler = None


    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(args.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(args.epochs, train_steps))
    

    best_score = 0

    for epoch in range(args.epochs):
        val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)

        if epoch == 0:
            test_batch=64
            query_folder_test=f'{args.data_folder}/{args.dataset_name}/Testing/{args.altitude}/query_drone'
            gallery_folder_test=f'{args.data_folder}/{args.dataset_name}/Testing/{args.altitude}/gallery_satellite'
            # Reference Satellite Images
            query_dataset_test = U1652DatasetEval(data_folder=query_folder_test,
                                                    mode="query",
                                                    transforms=val_transforms,
                                                    )
            
            query_dataloader_test = DataLoader(query_dataset_test,
                                            batch_size=test_batch,
                                            num_workers=args.num_workers,
                                            shuffle=False,
                                            pin_memory=True)
            
            # Query Ground Images Test
            gallery_dataset_test = U1652DatasetEval(data_folder=gallery_folder_test,
                                                    mode="gallery",
                                                    transforms=val_transforms,
                                                    sample_ids=query_dataset_test.get_sample_ids())
            
            gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                            batch_size=test_batch,
                                            num_workers=args.num_workers,
                                            shuffle=False,
                                            pin_memory=True)
            
            logger.info("Query Images Test: %d", len(query_dataset_test))
            logger.info("Gallery Images Test: %d", len(gallery_dataset_test))
        

            logger.info("\n{}[{}]{}".format(30*"-", "University-1652", 30*"-"))  

            r1_test = evaluate(config=args,
                       model=model,
                       query_loader=query_dataloader_test,
                       gallery_loader=gallery_dataloader_test, 
                       ranks=[1, 5, 10],
                       step_size=1000,
                       cleanup=True,
                       logger=logger)

        with torch.no_grad():
            if epoch == 0:
                # DBSCAN cluster
                d_eps = 0.77
                logger.info('Drone Clustering criterion: eps: {:.3f}'.format(d_eps))
                cluster_d = DBSCAN(eps=d_eps, min_samples=5, metric='precomputed', n_jobs=-1)
                s_eps = 0.7
                logger.info('Satellite Clustering criterion: eps: {:.3f}'.format(s_eps))
                cluster_s = DBSCAN(eps=s_eps, min_samples=6, metric='precomputed', n_jobs=-1)

            #pdb.set_trace()
            logger.info('==> Create pseudo labels for unlabeled drone data')

            cluster_loader_drone = get_test_loader_drone(sorted(drone_dataset.dataset), height=args.height, width=args.width, batch_size=args.batch_size, num_workers=args.num_workers, global_satellite_stats=global_satellite_stats)
            features_drone1, features_drone2, _ = extract_features_drone(model, cluster_loader_drone, print_freq=50, device=args.device, logger=logger)
            del cluster_loader_drone
            features_drone1 = torch.cat([features_drone1[f].unsqueeze(0) for f, _ in sorted(drone_dataset.dataset)], 0)
            features_drone2 = torch.cat([features_drone2[f].unsqueeze(0) for f, _ in sorted(drone_dataset.dataset)], 0)

            
            logger.info('==> Create pseudo labels for unlabeled satellite data')
            
            cluster_loader_satellite = get_test_loader(sorted(satellite_dataset.dataset), height=args.height, width=args.width, batch_size=args.batch_size, num_workers=args.num_workers)
            features_satellite, _ = extract_features(model, cluster_loader_satellite, print_freq=50, device=args.device, logger=logger)
            del cluster_loader_satellite
            features_satellite = torch.cat([features_satellite[f].unsqueeze(0) for f, _ in sorted(satellite_dataset.dataset)], 0)

            # Replicate the satellite features 50 times
            features_satellite_replicated = features_satellite.repeat(50, 1)
            #pdb.set_trace()

            rerank_dist_satellite = compute_jaccard_distance(features_satellite_replicated, k1=args.k1, k2=args.k2, search_option=3, logger=logger)#rerank_dist_all_jacard[features_rgb.size(0):,features_rgb.size(0):]#
            pseudo_labels_satellite = cluster_s.fit_predict(rerank_dist_satellite)
            rerank_dist_drone = compute_jaccard_distance(features_drone1, k1=args.k1, k2=args.k2, search_option=3, logger=logger)#rerank_dist_all_jacard[:features_rgb.size(0),:features_rgb.size(0)]#
            pseudo_labels_drone = cluster_d.fit_predict(rerank_dist_drone)


            # 在伪标签生成后调用
            log_cluster_statistics(pseudo_labels_satellite, stage="Satellite")
            log_cluster_statistics(pseudo_labels_drone, stage="Drone")
            del rerank_dist_drone
            del rerank_dist_satellite

            num_cluster_s = len(set(pseudo_labels_satellite)) - (1 if -1 in pseudo_labels_satellite else 0)
            num_cluster_d = len(set(pseudo_labels_drone)) - (1 if -1 in pseudo_labels_drone else 0)
            logger.info('Number of clusters in satellite: %d' % num_cluster_s)
            logger.info('Number of clusters in drone: %d' % num_cluster_d)

        
        print("--- 创建图片对并保存到JSON ---")
        output_file = 'contrastive_image_pairs.json'

        uav_image_paths_list_main = [fname for fname, _ in sorted(drone_dataset.dataset)]
        satellite_image_paths_list_main = [fname for fname, _ in sorted(satellite_dataset.dataset)]

        labels_satellite = pseudo_labels_satellite[: len(features_satellite)]

        results = graph_filtering(
            features_drone1=features_drone1,
            features_drone2=features_drone2,
            pseudo_labels_drone=pseudo_labels_drone,
            uav_image_paths_list=uav_image_paths_list_main,
            features_satellite=features_satellite,
            satellite_image_paths_list=satellite_image_paths_list_main,
            k_neighbors=2,
            output_json_path=output_file,
        )

        threshold = 0.3 - epoch * 0.05 
        print("Confidence threshold for epoch {}: {:.2f}".format(epoch, threshold))
        sample_path = osp.join(args.logs_dir, 'samples_epoch_{}.pth'.format(epoch))
        train_dataset = create_filtered_confidence_dataset_train(results_data=results, confidence_threshold=threshold, transforms_query=train_sat_transforms, 
                                                       transforms_gallery=train_drone_transforms, prob_flip=args.prob_flip, shuffle_batch_size=args.batch_size, output_samples_path=sample_path)
        train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=not args.custom_sampling,
                                  pin_memory=True)

        

        ######################################################################################################################################################################
        ######################################################################################################################################################################
        # Model                                                                       #
        # -----------------------------------------------------------------------------#

        if args.handcraft_model is not True:
            print("\nModel: {}".format(args.model))
            model = TimmModel(args.model,
                            pretrained=True,
                            img_size=args.img_size)
        else:
            from sample4geo.hand_convnext.model import make_model

            model = make_model(args)
            print("\nModel:{}".format("adjust model: handcraft convnext-base"))

        # -- print weight config infos
        print(
            f"\nweight_infonce:{args.weight_infonce}\nweight_gcc:{args.weight_cls}\nweight_dsa:{args.weight_dsa}\n")

        # print(model)

        data_config = model.get_config()
        print(data_config)
        mean = data_config["mean"]
        std = data_config["std"]
        img_size = (args.img_size, args.img_size)

        # Activate gradient checkpointing
        if args.grad_checkpointing:
            model.set_grad_checkpointing(True)

            # Data parallel
        print("GPUs available:", torch.cuda.device_count())
        if torch.cuda.device_count() > 1 and len(args.gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)

        # Model to device   
        model = model.to(args.device)

        print("\nImage Size Query:", img_size)
        print("Image Size Ground:", img_size)
        print("Mean: {}".format(mean))
        print("Std:  {}\n".format(std))

        # -----------------------------------------------------------------------------#
        # Loss                                                                        #
        # -----------------------------------------------------------------------------#
        # 1.infoNCE
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        loss_fn1 = InfoNCE(loss_function=loss_fn, device=args.device)

        # 2.Triplet
        loss_fn2 = TripletLoss(margin=args.triplet_loss)

        # 3.block infoNCE
        loss_fn3 = blocks_InfoNCE(loss_function=loss_fn, device=args.device)

        # 4.DSA loss infoNCE
        loss_fn4 = DSA_loss(loss_function=loss_fn, device=args.device)

        # all loss functions
        loss_functions = {"infoNCE": loss_fn1, "Triplet": loss_fn2, "blocks_infoNCE": loss_fn3, "DSA_loss": loss_fn4}

        if args.mixed_precision:
            scaler = GradScaler(init_scale=2. ** 10)
        else:
            scaler = None

        # -----------------------------------------------------------------------------#
        # optimizer                                                                   #
        # -----------------------------------------------------------------------------#

        if args.decay_exclue_bias:
            param_optimizer = list(model.named_parameters())
            no_decay = ["bias", "LayerNorm.bias"]
            optimizer_parameters = [
                {
                    "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    "weight_decay": 0.01,
                },
                {
                    "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            optimizer = torch.optim.AdamW(optimizer_parameters, lr=args.lr)

        elif args.lr_mlp is not None:
            model_params = []
            mlp_params = []
            for name, param in model.named_parameters():
                if 'back_mlp' in name:  # 根据参数名中是否包含 'mlp' 区分模型和 MLP 层的参数
                    mlp_params.append(param)
                else:
                    model_params.append(param)

            optimizer = torch.optim.AdamW([
                {'params': model_params, 'lr': args.lr},
                {'params': mlp_params, 'lr': args.lr_mlp}
            ])
        elif args.lr_decouple is not None:
            model_params = []
            logit_scale = []
            weights_params = []
            for name, param in model.named_parameters():
                if 'logit_scale' in name:
                    logit_scale.append(param)
                elif 'w_blocks' in name:
                    weights_params.append(param)
                else:
                    model_params.append(param)

            optimizer = torch.optim.AdamW([{'params': model_params, 'lr': args.lr},
                                        {'params': logit_scale, 'lr': args.lr_decouple},
                                        {'params': weights_params, 'lr': args.lr_blockweights}])


        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        # -----------------------------------------------------------------------------#
        # Scheduler                                                                   #
        # -----------------------------------------------------------------------------#

        # print(optimizer.param_groups[0]['lr'])

        train_steps_per = len(train_dataloader)
        train_steps = len(train_dataloader) * args.epochs
        # warmup_steps = len(train_dataloader) * config.warmup_epochs
        warmup_steps = train_steps * args.warmup_epochs

        if args.scheduler == "polynomial":
            print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(args.lr, args.lr_end))
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                                num_training_steps=train_steps,
                                                                lr_end=args.lr_end,
                                                                power=1.5,
                                                                num_warmup_steps=warmup_steps)

        elif args.scheduler == "cosine":
            print("\nScheduler: cosine - max LR: {}".format(args.lr))
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                        num_training_steps=train_steps,
                                                        num_warmup_steps=warmup_steps)

        elif args.scheduler == "constant":
            print("\nScheduler: constant - max LR: {}".format(args.lr))
            scheduler = get_constant_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=warmup_steps)

        else:
            scheduler = None

        print("Warmup Epochs: {} - Warmup Steps: {}".format(str(args.warmup_epochs).ljust(2), warmup_steps))
        print("Train Epochs:  {} - Train Steps:  {}".format(args.epochs, train_steps))

        # -----------------------------------------------------------------------------#
        # Shuffle                                                                     #
        # -----------------------------------------------------------------------------#
        if args.custom_sampling:
            train_dataloader.dataset.shuffle()
        ######################################################################################################################################################################
        ######################################################################################################################################################################

        train_loss = train(args,
                           model,
                           dataloader=train_dataloader,
                           loss_functions=loss_functions,
                           optimizer=optimizer,
                           epoch=epoch,
                           train_steps_per=train_steps_per,
                           scheduler=scheduler,
                           scaler=scaler)

        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch,
                                                                   train_loss,
                                                                   optimizer.param_groups[0]['lr']))

        if epoch >= 0 and ( (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
##############################
            test_batch=64
            query_folder_test=f'{args.data_folder}/{args.dataset_name}/Testing/{args.altitude}/query_drone'
            gallery_folder_test=f'{args.data_folder}/{args.dataset_name}/Testing/{args.altitude}/gallery_satellite'

            # Reference Satellite Images
            query_dataset_test = U1652DatasetEval(data_folder=query_folder_test,
                                                    mode="query",
                                                    transforms=val_transforms,
                                                    )
            
            query_dataloader_test = DataLoader(query_dataset_test,
                                            batch_size=test_batch,
                                            num_workers=args.num_workers,
                                            shuffle=False,
                                            pin_memory=True)
            
            # Query Ground Images Test
            gallery_dataset_test = U1652DatasetEval(data_folder=gallery_folder_test,
                                                    mode="gallery",
                                                    transforms=val_transforms,
                                                    sample_ids=query_dataset_test.get_sample_ids())
            
            gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                            batch_size=test_batch,
                                            num_workers=args.num_workers,
                                            shuffle=False,
                                            pin_memory=True)
            
            logger.info("Query Images Test: %d", len(query_dataset_test))
            logger.info("Gallery Images Test: %d", len(gallery_dataset_test))
        

            logger.info("\n{}[{}]{}".format(30*"-", "University-1652", 30*"-"))  

            r1_test = evaluate(config=args,
                       model=model,
                       query_loader=query_dataloader_test,
                       gallery_loader=gallery_dataloader_test, 
                       ranks=[1, 5, 10],
                       step_size=1000,
                       cleanup=True,
                       logger=logger)
            if r1_test > best_score:

                best_score = r1_test

                if torch.cuda.device_count() > 1 and len(args.gpu_ids) > 1:
                    torch.save(model.module.state_dict(),
                               '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
       
        
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised cross-view geo-localization")
    # Added for your modification
    parser.add_argument('--model', default='convnext_base.fb_in22k_ft_in1k_384', type=str, help='backbone model')
    parser.add_argument('--handcraft_model', default=True, type=bool, help='use modified backbone')
    parser.add_argument('--img_size', default=384, type=int, help='input image size')
    parser.add_argument('--views', default=2, type=int, help='only supports 2 branches retrieval')
    parser.add_argument('--record', default=True, type=bool, help='use tensorboard to record training procedure')

    # Model Config
    parser.add_argument('--nclasses', default=701, type=int, help='1652场景的类别数')
    parser.add_argument('--block', default=2, type=int)
    parser.add_argument('--triplet_loss', default=0.3, type=float)
    parser.add_argument('--resnet', default=False, type=bool)

    # Our tricks
    parser.add_argument('--weight_infonce', default=1.0, type=float)
    parser.add_argument('--weight_cls', default=0.1, type=float)
    parser.add_argument('--weight_dsa', default=0.6, type=float)

    # --
    parser.add_argument('--only_test', default=False, type=bool, help='use pretrained model to test')
    parser.add_argument('--ckpt_path',
                        default='checkpoints/university/convnext_base.fb_in22k_ft_in1k_384/0710120007/weights_e1_0.9169.pth',
                        type=str, help='path to pretrained checkpoint file')
    
     # Training Config
    parser.add_argument('--mixed_precision', default=True, type=bool)
    parser.add_argument('--custom_sampling', default=True, type=bool)
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    # parser.add_argument('--epochs', default=1, type=int, help='1 epoch for 1652')
    parser.add_argument('--batch_size', default=24, type=int, help='remember the bs is for 2 branches')
    parser.add_argument('--verbose', default=True, type=bool)
    parser.add_argument('--gpu_ids', default=(0, 4, 2, 3), type=tuple)

    # Eval Config
    parser.add_argument('--batch_size_eval', default=128, type=int)
    parser.add_argument('--eval_every_n_epoch', default=1, type=int)
    parser.add_argument('--normalize_features', default=True, type=bool)
    parser.add_argument('--eval_gallery_n', default=-1, type=int)

    # Optimizer Config
    parser.add_argument('--clip_grad', default=100.0, type=float)
    parser.add_argument('--decay_exclue_bias', default=False, type=bool)
    parser.add_argument('--grad_checkpointing', default=False, type=bool)

    # Loss Config
    parser.add_argument('--label_smoothing', default=0.1, type=float)

    # Learning Rate Config
    parser.add_argument('--lr', default=0.001, type=float, help='1 * 10^-4 for ViT | 1 * 10^-1 for CNN')
    parser.add_argument('--scheduler', default="cosine", type=str, help=r'"polynomial" | "cosine" | "constant" | None')
    parser.add_argument('--warmup_epochs', default=0.1, type=float)
    parser.add_argument('--lr_end', default=0.0001, type=float)

    # Learning part Config
    parser.add_argument('--lr_mlp', default=None, type=float)
    parser.add_argument('--lr_decouple', default=None, type=float)
    parser.add_argument('--lr_blockweights', default=2, type=float)


    # Dataset Config
    parser.add_argument('--data_folder', default=r'/data0/chenqi_data', type=str)
    parser.add_argument('--dataset_name', default='SUES-200-512x512', type=str)
    parser.add_argument('--altitude', default=200, type=int, help="150|200|250|300")

    # Augment Images Config
    parser.add_argument('--prob_flip', default=0.5, type=float, help='flipping the sat image and drone image simultaneously')

    # Savepath for model checkpoints Config
    parser.add_argument('--model_path', default='./checkpoints/sues-200', type=str)
    # Eval before training Config
    parser.add_argument('--zero_shot', default=False, type=bool)

    # Checkpoint to start from Config
    parser.add_argument('--checkpoint_start', default=None)

    # Set num_workers to 0 if on Windows Config
    parser.add_argument('--num_workers', default=0 if os.name == 'nt' else 4, type=int)

    # Train on GPU if available Config
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', type=str)

    # For better performance Config
    parser.add_argument('--cudnn_benchmark', default=True, type=bool)

    # Make cudnn deterministic Config
    parser.add_argument('--cudnn_deterministic', default=False, type=bool)

    # data
    parser.add_argument('-d', '--dataset', type=str, default='/data0/chenqi_data')
    parser.add_argument('--height', type=int, default=384, help="input height")
    parser.add_argument('--width', type=int, default=384, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")

    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    parser.add_argument('--adv-flag', type=bool, default=True)

    # optimizer
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs

    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--logs_dir', type=str, metavar='PATH', default="/data1/chenqi/DAC/check_point")
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")

    main()
