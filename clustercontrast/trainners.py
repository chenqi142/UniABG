from clustercontrast.utils.meters import AverageMeter
from torch.utils.data import DataLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data.preprocessor import Preprocessor, Preprocessor_drone
from clustercontrast.models.model import TimmModel
from clustercontrast.models.objectives import compute_sdm
import timm
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from collections import OrderedDict

from torch.cuda.amp import autocast
from sample4geo.loss.cal_loss import cal_kl_loss, cal_loss, cal_triplet_loss

import pdb

def get_test_loader(dataset, height, width, batch_size, num_workers=4, test_transform=None):
    """
    Get the test data loader.
    """
    # 定义 albumentations 的 Normalize
    normalize = A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])

    if test_transform is None:
        test_transform = A.Compose([
            A.Resize(height=height, width=width, interpolation=3),
            normalize,
            ToTensorV2(),
        ])
    
    test_loader = DataLoader(
        Preprocessor(dataset, transform=test_transform),
        batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=True
    )

    return test_loader

def get_test_loader_drone(dataset, height, width, batch_size, num_workers=4, test_transform=None, global_satellite_stats=None):
    """
    Get the test data loader.
    """
    # 定义 albumentations 的 Normalize
    normalize = A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])

    if test_transform is None:
        test_transform = A.Compose([
            A.Resize(height=height, width=width, interpolation=3),
            normalize,
            ToTensorV2(),
        ])
    
    test_loader = DataLoader(
        Preprocessor_drone(dataset, transform=test_transform, global_satellite_stats=global_satellite_stats),
        batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=True
    )

    return test_loader

def get_train_loader_satellite(args, dataset, height, width, batch_size, workers,
                     iters, trainset=None, train_transformer=None):


    train_set = sorted(dataset) if trainset is None else sorted(trainset)
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers,
                   shuffle=True, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_train_loader_drone(args, dataset, height, width, batch_size, workers,
                     iters, trainset=None, train_transformer=None, global_satellite_stats=None):


    train_set = sorted(dataset) if trainset is None else sorted(trainset)
    train_loader = IterLoader(
        DataLoader(Preprocessor_drone(train_set, transform=train_transformer, global_satellite_stats=global_satellite_stats),
                   batch_size=batch_size, num_workers=workers,
                   shuffle=True, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def create_model(args):
    model = TimmModel(args.model_name,
                          pretrained=True,
                          img_size=args.img_size)

    return model

class ClusterContrastTrainer_intra_view(object):
    def __init__(self, encoder, net_view_classifer, memory=None, device="cuda:0"):
        super(ClusterContrastTrainer_intra_view, self).__init__()
        self.encoder = encoder
        self.memory_satellite = memory
        self.memory_drone = memory
        self.net_view_classifer = net_view_classifer
        self.criterion1=nn.CrossEntropyLoss()
        self.criterion1.to(device)
        self.device = device

    def train(self, epoch, data_loader_satellite, data_loader_drone, optimizer, view_classifier_optimizer, print_freq=10, train_iters=400, adv_flag=True, logger=None):
        self.encoder.train()
        self.net_view_classifer.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        # 0: satellite view 1: drone view 2: Intermediate view
        view_s_labels = torch.ones(24).long().to(self.device)
        view_d_labels = torch.zeros(24).long().to(self.device)
        view_i_labels = 2 * torch.ones(24).long().to(self.device)
        view_adv_loss = 0
        true_label = torch.cat([view_d_labels, view_i_labels, view_s_labels], dim=0)
        loss = 0
        loss_satellite = 0
        loss_drone = 0
        view_classifier_acc = 0
        view_loss = 0

        for i in range(train_iters):
            # load data
            inputs_satellite = data_loader_satellite.next()
            inputs_drone = data_loader_drone.next()


            data_time.update(time.time() - end)

            # process inputs
            inputs_satellite, labels_satellite, indexes_satellite = self._parse_data_satellite(inputs_satellite)
            inputs_drone1 ,inputs_drone2, labels_drone, indexes_drone = self._parse_data_drone(inputs_drone)
            # forward
            inputs_drone = torch.cat((inputs_drone1, inputs_drone2), 0)
            labels_drone = torch.cat((labels_drone, labels_drone), -1)

            f_out_drone, f_out_satellite, labels_drone, labels_satellite = self._forward(inputs_drone, inputs_satellite, label_1=labels_drone, label_2=labels_satellite)

            if epoch <= 1:
                loss_satellite = self.memory_satellite(f_out_satellite, labels_satellite)
                loss_drone = self.memory_drone(f_out_drone, labels_drone)
                loss = loss_satellite + loss_drone

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if adv_flag==True:
                    f_view_in = torch.cat([f_out_drone, f_out_satellite], dim=0)
                    predit_view_label = self.net_view_classifer(f_view_in.detach())


                    _, predict =predit_view_label.max(1)
                    view_classifier_acc = predict.eq(true_label).sum().item()/true_label.shape[0]


                    view_loss = self.criterion1(predit_view_label[:f_out_satellite.shape[0]], view_d_labels)\
                    +self.criterion1(predit_view_label[f_out_satellite.shape[0]:f_out_satellite.shape[0]*2], view_i_labels)\
                    +self.criterion1(predit_view_label[f_out_satellite.shape[0]*2:], view_s_labels)

                    view_classifier_optimizer.zero_grad()
                    view_loss.backward()
                    view_classifier_optimizer.step()

            else:
                if adv_flag==True:
                    f_view_in = torch.cat([f_out_drone, f_out_satellite], dim=0)
                    predit_view_label = self.net_view_classifer(f_view_in.detach())

                    _, predict =predit_view_label.max(1)
                    view_classifier_acc = predict.eq(true_label).sum().item()/true_label.shape[0]

                    view_loss = self.criterion1(predit_view_label[:f_out_satellite.shape[0]], view_d_labels) \
                                 + self.criterion1(predit_view_label[f_out_satellite.shape[0]:f_out_satellite.shape[0] * 2],
                                                   view_i_labels) \
                                 + self.criterion1(predit_view_label[f_out_satellite.shape[0] * 2:], view_s_labels)
                    #
                    view_classifier_optimizer.zero_grad()
                    view_loss.backward()
                    view_classifier_optimizer.step()

                    predit_modal_label2 = self.net_view_classifer(f_view_in)
                    view_adv_loss = self.criterion1(predit_modal_label2[:f_out_satellite.shape[0]], view_i_labels) \
                                 + self.criterion1(predit_modal_label2[f_out_satellite.shape[0]:f_out_satellite.shape[0] * 2],
                                                   view_i_labels) \
                                 + self.criterion1(predit_modal_label2[f_out_satellite.shape[0] * 2:], view_i_labels)

                    loss_satellite = self.memory_satellite(f_out_satellite, labels_satellite)
                    loss_drone = self.memory_drone(f_out_drone, labels_drone)
                    loss = loss_satellite + loss_drone + 1.0 * view_adv_loss
                else:
                    loss_satellite = self.memory_satellite(f_out_satellite, labels_satellite)
                    loss_drone = self.memory_drone(f_out_drone, labels_drone)
                    loss = loss_satellite + loss_drone

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.update(loss.item())
            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                logger.info('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss satellite {:.3f}\t'
                      'Loss drone {:.3f}\t'
                      'Loss view {:.3f}\t'
                      'Loss view adv {:.3f}\t'
                      'view class acc {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_drone),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg, loss_satellite, loss_drone, view_loss, view_adv_loss, view_classifier_acc))

    def _parse_data_satellite(self, inputs):
        imgs, _, pids, indexes = inputs
        return imgs.to(self.device), pids.to(self.device), indexes.to(self.device)
    
    def _parse_data_drone(self, inputs):
        imgs1, imgs2, _, pids, indexes = inputs
        return imgs1.to(self.device), imgs2.to(self.device), pids.to(self.device), indexes.to(self.device)

    def _forward(self, x1, x2, label_1=None, label_2=None):
        out1 = self.encoder(x1)
        out2 = self.encoder(x2)
        return out1[-2], out2[-2], label_1, label_2
    

def train(train_config, model, dataloader, loss_functions, optimizer, epoch, train_steps_per,
          scheduler=None, scaler=None):
    # set model train mode
    model.train()

    losses = AverageMeter()

    # wait before starting progress bar
    time.sleep(0.1)

    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)

    step = 1

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    criterion = nn.CrossEntropyLoss()

    # for loop over one epoch
    for query, reference, ids, labels in bar:

        if scaler:
            with (autocast()):  # -- 使用混合精度
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
                labels = labels.to(train_config.device)

                # Forward pass
                if train_config.handcraft_model is not True:
                    features1, features2 = model(query, reference)
                else:
                    output1, output2 = model(query, reference)
                    features1, features2 = output1[-2], output2[-2]  # -- for contrastive
                    features_tri_1, features_tri_2 = output1[2], output2[2]  # -- for triplet
                    features_cls_1, features_cls_2 = output1[1], output2[1]  # -- for classifier
                    features_fine_1, features_fine_2 = output1[-1], output2[-1]  # -- for fine-grained
                    features_dsa_1, features_dsa_2 = output1[0], output2[0]  # -- for DSA loss

                    # 定义一个池化层，目标输出尺寸为 8x8
                    pool = nn.AdaptiveAvgPool2d((8, 8))

                    # 将两个特征图都池化到相同尺寸
                    features_dsa_1_pooled = pool(features_dsa_1)
                    features_dsa_2_pooled = pool(features_dsa_2)

                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                    loss = loss_functions["infoNCE"](features1, features2, model.module.logit_scale.exp())
                    # 2. Classification
                    loss_cls = cal_loss(features_cls_1, labels, criterion) + cal_loss(features_cls_2, labels, criterion)

                    # pdb.set_trace()
                    # 3. Domian Space Alignment Loss
                    loss_DSA = loss_functions["DSA_loss"](features_dsa_1_pooled, features_dsa_2_pooled,
                                                            model.module.logit_scale_blocks.exp())
                else:
                    # 1. infoNCE
                    loss = loss_functions["infoNCE"](features1, features2, model.logit_scale.exp())

                    # 2. Classification
                    loss_cls = cal_loss(features_cls_1, labels, criterion) + cal_loss(features_cls_2, labels, criterion)


                    # 3. Domian Space Alignment Loss
                    loss_DSA = loss_functions["DSA_loss"](features_dsa_1, features_dsa_2,
                                                            model.logit_scale_blocks.exp())


                lossall = train_config.weight_infonce * loss + train_config.weight_cls * loss_cls + train_config.weight_dsa * loss_DSA

                # lossall = 1.0 * loss + 0.0 * loss_cls + 0.0 * loss_DSA

                losses.update(lossall.item())

            # scaler.scale(loss).backward()  # -- 混合精度好像是这样用的
            scaler.scale(lossall).backward()  # -- 这里才是反向传播，上面就是记录一下

            # Gradient clipping
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

                # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()

        else:

            # data (batches) to device   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                loss = loss_functions["infoNCE"](features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_functions["infoNCE"](features1, features2, model.logit_scale.exp())
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()

            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

                # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()

        if train_config.verbose:
            # tst = model.logit_scale
            monitor = {
                "loss": "{:.4f}".format(loss.item()),
                "loss_cls": "{:.4f}".format(train_config.weight_cls * loss_cls.item()),
                "loss_dsa": "{:.4f}".format(train_config.weight_dsa * loss_DSA.item()),
                "loss_avg": "{:.4f}".format(losses.avg),
                "lr": "{:.6f}".format(optimizer.param_groups[0]['lr'])}

            bar.set_postfix(ordered_dict=monitor)

        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def predict(train_config, model, dataloader):
    model.eval()

    # wait before starting progress bar
    time.sleep(0.1)

    # -- draw visualization for ablation study
    draw_vis = False
    if draw_vis:
        import cv2
        import os
        import numpy as np
        pic_path = "./draw_vis"
        iterations = int(len(os.listdir(pic_path)) / 2)
        for i in range(iterations):
            uav_ori = cv2.imread(rf"{pic_path}/{i}_uav.jpg")
            sat_ori = cv2.imread(rf"{pic_path}/{i}_sat.jpg")

            uav_shape = uav_ori.shape[:-1]
            sat_shape = sat_ori.shape[:-1]

            uav = cv2.resize(uav_ori, (384, 384), interpolation=cv2.INTER_LINEAR).astype('float32') / 255.0
            sat = cv2.resize(sat_ori, (384, 384), interpolation=cv2.INTER_LINEAR).astype('float32') / 255.0

            uav = torch.tensor(uav).permute(2, 0, 1)
            sat = torch.tensor(sat).permute(2, 0, 1)

            # 图像标准化
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            import torchvision.transforms as transforms
            normalize = transforms.Normalize(mean=mean, std=std)
            uav = normalize(uav)[None, :, :, :]
            sat = normalize(sat)[None, :, :, :]

            with torch.no_grad():
                with autocast():
                    uav = uav.to(train_config.device)
                    sat = sat.to(train_config.device)

                    img_feature_uav = model(uav)[1]
                    img_feature_uav = F.normalize(img_feature_uav, dim=1)

                    img_feature_sat = model(sat)[1]
                    img_feature_sat = F.normalize(img_feature_sat, dim=1)

                    heat_map_uav = img_feature_uav[0].permute(1, 2, 0)
                    heat_map_uav = torch.mean(heat_map_uav, dim=2).detach().cpu().numpy()
                    heat_map_uav = (heat_map_uav - heat_map_uav.min()) / (heat_map_uav.max() - heat_map_uav.min())
                    heat_map_uav = cv2.resize(heat_map_uav, [uav_shape[1], uav_shape[0]])

                    heat_map_sat = img_feature_sat[0].permute(1, 2, 0)
                    heat_map_sat = torch.mean(heat_map_sat, dim=2).detach().cpu().numpy()
                    heat_map_sat = (heat_map_sat - heat_map_sat.min()) / (heat_map_sat.max() - heat_map_sat.min())
                    heat_map_sat = cv2.resize(heat_map_sat, [sat_shape[1], sat_shape[0]])

                    #  colorize
                    colored_image_uav = cv2.applyColorMap((heat_map_uav * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    colored_image_sat = cv2.applyColorMap((heat_map_sat * 255).astype(np.uint8), cv2.COLORMAP_JET)

                    # 设置半透明度（alpha值）
                    alpha = 0.5
                    # 将两个图像进行叠加
                    blended_image_uav = cv2.addWeighted(uav_ori, alpha, colored_image_uav, 1 - alpha, 0)
                    blended_image_sat = cv2.addWeighted(sat_ori, alpha, colored_image_sat, 1 - alpha, 0)


                    out_path = r"/data1/chenqi/DAC/DSA_off"
                    cv2.imwrite(rf"{out_path}/{i}_uav_vis.jpg", blended_image_uav)
                    cv2.imwrite(rf"{out_path}/{i}_sat_vis.jpg", blended_image_sat)

        return 0

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    img_features_list = []

    ids_list = []
    with torch.no_grad():

        for img, ids in bar:

            ids_list.append(ids)

            with autocast():
                img = img.to(train_config.device)

                if train_config.handcraft_model is not True:
                    img_feature = model(img)
                else:
                    img_feature = model(img)[-2]

                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)

            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))

        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0)
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)

    if train_config.verbose:
        bar.close()

    return img_features, ids_list
