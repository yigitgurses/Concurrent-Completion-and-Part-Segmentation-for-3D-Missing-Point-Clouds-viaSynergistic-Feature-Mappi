# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-07-04 11:01:37
# @Email:  cshzxie@gmail.com

import logging
import os
import torch

import utils.data_loaders
import utils.helpers
import data_utils as d_utils

from datetime import datetime
from time import time
from tensorboardX import SummaryWriter

from core.test import test_net
from core.test import test_net_new
from extensions.chamfer_dist import ChamferDistance
from extensions.gridding_loss import GriddingLoss
from models.grnet import GRNet
from utils.average_meter import AverageMeter
from utils.metrics import Metrics
import shapenet_part_loader

import time as t2
import random
random.seed(123546)
import sys

from extensions.gridding import Gridding, GriddingReverse

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np

draw_mode = False
train_seg = False

save_crop_mode = False

seg_class_no = 4

def distance_squre(p1,p2):
    tensor=torch.FloatTensor(p1)-torch.FloatTensor(p2)
    val=tensor.mul(tensor)
    val = val.sum()
    return val

def get_seg_gts(segs, gt_clouds, pred_clouds):
    gt_segs = torch.zeros(pred_clouds.size()[:2]).long().to(segs.device)
    for i, p in enumerate(pred_clouds):
        gt_p = gt_clouds[i]
        gt_segs[i] = get_seg_gt(segs[i], gt_p, p)
    return gt_segs

def get_seg_gt(seg, gt_cloud, pred_cloud):
    min_indices = torch.zeros(pred_cloud.size()[0]).long().to(seg.device)

    for p_idx, p in enumerate(pred_cloud):
        distances = torch.sum(((p - gt_cloud)**2), dim=1)
        min_idx = torch.argmin(distances)
        min_indices[p_idx] = min_idx

    gt_seg = seg[min_indices]
    return gt_seg

def get_data_seg(ptcloud, full_seg):
    scale  = 16
    temp_cloud = torch.round((ptcloud + 1) * scale - 0.501).long()
    temp_cloud[temp_cloud == -1] = 0
    segsT = torch.transpose(full_seg, 1, 4)

    preds = []
    for i, p in enumerate(temp_cloud):
        pred = segsT[i, p[:,0], p[:,1], p[:,2]].unsqueeze(dim=0)
        preds.append(pred)

    return torch.cat(preds, dim=0).contiguous()

def plot_ptcloud(ptcloud, seg, name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(6):
        pts = (ptcloud[seg == i]).cpu().detach().numpy()
        xs = pts[:,0]
        ys = pts[:,1]
        zs = pts[:,2]
        ax.scatter(xs, ys, zs)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.savefig(name)


def train_net_new(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set up data loader
    pnum = 2048
    crop_point_num = 512
    workers = 1
    batchSize = 16

    class_name = "Pistol"

    train_dataset_loader = shapenet_part_loader.PartDataset( root='./dataset/shapenetcore_partanno_segmentation_benchmark_v0/',
        classification=False, class_choice=class_name, npoints=pnum, split='train')
    train_data_loader = torch.utils.data.DataLoader(train_dataset_loader, batch_size=batchSize,
                                            shuffle=True,num_workers = int(workers))


    test_dataset_loader = shapenet_part_loader.PartDataset( root='./dataset/shapenetcore_partanno_segmentation_benchmark_v0/',
        classification=False, class_choice=class_name, npoints=pnum, split='test')
    val_data_loader = torch.utils.data.DataLoader(test_dataset_loader, batch_size=batchSize,
                                            shuffle=True,num_workers = int(workers))

    # Set up folders for logs and checkpoints
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', datetime.now().isoformat())
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    cfg.DIR.LOGS = output_dir % 'logs'
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    # Create the networks
    grnet = GRNet(cfg, seg_class_no)
    grnet.apply(utils.helpers.init_weights)
    logging.debug('Parameters in GRNet: %d.' % utils.helpers.count_parameters(grnet))

    # Move the network to GPU if possible
    grnet = grnet.to(device)

    # Create the optimizers
    grnet_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, grnet.parameters()),
                                       lr=cfg.TRAIN.LEARNING_RATE,
                                       weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                       betas=cfg.TRAIN.BETAS)
    grnet_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(grnet_optimizer,
                                                              milestones=cfg.TRAIN.LR_MILESTONES,
                                                              gamma=cfg.TRAIN.GAMMA)

    # Set up loss functions
    chamfer_dist = ChamferDistance()
    gridding_loss = GriddingLoss(    # lgtm [py/unused-local-variable]
        scales=cfg.NETWORK.GRIDDING_LOSS_SCALES,
        alphas=cfg.NETWORK.GRIDDING_LOSS_ALPHAS)
    seg_criterion = torch.nn.CrossEntropyLoss().cuda()

    # Load pretrained model if exists
    init_epoch = 0
    best_metrics = None
    if 'WEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        grnet.load_state_dict(checkpoint['grnet'])
        logging.info('Recover complete. Current epoch = #%d; best metrics = %s.' % (init_epoch, best_metrics))

    train_seg_on_sparse = False
    train_seg_on_dense = False

    miou = 0

    # Training/Testing the network
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):
        epoch_start_time = time()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['SparseLoss', 'DenseLoss'])
        
        grnet.train()

        if epoch_idx == 5:
            train_seg_on_sparse = True

        if epoch_idx == 7:
            train_seg_on_dense = True

        batch_end_time = time()
        n_batches = len(train_data_loader)
        for batch_idx, (data, seg, model_ids,) in enumerate(train_data_loader):
            data_time.update(time() - batch_end_time)

            input_cropped1 = torch.FloatTensor(data.size()[0], pnum, 3)
            input_cropped1 = input_cropped1.data.copy_(data)

            if batch_idx == 10:
                pass #break

            data = data.to(device)
            seg = seg.to(device)
            
            input_cropped1 = input_cropped1.to(device)

            # remove points to make input incomplete
            choice = [torch.Tensor([1,0,0]),torch.Tensor([0,0,1]),torch.Tensor([1,0,1]),torch.Tensor([-1,0,0]),torch.Tensor([-1,1,0])]
            for m in range(data.size()[0]):
                index = random.sample(choice,1)
                p_center = index[0].to(device)
                distances = torch.sum((data[m] - p_center)**2, dim=1)
                order = torch.argsort(distances)
                
                zero_point = torch.FloatTensor([0,0,0]).to(device)
                input_cropped1.data[m, order[:crop_point_num]] = zero_point

            if save_crop_mode:
                np.save(class_name + "_orig", data[0].detach().cpu().numpy())
                np.save(class_name + "_cropped", input_cropped1[0].detach().cpu().numpy())
                sys.exit()

            sparse_ptcloud, dense_ptcloud, sparse_seg, full_seg, dense_seg = grnet(input_cropped1)

            
            data_seg = get_data_seg(data, full_seg)
            seg_loss = seg_criterion(torch.transpose(data_seg,1,2), seg)
            if train_seg_on_sparse and train_seg:
                gt_seg = get_seg_gts(seg, data, sparse_ptcloud)
                seg_loss += seg_criterion(torch.transpose(sparse_seg,1,2), gt_seg)
                seg_loss /= 2
            
            if train_seg_on_dense and train_seg:
                gt_seg = get_seg_gts(seg, data, dense_ptcloud)
                dense_seg_loss = seg_criterion(torch.transpose(dense_seg,1,2), gt_seg)
                print(dense_seg_loss.item())

            if draw_mode:
                plot_ptcloud(data[0], seg[0], "orig")
                plot_ptcloud(input_cropped1[0], seg[0], "cropped")
                plot_ptcloud(sparse_ptcloud[0], torch.argmax(sparse_seg[0], dim=1), "sparse_pred")
                if not train_seg_on_sparse:
                    gt_seg = get_seg_gts(seg, data, sparse_ptcloud)
                #plot_ptcloud(sparse_ptcloud[0], gt_seg[0], "sparse_gt")
                #if not train_seg_on_dense:
                    #gt_seg = get_seg_gts(seg, data, sparse_ptcloud)
                print(dense_seg.size())
                plot_ptcloud(dense_ptcloud[0], torch.argmax(dense_seg[0], dim=1), "dense_pred")
                sys.exit()
        
            print(seg_loss.item())

            lamb = 0.8
            sparse_loss = chamfer_dist(sparse_ptcloud, data).to(device)
            dense_loss = chamfer_dist(dense_ptcloud, data).to(device)
            grid_loss = gridding_loss(sparse_ptcloud, data).to(device)
            if train_seg:
                _loss = lamb * (sparse_loss + dense_loss + grid_loss) + (1-lamb) * seg_loss 
            else:
                _loss = (sparse_loss + dense_loss + grid_loss)
            if train_seg_on_dense and train_seg:
                _loss += (1 - lamb) * dense_seg_loss 
            _loss.to(device)
            losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])

            grnet.zero_grad()
            _loss.backward()
            grnet_optimizer.step()

            n_itr = (epoch_idx - 1) * n_batches + batch_idx
            train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
            train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)

            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            logging.info('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s' %
                         (epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches, batch_time.val(), data_time.val(),
                          ['%.4f' % l for l in losses.val()]))

        # Validate the current model
        if train_seg:
            miou_new = test_net_new(cfg, epoch_idx, val_data_loader, val_writer, grnet)
        else:
            miou_new = 0

        grnet_lr_scheduler.step()
        epoch_end_time = time()
        train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch_idx)
        train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch_idx)
        logging.info(
            '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
            (epoch_idx, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]))
        
        
        if not train_seg or miou_new > miou:
            file_name = class_name + 'noseg-ckpt-epoch.pth' 
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            torch.save({
                'epoch_index': epoch_idx,
                'grnet': grnet.state_dict()
            }, output_path)  # yapf: disable

            logging.info('Saved checkpoint to %s ...' % output_path)
            miou = miou_new

    train_writer.close()
    val_writer.close()


def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data loader
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TRAIN),
                                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                                    num_workers=cfg.CONST.NUM_WORKERS,
                                                    collate_fn=utils.data_loaders.collate_fn,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.VAL),
                                                  batch_size=1,
                                                  num_workers=cfg.CONST.NUM_WORKERS,
                                                  collate_fn=utils.data_loaders.collate_fn,
                                                  pin_memory=True,
                                                  shuffle=False)

    # Set up folders for logs and checkpoints
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', datetime.now().isoformat())
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    cfg.DIR.LOGS = output_dir % 'logs'
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    # Create the networks
    grnet = GRNet(cfg)
    grnet.apply(utils.helpers.init_weights)
    logging.debug('Parameters in GRNet: %d.' % utils.helpers.count_parameters(grnet))

    # Move the network to GPU if possible
    if torch.cuda.is_available():
        grnet = torch.nn.DataParallel(grnet).cuda()

    # Create the optimizers
    grnet_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, grnet.parameters()),
                                       lr=cfg.TRAIN.LEARNING_RATE,
                                       weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                       betas=cfg.TRAIN.BETAS)
    grnet_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(grnet_optimizer,
                                                              milestones=cfg.TRAIN.LR_MILESTONES,
                                                              gamma=cfg.TRAIN.GAMMA)

    # Set up loss functions
    chamfer_dist = ChamferDistance()
    gridding_loss = GriddingLoss(    # lgtm [py/unused-local-variable]
        scales=cfg.NETWORK.GRIDDING_LOSS_SCALES,
        alphas=cfg.NETWORK.GRIDDING_LOSS_ALPHAS)

    # Load pretrained model if exists
    init_epoch = 0
    best_metrics = None
    if 'WEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        best_metrics = Metrics(cfg.TEST.METRIC_NAME, checkpoint['best_metrics'])
        grnet.load_state_dict(checkpoint['grnet'])
        logging.info('Recover complete. Current epoch = #%d; best metrics = %s.' % (init_epoch, best_metrics))

    # Training/Testing the network
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):
        epoch_start_time = time()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['SparseLoss', 'DenseLoss'])

        grnet.train()

        batch_end_time = time()
        n_batches = len(train_data_loader)
        for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(train_data_loader):
            data_time.update(time() - batch_end_time)
            for k, v in data.items():
                data[k] = utils.helpers.var_or_cuda(v)

            sparse_ptcloud, dense_ptcloud = grnet(data)
            sparse_loss = chamfer_dist(sparse_ptcloud, data['gtcloud'])
            dense_loss = chamfer_dist(dense_ptcloud, data['gtcloud'])
            _loss = sparse_loss + dense_loss
            #losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])

            grnet.zero_grad()
            _loss.backward()
            grnet_optimizer.step()

            n_itr = (epoch_idx - 1) * n_batches + batch_idx
            train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
            train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)

            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            logging.info('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s' %
                         (epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches, batch_time.val(), data_time.val(),
                          ['%.4f' % l for l in losses.val()]))

        grnet_lr_scheduler.step()
        epoch_end_time = time()
        train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch_idx)
        train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch_idx)
        logging.info(
            '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
            (epoch_idx, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]))

        # Validate the current model
        metrics = test_net(cfg, epoch_idx, val_data_loader, val_writer, grnet)

        # Save ckeckpoints
        if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or metrics.better_than(best_metrics):
            file_name = 'ckpt-best.pth' if metrics.better_than(best_metrics) else 'ckpt-epoch-%03d.pth' % epoch_idx
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            torch.save({
                'epoch_index': epoch_idx,
                'best_metrics': metrics.state_dict(),
                'grnet': grnet.state_dict()
            }, output_path)  # yapf: disable

            logging.info('Saved checkpoint to %s ...' % output_path)
            if metrics.better_than(best_metrics):
                best_metrics = metrics

    train_writer.close()
    val_writer.close()
