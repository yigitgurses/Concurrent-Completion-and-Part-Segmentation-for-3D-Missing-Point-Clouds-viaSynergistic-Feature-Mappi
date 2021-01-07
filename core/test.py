# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 19:29:37
# @Email:  cshzxie@gmail.com

import logging
import torch

import utils.data_loaders
import utils.helpers

from extensions.chamfer_dist import ChamferDistance
from extensions.gridding_loss import GriddingLoss
from models.grnet import GRNet
from utils.average_meter import AverageMeter
from utils.metrics import Metrics
import shapenet_part_loader

import random
random.seed(1256)
import sys
import numpy as np

seg_class_no = 4

save_mode = True
save_name = "Pistol"

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

def miou(seg_gt, seg_pred):
    seg_gt = torch.flatten(seg_gt) * seg_class_no
    seg_pred = torch.flatten(seg_pred)
    
    categ = seg_gt + seg_pred
    conf_mat = torch.zeros((seg_class_no, seg_class_no)).long()

    diag = torch.zeros(seg_class_no).long()
    non_diag = 0

    for i in range(seg_class_no):
        for j in range(seg_class_no):
            conf_mat[i][j] = (categ[categ == (seg_class_no * i) + j]).nelement()
            if i == j:
                diag[i] = conf_mat[i][j]
            

    union = torch.sum(conf_mat, dim=0) + torch.sum(conf_mat, dim=1) - diag
    iou = diag.float() / union.float()
    miou = torch.FloatTensor(np.asarray(np.nanmean(iou.detach().cpu().numpy()), dtype='float32'))
    acc = torch.sum(diag).float() / seg_pred.nelement()
    return miou, acc


def test_net_new(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, grnet=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pnum = 2048
    crop_point_num = 512
    workers = 1
    batchSize = 16

    if test_data_loader == None:
        test_dataset_loader = shapenet_part_loader.PartDataset( root='./dataset/shapenetcore_partanno_segmentation_benchmark_v0/',
        classification=False, class_choice=save_name, npoints=pnum, split='test')
        test_data_loader = torch.utils.data.DataLoader(test_dataset_loader, batch_size=batchSize,
                                                shuffle=True,num_workers = int(workers))

    # Setup networks and initialize networks
    if grnet is None:
        grnet = GRNet(cfg, 4)

        if torch.cuda.is_available():
            grnet = grnet.to(device)

        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        grnet.load_state_dict(checkpoint['grnet'])

    # Switch models to evaluation mode
    grnet.eval()

    # Set up loss functions
    chamfer_dist = ChamferDistance()
    gridding_loss = GriddingLoss(scales=cfg.NETWORK.GRIDDING_LOSS_SCALES,
                                 alphas=cfg.NETWORK.GRIDDING_LOSS_ALPHAS)    # lgtm [py/unused-import]
    seg_criterion = torch.nn.CrossEntropyLoss().cuda()

    total_sparse_cd = 0
    total_dense_cd = 0

    total_sparse_ce = 0
    total_dense_ce = 0

    total_sparse_miou = 0
    total_dense_miou = 0

    total_sparse_acc = 0
    total_dense_acc = 0

    # Testing loop
    for batch_idx, (data, seg, model_ids,) in enumerate(test_data_loader):
        model_id = model_ids[0]

        with torch.no_grad():
            input_cropped1 = torch.FloatTensor(data.size()[0], pnum, 3)
            input_cropped1 = input_cropped1.data.copy_(data)

            if batch_idx == 200:
                pass # break

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

            sparse_ptcloud, dense_ptcloud, sparse_seg, full_seg, dense_seg = grnet(input_cropped1)

            if save_mode:
                np.save("./saved_results/original_" + save_name, data.detach().cpu().numpy())
                np.save("./saved_results/original_seg_" + save_name, seg.detach().cpu().numpy())
                np.save("./saved_results/cropped_" + save_name, input_cropped1.detach().cpu().numpy())
                np.save("./saved_results/sparse_" + save_name, sparse_ptcloud.detach().cpu().numpy())
                np.save("./saved_results/sparse_seg_" + save_name, sparse_seg.detach().cpu().numpy())
                np.save("./saved_results/dense_" + save_name, dense_ptcloud.detach().cpu().numpy())
                np.save("./saved_results/dense_seg_" + save_name, dense_seg.detach().cpu().numpy())
                sys.exit()

            total_sparse_cd += chamfer_dist(sparse_ptcloud, data).to(device)
            total_dense_cd += chamfer_dist(dense_ptcloud, data).to(device)

            sparse_seg_gt = get_seg_gts(seg, data, sparse_ptcloud)
            sparse_miou, sparse_acc = miou(torch.argmax(sparse_seg, dim=2), sparse_seg_gt)
            total_sparse_miou += sparse_miou
            total_sparse_acc += sparse_acc

            print(batch_idx)

            total_sparse_ce += seg_criterion(torch.transpose(sparse_seg,1,2), sparse_seg_gt)

            dense_seg_gt = get_seg_gts(seg, data, dense_ptcloud)
            dense_miou, dense_acc = miou(torch.argmax(dense_seg, dim=2), dense_seg_gt)
            total_dense_miou += dense_miou
            print(dense_miou)
            total_dense_acc += dense_acc
            total_dense_ce += seg_criterion(torch.transpose(dense_seg,1,2), dense_seg_gt)

    length = len(test_data_loader)
    print("sparse cd: " + str(total_sparse_cd * 1000 / length))
    print("dense cd: " + str(total_dense_cd * 1000 / length))
    print("sparse acc: " + str(total_sparse_acc / length))
    print("dense acc: " + str(total_dense_acc / length))
    print("sparse miou: " + str(total_sparse_miou / length))
    print("dense miou: " + str(total_dense_miou / length))
    print("sparse ce: " + str(total_sparse_ce / length))
    print("dense ce: " + str(total_dense_ce / length))

    return total_dense_miou / length


def test_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, grnet=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    if test_data_loader is None:
        # Set up data loader
        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetSubset.TEST),
                                                       batch_size=1,
                                                       num_workers=cfg.CONST.NUM_WORKERS,
                                                       collate_fn=utils.data_loaders.collate_fn,
                                                       pin_memory=True,
                                                       shuffle=False)

    # Setup networks and initialize networks
    if grnet is None:
        grnet = GRNet(cfg, 4)

        if torch.cuda.is_available():
            grnet = torch.nn.DataParallel(grnet).cuda()

        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        grnet.load_state_dict(checkpoint['grnet'])

    # Switch models to evaluation mode
    grnet.eval()

    # Set up loss functions
    chamfer_dist = ChamferDistance()
    gridding_loss = GriddingLoss(scales=cfg.NETWORK.GRIDDING_LOSS_SCALES,
                                 alphas=cfg.NETWORK.GRIDDING_LOSS_ALPHAS)    # lgtm [py/unused-import]

    # Testing loop
    n_samples = len(test_data_loader)
    test_losses = AverageMeter(['SparseLoss', 'DenseLoss'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()

    # Testing loop
    for model_idx, (taxonomy_id, model_id, data) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        model_id = model_id[0]

        with torch.no_grad():
            for k, v in data.items():
                data[k] = utils.helpers.var_or_cuda(v)

            sparse_ptcloud, dense_ptcloud = grnet(data)
            sparse_loss = chamfer_dist(sparse_ptcloud, data['gtcloud'])
            dense_loss = chamfer_dist(dense_ptcloud, data['gtcloud'])
            test_losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])
            _metrics = Metrics.get(dense_ptcloud, data['gtcloud'])
            test_metrics.update(_metrics)

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            if test_writer is not None and model_idx < 3:
                sparse_ptcloud = sparse_ptcloud.squeeze().cpu().numpy()
                sparse_ptcloud_img = utils.helpers.get_ptcloud_img(sparse_ptcloud)
                test_writer.add_image('Model%02d/SparseReconstruction' % model_idx, sparse_ptcloud_img, epoch_idx)
                dense_ptcloud = dense_ptcloud.squeeze().cpu().numpy()
                dense_ptcloud_img = utils.helpers.get_ptcloud_img(dense_ptcloud)
                test_writer.add_image('Model%02d/DenseReconstruction' % model_idx, dense_ptcloud_img, epoch_idx)
                gt_ptcloud = data['gtcloud'].squeeze().cpu().numpy()
                gt_ptcloud_img = utils.helpers.get_ptcloud_img(gt_ptcloud)
                test_writer.add_image('Model%02d/GroundTruth' % model_idx, gt_ptcloud_img, epoch_idx)

            logging.info('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                         (model_idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()
                                                                            ], ['%.4f' % m for m in _metrics]))

    # Print testing results
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    for metric in test_metrics.items:
        print(metric, end='\t')
    print()

    for taxonomy_id in category_metrics:
        print(taxonomy_id, end='\t')
        print(category_metrics[taxonomy_id].count(0), end='\t')
        for value in category_metrics[taxonomy_id].avg():
            print('%.4f' % value, end='\t')
        print()

    print('Overall', end='\t\t\t')
    for value in test_metrics.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(1), epoch_idx)
        for i, metric in enumerate(test_metrics.items):
            test_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch_idx)

    return Metrics(cfg.TEST.METRIC_NAME, test_metrics.avg())
