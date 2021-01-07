# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-09-06 11:35:30
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 19:20:36
# @Email:  cshzxie@gmail.com

import torch

from extensions.gridding import Gridding, GriddingReverse
from extensions.cubic_feature_sampling import CubicFeatureSampling

# determines if the model will use farthest  sampling or random sampling
useFarthestSampling = False



class RandomPointSampling(torch.nn.Module):
    def __init__(self, n_points):
        super(RandomPointSampling, self).__init__()
        self.n_points = n_points

    def forward(self, pred_cloud, partial_cloud=None):
        if partial_cloud is not None:
            pred_cloud = torch.cat([partial_cloud, pred_cloud], dim=1)

        _ptcloud = torch.split(pred_cloud, 1, dim=0)
        ptclouds = []
        for p in _ptcloud:
            non_zeros = torch.sum(p, dim=2).ne(0)
            p = p[non_zeros].unsqueeze(dim=0)
            n_pts = p.size(1)
            if n_pts < self.n_points:
                rnd_idx = torch.cat([torch.randint(0, n_pts, (self.n_points, ))])
            else:
                rnd_idx = torch.randperm(p.size(1))[:self.n_points]
            ptclouds.append(p[:, rnd_idx, :])

        return torch.cat(ptclouds, dim=0).contiguous()

class GetSegPred(torch.nn.Module):
    def __init__(self, scale):
        super(GetSegPred, self).__init__()
        self.scale = scale // 2

    def forward(self, segs, ptcloud):
        temp_cloud = torch.round((ptcloud + 1) * self.scale - 0.501).long().to(segs.device)
        temp_cloud[temp_cloud == -1] = 0
        segsT = torch.transpose(segs, 1, 4)

        preds = []
        for i, p in enumerate(temp_cloud):
            pred = segsT[i, p[:,0], p[:,1], p[:,2]].unsqueeze(dim=0)
            preds.append(pred)

        return torch.cat(preds, dim=0).contiguous()

class FarthestPointSampling(torch.nn.Module):
    def __init__(self, n_points):
        super(FarthestPointSampling, self).__init__()
        self.n_points = n_points

    def forward(self, pred_cloud, partial_cloud=None):
        if partial_cloud is not None:
            pred_cloud = torch.cat([partial_cloud, pred_cloud], dim=1)

        _ptcloud = torch.split(pred_cloud, 1, dim=0)
        ptclouds = []

        for p in _ptcloud:
            non_zeros = torch.sum(p, dim=2).ne(0)
            p = p[non_zeros].unsqueeze(dim=0)

            # if we have less points thane n_points, just take a permutation of all points (IFPS is irrelevant)
            n_pts = p.size(1)
            if n_pts < self.n_points:
                rnd_idx = torch.cat([torch.randint(0, n_pts, (self.n_points, ))])
                ptclouds.append(p[:, rnd_idx, :])
                continue

            # if there are more points than n_points, apply IFPS to select n_points
            device = p.device
            B, N, C = p.shape
            centroids = torch.zeros(B, self.n_points, dtype=torch.long).to(device)

            # initialize distances on the infinity
            distance = torch.ones(B, N).to(device) * 1e10

            # TODO figure out RAN
            # selects an initial point ?
            if True: # RAN:
                farthest = torch.randint(0, 1, (B,), dtype=torch.long).to(device)
            else:
                farthest = torch.randint(1, 2, (B,), dtype=torch.long).to(device)
            
            for i in range(self.n_points):
                centroids[:, i] = farthest
                centroid = p[0, farthest, :]
                diff = (p - centroid) ** 2
                dist = torch.sum(diff, dim=2)

                mask = dist < distance
                distance[mask] = dist[mask]
                farthest = torch.max(distance, dim=1)[1]

            res = p[:,centroids[0],:]
            ptclouds.append(res)
        return torch.cat(ptclouds, dim=0).contiguous()

class GRNet(torch.nn.Module):
    def __init__(self, cfg, seg_class_no):
        super(GRNet, self).__init__()
        self.dense_multiple = 4
        self.seg_class_no = seg_class_no
        self.gridding = Gridding(scale=32)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 256, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.fc5 = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU()
        )
        self.fc6 = torch.nn.Sequential(
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU()
        )
        self.dconv7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.dconv8 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.dconv9 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.dconv10 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(1),
            torch.nn.ReLU()
        )
        self.dconv_seg = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, seg_class_no, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(seg_class_no),
            torch.nn.ReLU(),
            torch.nn.Softmax()
        )
        self.gridding_rev = GriddingReverse(scale=32)
        if  useFarthestSampling:
            self.point_sampling = FarthestPointSampling(n_points=512)
        else:
            self.point_sampling = RandomPointSampling(n_points=512)

        self.get_seg_pred = GetSegPred(scale=32)
        self.feature_sampling = CubicFeatureSampling()
        self.fc11_offset = torch.nn.Sequential(
            torch.nn.Linear(1796, 2000),
            torch.nn.ReLU()
        )
        self.fc11_seg = torch.nn.Sequential(
            torch.nn.Linear(1792, 2000),
            torch.nn.ReLU()
        )
        self.fc12_offset = torch.nn.Sequential(
            torch.nn.Linear(2000, 1000),
            torch.nn.ReLU()
        )
        self.fc12_seg = torch.nn.Sequential(
            torch.nn.Linear(2000, 1000),
            torch.nn.ReLU()
        )
        self.fc13_offset = torch.nn.Sequential(
            torch.nn.Linear(1000, 500),
            torch.nn.ReLU()
        )
        self.fc13_seg = torch.nn.Sequential(
            torch.nn.Linear(1000, 500),
            torch.nn.ReLU()
        )
        self.fc14_offset = torch.nn.Linear(500, 3 * self.dense_multiple)
        self.fc14_seg = torch.nn.Linear(500, self.seg_class_no)

    def forward(self, data):
        partial_cloud = data
        #partial_cloud = data['partial_cloud']
        #print(partial_cloud.size())     # torch.Size([batch_size, 2048, 3])
        pt_features_64_l = self.gridding(data).view(-1, 1, 32, 32, 32)
        #print(pt_features_64_l.size())  # torch.Size([batch_size, 1, 64, 64, 64])
        pt_features_32_l = self.conv1(pt_features_64_l)
        #print(pt_features_32_l.size())  # torch.Size([batch_size, 32, 32, 32, 32])
        pt_features_16_l = self.conv2(pt_features_32_l)
        # print(pt_features_16_l.size())  # torch.Size([batch_size, 64, 16, 16, 16])
        pt_features_8_l = self.conv3(pt_features_16_l)
        # print(pt_features_8_l.size())   # torch.Size([batch_size, 128, 8, 8, 8])
        pt_features_4_l = self.conv4(pt_features_8_l)
        # print(pt_features_4_l.size())   # torch.Size([batch_size, 256, 4, 4, 4])
        features = self.fc5(pt_features_4_l.view(-1, 2048))
        # print(features.size())          # torch.Size([batch_size, 2048])
        pt_features_4_r = self.fc6(features).view(-1, 256, 2, 2, 2) + pt_features_4_l
        # print(pt_features_4_r.size())   # torch.Size([batch_size, 256, 4, 4, 4])
        pt_features_8_r = self.dconv7(pt_features_4_r) + pt_features_8_l
        # print(pt_features_8_r.size())   # torch.Size([batch_size, 128, 8, 8, 8])
        pt_features_16_r = self.dconv8(pt_features_8_r) + pt_features_16_l
        # print(pt_features_16_r.size())  # torch.Size([batch_size, 64, 16, 16, 16])
        pt_features_32_r = self.dconv9(pt_features_16_r) + pt_features_32_l
        # print(pt_features_32_r.size())  # torch.Size([batch_size, 32, 32, 32, 32])
        pt_features_64_r = self.dconv10(pt_features_32_r) + pt_features_64_l
        # print(pt_features_64_r.size())  # torch.Size([batch_size, 1, 64, 64, 64])
        sparse_cloud = self.gridding_rev(pt_features_64_r.squeeze(dim=1))
        # print(sparse_cloud.size())      # torch.Size([batch_size, 262144, 3])

        segs = self.dconv_seg(pt_features_32_r)
        sparse_cloud = self.point_sampling(sparse_cloud, partial_cloud)
        seg_preds = self.get_seg_pred(segs, sparse_cloud)

        sparse_size = sparse_cloud.size()[1]
        
        point_features_32 = self.feature_sampling(sparse_cloud, pt_features_32_r).view(-1, sparse_size, 256)
        point_features_16 = self.feature_sampling(sparse_cloud, pt_features_16_r).view(-1, sparse_size, 512)
        point_features_8 = self.feature_sampling(sparse_cloud, pt_features_8_r).view(-1, sparse_size, 1024)
        point_features = torch.cat([point_features_32, point_features_16, point_features_8, seg_preds], dim=2)

        dense_point_no = sparse_size * self.dense_multiple

        offset_features = self.fc11_offset(point_features)
        offset_features = self.fc12_offset(offset_features)
        offset_features = self.fc13_offset(offset_features)
        offset_features = self.fc14_offset(offset_features)
        point_offset = offset_features.view(-1, dense_point_no, 3)
        dense_cloud = sparse_cloud.unsqueeze(dim=2).repeat(1, 1, self.dense_multiple, 1).view(-1, dense_point_no, 3) + point_offset

        point_features_32 = self.feature_sampling(dense_cloud, pt_features_32_r).view(-1, dense_point_no, 256)
        point_features_16 = self.feature_sampling(dense_cloud, pt_features_16_r).view(-1, dense_point_no, 512)
        point_features_8 = self.feature_sampling(dense_cloud, pt_features_8_r).view(-1, dense_point_no, 1024)
        point_features = torch.cat([point_features_32, point_features_16, point_features_8], dim=2)

        seg_features = self.fc11_seg(point_features)
        seg_features = self.fc12_seg(seg_features)
        seg_features = self.fc13_seg(seg_features)
        seg_features = self.fc14_seg(seg_features)
        dense_seg = seg_features.view(-1, dense_point_no, self.seg_class_no)

        return sparse_cloud, dense_cloud, seg_preds, segs, dense_seg
