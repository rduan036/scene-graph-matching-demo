#######################################################################################################################
# This is the demo code for the submitted TGRS paper "A Scene Graph Encoding and Matching Network for UAV Visual Localization"
# Author: Dr. Ran Duan, LSGI, PolyU, HK
# Contct: rduan@polyu.edu.hk
#######################################################################################################################
import torch
from torch import nn
from sceneGraphEncodingNet.non_local_dot_product import NONLocalBlock2D
import torch.nn.functional as F
import numpy as np

class CSMG(nn.Module):
    def __init__(self, input_channel=256, num_clusters=4, alpha=1.0):
        super(CSMG, self).__init__()
        self.num_clusters = num_clusters
        # using non-locak block for semi-global feature learning
        self.nl_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=256, kernel_size=3, stride=1, padding=1),
            NONLocalBlock2D(in_channels=256),
            nn.MaxPool2d(2, 2),
        )
        # cluster similarity masking graph
        self.alpha = alpha
        self.conv_node = nn.Conv2d(256, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, 256))
        self.conv_node.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv_node.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.nl_conv(x)
        B, D = x.shape[:2]
        x = F.normalize(x, p=2, dim=1)  # across descriptor dim
        centroids_norm = F.normalize(self.centroids, p=2, dim=1)
        cluster_c = centroids_norm.unsqueeze(0).expand(B, -1, -1)
        soft_assign = self.conv_node(x).view(B, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        x_flatten = x.view(B, D, -1)
        x_expand = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3)
        sim_scores = torch.bmm(cluster_c, x_flatten)
        sim_scores = self.relu(sim_scores)
        sim_scores_mask = sim_scores.expand(D, -1, -1, -1).permute(1, 2, 0, 3)
        X_star = x_expand * soft_assign.unsqueeze(2) * sim_scores_mask
        d = X_star.sum(dim=-1)
        d = F.normalize(d, p=2, dim=2)  # intra-normalization
        d_flatten = d.view(B, -1)  # flatten
        d_flatten = F.normalize(d_flatten, p=2, dim=1)  # L2 normalize
        return sim_scores, d, d_flatten

class JointNet(nn.Module):
    def __init__(self, backbone, CSMG):
        super(JointNet, self).__init__()
        self.backbone = backbone
        self.module = CSMG
        self.ref_feat = None

    def forward(self, x):
        x = self.backbone(x)
        sim_scores, d, d_flatten = self.module(x)
        sim_scores = sim_scores.squeeze(0)
        sim_scores = F.normalize(sim_scores, p=1, dim=1)
        sim_scores = sim_scores.cpu().detach().numpy()
        p = []
        for sim_score_by_cluster in sim_scores:
            idx_k = np.where(sim_score_by_cluster > 0)
            scores = sim_score_by_cluster[idx_k]
            idx_k = np.asarray(idx_k)
            scores = np.asarray(scores)
            point_x = idx_k // 14
            point_x = point_x @ scores
            point_y = idx_k % 14
            point_y = point_y @ scores
            p.append((point_x[0]*14, point_y[0]*14))

        output = {
            'descriptor': d,
            'descriptor_flatten': d_flatten,
            'position': p,
        }
        return output