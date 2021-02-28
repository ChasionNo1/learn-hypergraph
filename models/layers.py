import math
import copy
import torch
import time
from torch import nn
from torch.nn.parameter import Parameter
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from utils.layer_utils import sample_ids, sample_ids_v2, cos_dis


class Transform(nn.Module):
    """
    A Vertex Transformation module  顶点转换模块
    置换不变变换
    Permutation invariant transformation: (N, k, d) -> (N, k, d)
    N:batch大小
    k: k个邻域
    d:特征维数
    """
    def __init__(self, dim_in, k):
        """
        :param dim_in: input feature dimension
        :param k: k neighbors
        """
        super().__init__()
        # 一维卷积：
        # k是输入通道数，有k个邻域
        # k*k是输出通道数，需要k*k个一维卷积
        # dim_in是卷积核大小，输入特征维数
        self.convKK = nn.Conv1d(k, k * k, dim_in, groups=k)
        self.activation = nn.Softmax(dim=-1)
        self.dp = nn.Dropout()

    def forward(self, region_feats):
        """
        :param region_feats: (N, k, d) 区域特征
        :return: (N, k, d)
        """
        N, k, _ = region_feats.size()  # (N, k, d)
        # 一维卷积后特征维度为1
        conved = self.convKK(region_feats)  # (N, k*k, 1)
        # view相当于reshape,
        multiplier = conved.view(N, k, k)  # (N, k, k)
        # 得到转换矩阵
        multiplier = self.activation(multiplier)  # softmax along last dimension
        # 得到转换矩阵与原始特征的乘积
        transformed_feats = torch.matmul(multiplier, region_feats)  # (N, k, d)
        return transformed_feats


class VertexConv(nn.Module):
    """
    A Vertex Convolution layer
    Transform (N, k, d) feature to (N, d) feature by transform matrix and 1-D convolution
    """
    def __init__(self, dim_in, k):
        """
        :param dim_in: input feature dimension
        :param k: k neighbors
        """
        super().__init__()
        # 转换矩阵
        self.trans = Transform(dim_in, k)                   # (N, k, d) -> (N, k, d)
        self.convK1 = nn.Conv1d(k, 1, 1)                    # (N, k, d) -> (N, 1, d)

    def forward(self, region_feats):
        """
        :param region_feats: (N, k, d)
        :return: (N, d)
        """
        # 变换矩阵
        transformed_feats = self.trans(region_feats)
        # 一维卷积
        pooled_feats = self.convK1(transformed_feats)             # (N, 1, d)
        # 压缩为(N,d),去掉维度值为1的维度
        pooled_feats = pooled_feats.squeeze(1)
        return pooled_feats


class GraphConvolution(nn.Module):
    """
    A GCN layer
    """
    def __init__(self, **kwargs):
        """
        :param kwargs:
        # dim_in,
        # dim_out,
        # dropout_rate=0.5,
        # activation
        """
        super().__init__()

        self.dim_in = kwargs['dim_in']
        self.dim_out = kwargs['dim_out']
        self.fc = nn.Linear(self.dim_in, self.dim_out, bias=kwargs['has_bias'])
        self.dropout = nn.Dropout(p=0.5)
        self.activation = kwargs['activation']

    def _region_aggregate(self, feats, edge_dict):
        N = feats.size()[0]
        pooled_feats = torch.stack([torch.mean(feats[edge_dict[i]], dim=0) for i in range(N)])

        return pooled_feats

    def forward(self, ids, feats, edge_dict, G, ite):
        """
        :param ids: compatible with `MultiClusterConvolution`
        :param feats:
        :param edge_dict:
        :return:
        """
        x = feats  # (N, d)
        x = self.dropout(self.activation(self.fc(x)))  # (N, d')
        x = self._region_aggregate(x, edge_dict)  # (N, d)
        return x


class EdgeConv(nn.Module):
    """
    A Hyperedge Convolution layer
    自注意力聚合超边
    Using self-attention to aggregate hyperedges
    """
    def __init__(self, dim_ft, hidden):
        """
        :param dim_ft: feature dimension
        :param hidden: number of hidden layer neurons
        """
        super().__init__()
        # nn.Linear:线性变换层，对输入数据做线性变换：y=Ax+b
        # 参数：
        # in_features - 每个输入样本的大小
        # out_features - 每个输出样本的大小
        # bias - 若设置为False，这层不会学习偏置。默认值：True
        self.fc = nn.Sequential(nn.Linear(dim_ft, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, ft):
        """
        use self attention coefficient to compute weighted average on dim=-2
        :param ft (N, t, d)
        :return: y (N, d)
        """
        scores = []
        # 超边个数，也就是参数t
        n_edges = ft.size(1)
        for i in range(n_edges):
            # 对ft中，每个超边进行MLP计算，得到一个分数
            scores.append(self.fc(ft[:, i]))
        # 将分数权重进行softmax，所有边的权重和为1
        scores = torch.softmax(torch.stack(scores, 1), 1)
        # 得分与特征乘积再求和
        return (scores * ft).sum(1)


class DHGLayer(GraphConvolution):
    """
    A Dynamic Hypergraph Convolution Layer
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ks = kwargs['structured_neighbor'] # number of sampled nodes in graph adjacency
        self.n_cluster = kwargs['n_cluster']              # number of clusters
        self.n_center = kwargs['n_center']                # a node has #n_center adjacent clusters
        self.kn = kwargs['nearest_neighbor']    # number of the 'k' in k-NN
        self.kc = kwargs['cluster_neighbor']    # number of sampled nodes in a adjacent k-means cluster
        self.wu_knn=kwargs['wu_knn']
        self.wu_kmeans=kwargs['wu_kmeans']
        self.wu_struct=kwargs['wu_struct']
        self.vc_sn = VertexConv(self.dim_in, self.ks+self.kn)    # structured trans
        self.vc_s = VertexConv(self.dim_in, self.ks)    # structured trans
        self.vc_n = VertexConv(self.dim_in, self.kn)    # nearest trans
        self.vc_c = VertexConv(self.dim_in, self.kc)   # k-means cluster trans
        self.ec = EdgeConv(self.dim_in, hidden=self.dim_in//4)
        self.kmeans = None
        self.structure = None

    def _vertex_conv(self, func, x):
        return func(x)

    def _structure_select(self, ids, feats, edge_dict):
        """
        :param ids: indices selected during train/valid/test, torch.LongTensor
        :param feats:
        :param edge_dict: torch.LongTensor
        :return: mapped graph neighbors
        """
        if self.structure is None:
            _N = feats.size(0)
            idx = torch.LongTensor([sample_ids(edge_dict[i], self.ks) for i in range(_N)])    # (_N, ks)
            self.structure = idx
        else:
            idx = self.structure

        idx = idx[ids]
        N = idx.size(0)
        d = feats.size(1)
        region_feats = feats[idx.view(-1)].view(N, self.ks, d)          # (N, ks, d)
        return region_feats

    def _nearest_select(self, ids, feats):
        """
        :param ids: indices selected during train/valid/test, torch.LongTensor
        :param feats:
        :return: mapped nearest neighbors
        """
        dis = cos_dis(feats)
        _, idx = torch.topk(dis, self.kn, dim=1)
        idx = idx[ids]
        N = len(idx)
        d = feats.size(1)
        nearest_feature = feats[idx.view(-1)].view(N, self.kn, d)         # (N, kn, d)
        return nearest_feature

    def _cluster_select(self, ids, feats):
        """
        compute k-means centers and cluster labels of each node
        return top #n_cluster nearest cluster transformed features
        :param ids: indices selected during train/valid/test, torch.LongTensor
        :param feats:
        :return: top #n_cluster nearest cluster mapped features
        """
        if self.kmeans is None:
            _N = feats.size(0)
            np_feats = feats.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=self.n_cluster, random_state=0, n_jobs=-1).fit(np_feats)
            centers = kmeans.cluster_centers_
            dis = euclidean_distances(np_feats, centers)
            _, cluster_center_dict = torch.topk(torch.Tensor(dis), self.n_center, largest=False)
            cluster_center_dict = cluster_center_dict.numpy()
            point_labels = kmeans.labels_
            point_in_which_cluster = [np.where(point_labels == i)[0] for i in range(self.n_cluster)]
            idx = torch.LongTensor([[sample_ids_v2(point_in_which_cluster[cluster_center_dict[point][i]], self.kc)   
                        for i in range(self.n_center)] for point in range(_N)])    # (_N, n_center, kc)
            self.kmeans = idx
        else:
            idx = self.kmeans
        
        idx = idx[ids]
        N = idx.size(0)
        d = feats.size(1)
        cluster_feats = feats[idx.view(-1)].view(N, self.n_center, self.kc, d)

        return cluster_feats                    # (N, n_center, kc, d)

    def _edge_conv(self, x):
        return self.ec(x)

    def _fc(self, x):
        return self.activation(self.fc(self.dropout(x)))

    def forward(self, ids, feats, edge_dict, G, ite):
        hyperedges = []    
        if ite >= self.wu_kmeans:
            c_feat = self._cluster_select(ids, feats)
            for c_idx in range(c_feat.size(1)):
                xc = self._vertex_conv(self.vc_c, c_feat[:, c_idx, :, :])
                xc  = xc.view(len(ids), 1, feats.size(1))               # (N, 1, d)          
                hyperedges.append(xc)
        if ite >= self.wu_knn:
            n_feat = self._nearest_select(ids, feats)
            xn = self._vertex_conv(self.vc_n, n_feat)
            xn  = xn.view(len(ids), 1, feats.size(1))                   # (N, 1, d)
            hyperedges.append(xn)
        if ite >= self.wu_struct:
            s_feat = self._structure_select(ids, feats, edge_dict)
            xs = self._vertex_conv(self.vc_s, s_feat)
            xs  = xs.view(len(ids), 1, feats.size(1))                   # (N, 1, d)
            hyperedges.append(xs)
        x = torch.cat(hyperedges, dim=1)
        x = self._edge_conv(x)                                          # (N, d)
        x = self._fc(x)                                                 # (N, d')
        return x


class HGNN_conv(nn.Module):
    """
    A HGNN layer
    """
    def __init__(self, **kwargs):
        super(HGNN_conv, self).__init__()

        self.dim_in = kwargs['dim_in']
        self.dim_out = kwargs['dim_out']
        self.fc = nn.Linear(self.dim_in, self.dim_out, bias=kwargs['has_bias'])
        self.dropout = nn.Dropout(p=0.5)
        self.activation = kwargs['activation']

    def forward(self, ids, feats, edge_dict, G, ite):
        x = feats
        x = self.activation(self.fc(x))
        x = G.matmul(x)
        x = self.dropout(x)
        return x
