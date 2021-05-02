#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: JamieJ
# @license: Apache Licence 
# @file: GATLayer.py 
# @time: 2021/05/01
# @contact: mingj@kth,se
# @software: PyCharm 
# May the Force be with you.

from Data import *
import numpy as np
from scipy.sparse import coo_matrix
from scipy import sparse
from torch.nn import Module
import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GATLayer(Module):
    # TODO: add multi-head attention
    def __init__(self, in_dim, out_dim, dropout, alpha):
        super(GATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.alpha = alpha  # activation param of leakyReLU, set as 0.2 according to the paper

        # parameters W,  (F x F')
        self.W = nn.Parameter(torch.zeros(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # separate a (2F' x 1) into two part, (F' x 1) for each
        self.self_attn_kernel = nn.Parameter(torch.zeros(size=(out_dim, 1)))
        nn.init.xavier_uniform_(self.self_attn_kernel.data, gain=1.414)

        self.neighbours_attn_kernel = nn.Parameter(torch.zeros(size=(out_dim, 1)))
        nn.init.xavier_uniform_(self.neighbours_attn_kernel.data, gain=1.414)

        self.attn_kernels = [self.self_attn_kernel, self.neighbours_attn_kernel]

        self.leakyrelu = nn.LeakyReLU(self.alpha)  # 0.2

    def forward(self, features, A):
        """
        :param feartures: (N x F) => N = uN+iN
        :param A: adjacency matrix (N, N)
        """

        # equation 1 in the paper.
        # z_i = w h_i
        z = torch.mm(features, self.W)  # (N x F')

        # equation 2
        # e_ij = leakyReLU(a^T (z_i || z_j)) = [a_0 || a_1]^T [z_i || z_j]
        # = [a_0]^T [z_i] + [a_1]^T [z_j]
        attention_kernels = self.attn_kernels
        attn_for_self = torch.mm(z, attention_kernels[0]) # (N x 1), [a_0]^T [z_i]
        attn_for_neighbours = torch.mm(z, attention_kernels[1]) # (N x 2), [a_1]^T [z_j]

        e = attn_for_self + attn_for_neighbours.t()
        e = self.leakyrelu(e)

        # sparse matrix cannot do A > 0...\ Orzzzzzz
        A = A.to_dense()
        A_ = torch.where(A > 0, 1.0, 0.0)

        # mask the unlinked edge to -inf
        mask = -1e12 * (1.0 - A_)
        attention = e + mask

        # equation 3: softmax
        attention = F.softmax(attention, dim=1)  # [N, N]

        attention = F.dropout(attention, self.dropout, training=self.training)

        # equation 4: h = a z
        h = torch.matmul(attention, z)  # [N, N].[N, out_dim] => [N, out_dim]
        return h

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim) + ')'


def getFeatureMat(userNum, itemNum, embedSize):
    uidx = torch.LongTensor([i for i in range(userNum)])
    iidx = torch.LongTensor([i for i in range(itemNum)])
    uidx = uidx.to(device)
    iidx = iidx.to(device)
    uEmbd = nn.Embedding(userNum, embedSize)
    iEmbd = nn.Embedding(itemNum, embedSize)
    userEmbd = uEmbd(uidx)
    itemEmbd = iEmbd(iidx)
    features = torch.cat([userEmbd, itemEmbd], dim=0)
    return features


def buildAdjacencyMat(rt, userNum, itemNum):
    rt_item = rt.iId + userNum
    uiMat = coo_matrix((rt.rt, (rt.uId, rt.iId)))
    uiMat_upperPart = coo_matrix((rt.rt, (rt.uId, rt_item)))
    uiMat = uiMat.transpose()
    uiMat.resize((itemNum, userNum + itemNum))
    A = sparse.vstack([uiMat_upperPart, uiMat])
    # selfLoop = sparse.eye(userNum + itemNum)
    L = A
    L = sparse.coo_matrix(L)
    row = L.row
    col = L.col
    i = torch.LongTensor([row, col])
    data = torch.FloatTensor(L.data)
    SparseL = torch.sparse.FloatTensor(i, data)
    return SparseL


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rt = load_data()
    userNum = rt['reviewerID'].max() + 1
    itemNum = rt['asin'].max() + 1
    ds = DM(rt)
    embedSize, out_dim = 10, 5
    features = getFeatureMat(userNum, itemNum, embedSize)
    A = buildAdjacencyMat(ds, userNum, itemNum)
    gat = GATLayer(embedSize, out_dim, 0.2, 0.2)
    outputs = gat(features, A)
    print(outputs.shape)

