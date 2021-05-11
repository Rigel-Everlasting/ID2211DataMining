#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: JamieJ
# @license: Apache Licence
# @file: MatrixFct.py
# @time: 2021/05/10
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


class biasSVD(Module):
    # matrix factorization through the embedding layer of pytorch
    # SVD with bias on users and items
    # users bias: the rating habits of users
    # item bias: the overall rating unrelated to users
    def __init__(self, user_num, item_num, embed_size):
        super(biasSVD, self).__init__()
        self.userEmbd = nn.Embedding(user_num, embed_size)
        self.itemEmbd = nn.Embedding(item_num, embed_size)
        self.userBias = nn.Embedding(user_num, 1)
        self.itemBias = nn.Embedding(item_num, 1)

        self.overall_bias = nn.Parameter(torch.Tensor([0]))

    def forward(self, user_idx, item_idx):
        user_embedding = self.userEmbd(user_idx)
        item_embedding = self.userEmbd(item_idx)
        user_bias = self.userBias(user_idx)
        item_bias = self.itemBias(item_idx)

        bias = user_bias + item_bias + self.overall_bias

        predictions = torch.sum(torch.mul(user_embedding, item_embedding), dim=1) + bias.flatten()

        # # unbiased_rating = user_embedding item_embedding.T
        # predictions = torch.mm(user_embedding, item_embedding.T)
        # # add bias
        # predictions = predictions + user_bias + item_bias.resize(item_bias.shape[0])
        # # add overall bias
        # predictions = predictions.flatten() + self.overall_bias

        return predictions