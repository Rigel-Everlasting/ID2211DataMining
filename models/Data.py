#!/usr/bin/env python  
#-*- coding:utf-8 -*-  
# @author:Clarky Clark Wang
# @license: Apache Licence 
# @file: data.py 
# @time: 2021/04/26
# @contact: wangz@kth,se
# @software: PyCharm 
# Import Libs and Let's get started, shall we?
import pandas as pd
import gzip
import json
from torch.utils.data import Dataset
from sklearn import preprocessing

def load_data(path='dataset/Digital_Music_5.json.gz'):
    data = {}
    with gzip.open(path, 'rb') as g:
        i = 0
        for line in g:
            line = json.loads(line)
            # we only need product, reviewer ID and rating
            keys = ['reviewerID', 'asin', 'overall', 'unixReviewTime']
            line['overall'] = int(line['overall'])
            l = {k: line[k] for k in keys}
            data[i] = l
            i += 1
    data = pd.DataFrame.from_dict(data, orient='index')
    le_review = preprocessing.LabelEncoder()
    le_review.fit(data['reviewerID'])
    data['reviewerID'] = le_review.transform(data['reviewerID'])
    le_asin = preprocessing.LabelEncoder()
    le_asin.fit(data['asin'])
    data['asin'] = le_asin.transform(data['asin'])
    data = data.sort_values(by='unixReviewTime', ascending=True)
    return data

class DM(Dataset):

    def __init__(self, rt):
        super(Dataset, self).__init__()
        self.uId = list(rt['reviewerID'])
        self.iId = list(rt['asin'])
        self.rt = list(rt['overall'])

    def __len__(self):
        return len(self.uId)

    def __getitem__(self, item):
        return (self.uId[item],self.iId[item],self.rt[item])
